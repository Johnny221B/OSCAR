from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPWrapper(nn.Module):
    """
    本地 CLIP 加载器（图像分支）：
    - OpenAI JIT: 传入 jit_path=~/.cache/clip/ViT-B-32.pt
    - open_clip : 传入 checkpoint_path=...（可选）

    公开方法：
    - encode_image_from_pixels(x, size): [B,3,H,W]、数值范围 [0,1] -> [B,D]（L2 归一化）
      * 可梯度，卷积等都在所指定的 device 上执行
    """

    def __init__(
        self,
        impl: str = "openai_clip",            # 'openai_clip' or 'open_clip'
        arch: str = "ViT-B-32",
        checkpoint_path: Optional[str] = None,
        jit_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.impl = impl

        self.model, self.D = self._load_local(impl, arch, checkpoint_path, jit_path)
        self.model.eval().to(self.device)

        # OpenAI CLIP 的 mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _load_local(self, impl, arch, checkpoint_path, jit_path):
        if impl == "openai_clip":
            if jit_path is None:
                raise ValueError("openai_clip requires `jit_path` to the local JIT .pt (e.g., ~/.cache/clip/ViT-B-32.pt)")
            m = torch.jit.load(jit_path, map_location="cpu")  # 原始 JIT 模型
            # OpenAI JIT 下 visual.output_dim 通常可取到；取不到就回退 512（ViT-B/32）
            D = getattr(getattr(m, "visual", None), "output_dim", None) or 512

            class _OpenAI(nn.Module):
                def __init__(self, core): super().__init__(); self.core = core
                def forward(self, x):     return self.core.encode_image(x)

            return _OpenAI(m), D

        elif impl == "open_clip":
            import open_clip
            model = open_clip.create_model(arch, pretrained=None)
            if checkpoint_path is None:
                raise ValueError("open_clip requires `checkpoint_path` to local weights")
            state = torch.load(checkpoint_path, map_location="cpu")
            state_dict = state.get("state_dict", state.get("model", state))
            model.load_state_dict(state_dict, strict=False)
            D = model.visual.output_dim

            class _OpenCLIP(nn.Module):
                def __init__(self, core): super().__init__(); self.core = core
                def forward(self, x):     return self.core.encode_image(x)

            return _OpenCLIP(model), D

        else:
            raise ValueError(f"Unknown CLIP impl: {impl}")

    @torch.enable_grad()  # 需要对输入图像求梯度，不能关掉 grad
    def encode_image_from_pixels(
        self,
        x: torch.Tensor,
        size: Union[int, Tuple[int, int]] = 224,
    ) -> torch.Tensor:
        """
        输入:
            x    : [B,3,H,W]，数值范围期望在 [0,1]
            size : 最终送入 CLIP 的空间尺寸（int 或 (H,W)）；默认 224（ViT-B/32）
        返回:
            phi  : [B,D] 的 L2 归一化特征（float32）
        """
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"x shape must be [B,3,H,W], got {tuple(x.shape)}")

        # 送到 CLIP 所在设备，确保数值精度足够
        x = x.to(self.device, dtype=torch.float32)

        # 尺寸对齐
        if isinstance(size, int):
            size = (size, size)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)

        # 归一化（在 x.device 上）
        mean = self.mean.to(x.device, dtype=x.dtype)
        std  = self.std.to(x.device, dtype=x.dtype)
        x = (x - mean) / std

        # 前向：OpenAI JIT / open_clip 的 encode_image 都接受 NCHW、已归一化的 float
        import torch.backends.cudnn as cudnn
        # with cudnn.flags(enabled=False, benchmark=False, deterministic=False):
        feats = self.model(x)            # [B,D] 或兼容
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats.float()

        # L2 归一化
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        return feats