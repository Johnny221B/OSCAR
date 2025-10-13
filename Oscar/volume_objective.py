from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from .config import DiversityConfig
from .clip_wrapper import CLIPWrapper
from .utils import pairwise_cosine_angles

class VolumeObjective:
    def __init__(self, clip_model: CLIPWrapper, cfg: DiversityConfig):
        self.clip = clip_model
        self.cfg = cfg

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] in [0,1]（通常在 CLIP 的 device 上）
        返回: phi [B,D]
        - L2 和去中心化仍在 CLIP 设备
        - 白化（最耗显存）搬到 CPU 做，再搬回
        """
        phi = self.clip.encode_image_from_pixels(x, size=self.cfg.clip_image_size)  # [B,D] on clip.device

        if self.cfg.feature_l2norm:
            phi = F.normalize(phi, dim=-1)

        if self.cfg.feature_center:
            phi = phi - phi.mean(dim=0, keepdim=True)

        if getattr(self.cfg, "whiten", False) and (phi.size(0) >= getattr(self.cfg, "whiten_min_B", 16)):
            B, D = phi.shape
            phi_cpu = phi.detach().float().to("cpu")  # [B,D] on CPU
            cov = (phi_cpu.t() @ phi_cpu) / max(B - 1, 1)  # [D,D]
            cov = cov + self.cfg.eps_logdet * torch.eye(D, device=cov.device, dtype=cov.dtype)
            evals, evecs = torch.linalg.eigh(cov)  # on CPU
            inv_sqrt = (evecs * evals.clamp_min(1e-5).rsqrt()).mm(evecs.t())
            phi = (phi_cpu @ inv_sqrt).to(phi.device, dtype=phi.dtype)  # back to original device/dtype

        return phi


    def _kernel(self, phi: torch.Tensor) -> torch.Tensor:
        return phi @ phi.t()

    def _logdetI_tauK(self, K: torch.Tensor) -> torch.Tensor:
        B = K.size(0)
        I = torch.eye(B, device=K.device, dtype=K.dtype)
        # A：trace-scaled ε
        eps_eff = self.cfg.eps_logdet * (torch.trace(K) / max(B, 1))
        M = I + self.cfg.tau * K + eps_eff * I
        sign, logabsdet = torch.linalg.slogdet(M)
        return (sign.clamp_min(0.0) * logabsdet)


    def _leverage_weights(self, K: torch.Tensor) -> torch.Tensor:
        if self.cfg.leverage_alpha <= 0:
            return torch.ones(K.size(0), device=K.device, dtype=K.dtype)
        B = K.size(0)
        I = torch.eye(B, device=K.device, dtype=K.dtype)
        M = K + self.cfg.eps_logdet * I
        L = torch.linalg.cholesky(M)
        Minv = torch.cholesky_inverse(L)
        s = Minv.diag().clamp_min(1e-8)
        w = s.pow(self.cfg.leverage_alpha)
        w = w * (B / w.sum().clamp_min(1e-8))
        return w

    def volume_loss_and_grad(self, x_in: torch.Tensor):
        import torch
        dev = self.clip.device
        use_amp = (dev.type == "cuda")

        B = x_in.size(0)
        chunk = int(getattr(self.cfg, "clip_grad_chunk", 2))
        chunk = max(1, min(chunk, B))

        # ===== Pass A: 无梯度，整批算 phi_all =====
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            x_all = x_in.to(dev, dtype=torch.float16 if use_amp else torch.float32)
            phi_list = []
            for s in range(0, B, chunk):
                phi_list.append(self._features(x_all[s:s+chunk]))  # [bs,D]
            phi_all = torch.cat(phi_list, dim=0)  # [B,D]

        # ===== (!!) 从这里开始显式开启梯度，构建对 phi 的损失 =====
        with torch.enable_grad():
            # 用 float32 更稳（配合上面的 Cholesky）
            phi_leaf = phi_all.detach().to(torch.float32).clone().requires_grad_(True)  # [B,D]

            K = self._kernel(phi_leaf)  # [B,B]
            w = self._leverage_weights(K)
            if (w is not None) and (w.ndim == 1):
                Wsqrt = torch.diag(w.clamp_min(1e-8).sqrt())
                K = Wsqrt @ K @ Wsqrt

            # 0.5 * logdet(...) （内部在 float32 上做 Cholesky）
            logdet = 0.5 * self._logdetI_tauK(K)  # 标量 Tensor
            loss = -logdet                        # 标量 Tensor（需可导）

            # 关键断言（若失败，直接定位是哪一步没进图）
            assert phi_leaf.requires_grad, "phi_leaf must require grad"
            assert K.requires_grad,        "K must require grad (check no_grad leakage)"
            assert loss.requires_grad,     "loss must require grad (dtype/path issue?)"

            grad_phi = torch.autograd.grad(
                loss, phi_leaf, retain_graph=False, create_graph=False
            )[0].detach()  # [B,D]

        # ===== Pass B: 有梯度，逐块重算 φ(x) 做 VJP =====
        grads = []
        for s in range(0, B, chunk):
            x_chunk = (
                x_in[s:s+chunk]
                .detach()
                .to(dev, dtype=torch.float32)  # 强制 float32
                .clone()
                .requires_grad_(True)
            )
            with torch.enable_grad():         # 不用 autocast，保证数值稳定  torch.amp.autocast("cuda", enabled=use_amp)
                phi_chunk = self._features(x_chunk)         # [bs,D]，内部已是 float32
                go = grad_phi[s:s+chunk].to(phi_chunk.dtype)  # 也是 float32
                g_chunk = torch.autograd.grad(
                    phi_chunk, x_chunk, grad_outputs=go,
                    retain_graph=False, create_graph=False
                )[0]  # [bs,3,H,W]
            grads.append(g_chunk)


            # —— 用完这块就清 —— #
            del x_chunk, phi_chunk, g_chunk, go
            if use_amp and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()

        grad_x = torch.cat(grads, dim=0)

        with torch.no_grad():
            from .utils import pairwise_cosine_angles
            angles = pairwise_cosine_angles(F.normalize(phi_all, dim=-1))
            pos = angles > 0
            min_angle = angles[pos].min().item() if pos.any() else 0.0
            mean_angle = angles[pos].mean().item() if pos.any() else 0.0
            logs = dict(
                logdet=float(logdet.detach().item()),
                loss=float(loss.detach().item()),
                min_angle_deg=min_angle,
                mean_angle_deg=mean_angle,
            )

        return loss.detach(), grad_x, logs