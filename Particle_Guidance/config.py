# -*- coding: utf-8 -*-
from dataclasses import dataclass
import torch


@dataclass
class FlowSamplerConfig:
    model_path: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    steps: int = 30
    cfg_scale: float = 5.0
    height: int = 1024
    width: int = 1024
    enable_attention_slicing: bool = False


__all__ = ["FlowSamplerConfig"]
