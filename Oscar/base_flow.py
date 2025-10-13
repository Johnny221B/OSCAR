from typing import Optional, Dict
import torch
import torch.nn as nn

class BaseFlow(nn.Module):
    """ Base interface: implement velocity(x, t, cond)->v """
    def velocity(self, x: torch.Tensor, t_scalar: float, cond: Optional[Dict]=None) -> torch.Tensor:
        raise NotImplementedError


class DummyFlow(BaseFlow):
    """
    A toy velocity: v(x,t) = -(x - x0)*a pulling to a fixed clean image x0.
    Replace with your actual flow model that loads local weights.
    """
    def __init__(self, x_clean: torch.Tensor):
        super().__init__()
        self.register_buffer("x_clean", x_clean)

    @torch.no_grad()
    def velocity(self, x: torch.Tensor, t_scalar: float, cond: Optional[Dict]=None) -> torch.Tensor:
        a = 1.0
        return -(x - self.x_clean) * a