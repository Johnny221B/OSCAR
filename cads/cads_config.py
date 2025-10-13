# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional


@dataclass
class CADSConfig:
    num_inference_steps: int
    tau1: float = 0.6
    tau2: float = 0.9
    s: float = 0.10
    psi: float = 1.0
    rescale: bool = True
    dynamic_cfg: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        assert 0.0 <= self.tau1 <= self.tau2 <= 1.0, "Require 0<=tau1<=tau2<=1"
        assert self.num_inference_steps >= 1
        assert 0.0 <= self.s <= 1.0
        assert 0.0 <= self.psi <= 1.0


__all__ = ["CADSConfig"]
