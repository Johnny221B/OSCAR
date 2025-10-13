# -*- coding: utf-8 -*-
"""
FlowSampler wraps Diffusers' Stable Diffusion 3.x pipeline and injects Particle
Guidance updates to latents at the end of each step via the pipeline's callback.

It auto-resolves ModelScope-style nested directories by searching for a
`model_index.json` up to two levels below the provided --model path.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline

from .particle_guidance import ParticleGuidance, PGConfig
from .config import FlowSamplerConfig
from .utils import resolve_model_dir


class FlowSampler:
    def __init__(self, cfg: FlowSamplerConfig, pg_cfg: PGConfig):
        self.cfg = cfg
        self.pg = ParticleGuidance(pg_cfg)

        # Resolve model dir
        self.model_dir = resolve_model_dir(Path(cfg.model_path))
        print(f"[PG] Loading SD3 pipeline from: {self.model_dir}")

        # Load pipeline (offline/local)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            str(self.model_dir),
            torch_dtype=cfg.dtype,
            local_files_only=True,
        )
        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing()
        self.pipe.to(torch.device(cfg.device))
        # We'll show our own tqdm progress; disable internal
        self.pipe.set_progress_bar_config(disable=True)

    def _callback(self, num_steps: int, pbar=None):
        pg = self.pg

        def cb(pipeline, step_index, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents")
            if latents is None:
                if pbar is not None:
                    pbar.update(1)
                return callback_kwargs
            # normalized time: early steps -> t≈1, late -> t≈0
            t_norm = float(
                max(0.0, 1.0 - (step_index + 1) / max(num_steps, 1)))
            dt = 1.0 / max(num_steps, 1)
            latents = pg.apply(latents, t_norm, dt)
            callback_kwargs["latents"] = latents
            if pbar is not None:
                pbar.update(1)
            return callback_kwargs

        return cb

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 4,
        seed: Optional[int] = None,
    ) -> List["PIL.Image.Image"]:
        from tqdm.auto import tqdm

        if seed is not None:
            g = torch.Generator(device=self.pipe.device).manual_seed(int(seed))
        else:
            g = None

        pbar = tqdm(total=self.cfg.steps, desc="[PG] sampling", leave=True)
        try:
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                height=self.cfg.height,
                width=self.cfg.width,
                num_inference_steps=self.cfg.steps,
                guidance_scale=self.cfg.cfg_scale,
                generator=g,
                callback_on_step_end=self._callback(self.cfg.steps, pbar),
                callback_on_step_end_tensor_inputs=["latents"],
                output_type="pil",
            ).images
        finally:
            pbar.close()
        return images
