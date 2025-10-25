import json
import time
import argparse
import traceback
from typing import Any, Dict, Optional, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from oscar.utils import (
    print_mem_all,
    inspect_pipe_devices,
    assert_on,
    log as _log,
    slugify as _slugify,
    resolve_model_dir as _resolve_model_dir,
    parse_concepts_spec as _parse_concepts_spec,
    build_root_out as _build_root_out,
    prompt_run_dir as _prompt_run_dir,
    project_partial_orth,                 
    batched_norm as _bn,                  
    sched_factor as time_sched_factor,   
)

# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT)'
    )

    # spec / prompt
    ap.add_argument('--spec', type=str, default=None, help='Path to JSON: {concept:[prompts...]}')
    ap.add_argument('--prompt', type=str, default=None, help='Single prompt if --spec not provided')
    ap.add_argument('--negative', type=str, default='')

    # generation
    ap.add_argument('--G', type=int, default=64)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)

    # multi guidance / seed
    ap.add_argument('--guidances', type=float, nargs='+', default=[3.0, 5.0, 7.5])
    ap.add_argument('--guidance', type=float, default=3.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444])
    ap.add_argument('--seed', type=int, default=42)

    # local model path
    ap.add_argument('--model-dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))

    # output/method
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method', type=str, default='ourmethod')

    # diversity config (merged)
    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.95)
    ap.add_argument('--t-gate', type=str, default='0.85,0.95')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2', 't1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)

    # noise timing / controls
    ap.add_argument('--noise-timing', type=str, default='early', choices=['early', 'late'])
    ap.add_argument('--eta-sde', type=float, default=1.0)
    ap.add_argument('--rho', type=float, default=0.25)
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4)

    # init orthogonalization
    ap.add_argument('--init-ortho', type=float, default=0.2)
    ap.add_argument('--init-ortho-mode', type=str, default='blend', choices=['blend','off','replace'])

    # devices
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae', type=str, default='cuda:0')
    ap.add_argument('--device-clip', type=str, default='cuda:0')
    ap.add_argument('--device-text1', type=str, default='cuda:1')
    ap.add_argument('--device-text2', type=str, default='cuda:1')
    ap.add_argument('--device-text3', type=str, default='cuda:1')

    # memory/debug
    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()


# -------------------- main --------------------

def main():
    args = parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # parse spec / prompt
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec_obj = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec_obj)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide --spec (JSON with {concept:[prompts...]}) or --prompt")

    # TE fallback to transformer if not provided
    if args.device_text1 is None:
        args.device_text1 = args.device_transformer
    if args.device_text2 is None:
        args.device_text2 = args.device_transformer
    if args.device_text3 is None:
        args.device_text3 = args.device_transformer

    # effective guidances / seeds
    guidances = args.guidances if args.guidances else [args.guidance]
    seeds = args.seeds if args.seeds else [args.seed]

    try:
        import torch.backends.cudnn as cudnn
        from diffusers import StableDiffusion3Pipeline
        from torchvision.utils import save_image  # noqa

        from oscar.config import DiversityConfig
        from oscar.clip_wrapper import CLIPWrapper
        from oscar.volume_objective import VolumeObjective

        # devices + dtype
        dev_tr  = torch.device(args.device_transformer)
        dev_vae = torch.device(args.device_vae)
        dev_clip= torch.device(args.device_clip)
        dtype   = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

        _log(f"Devices: transformer={dev_tr}, vae={dev_vae}, clip={dev_clip}", args.debug)
        _log(f"Model dir: {args.model_dir}", args.debug)
        _log(f"CLIP JIT: {args.clip_jit}", args.debug)
        print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_clip])

        # 1) load on CPU then move to devices
        model_dir = _resolve_model_dir(args.model_dir)
        _log("Loading SD3.5 (CPU) ...", args.debug)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir, torch_dtype=dtype, local_files_only=True,
        )
        pipe.set_progress_bar_config(leave=True)
        pipe = pipe.to("cpu")
        print("scheduler:", pipe.scheduler.__class__.__name__)

        _log("Moving modules to target devices ...", args.debug)
        if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
        if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        if args.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                _log(f"enable_xformers failed: {e}", args.debug)

        inspect_pipe_devices(pipe)
        if hasattr(pipe, "transformer"):    assert_on(pipe.transformer, dev_tr)
        if hasattr(pipe, "text_encoder"):   assert_on(pipe.text_encoder,   dev_tr)
        if hasattr(pipe, "text_encoder_2"): assert_on(pipe.text_encoder_2, dev_tr)
        if hasattr(pipe, "text_encoder_3"): assert_on(pipe.text_encoder_3, dev_tr)
        if hasattr(pipe, "vae"):            assert_on(pipe.vae,            dev_vae)

        # 2) CLIP & Volume objective
        _log("Loading CLIP ...", args.debug)
        clip = CLIPWrapper(
            impl="openai_clip", arch="ViT-B-32",
            jit_path=args.clip_jit, checkpoint_path=None,
            device=dev_clip if dev_clip.type=='cuda' else torch.device("cpu"),
        )
        _log("CLIP ready.", args.debug)

        t0, t1 = args.t_gate.split(',')
        cfg = DiversityConfig(
            num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
            gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
            partial_ortho=args.partial_ortho, t_gate=(float(t0), float(t1)),
            sched_shape=args.sched_shape, clip_image_size=224,
            leverage_alpha=0.5,
        )
        vol = VolumeObjective(clip, cfg)
        _log("Volume objective ready.", args.debug)

        # helpers (kept local on purpose)
        def _brownian_std_from_scheduler(ppl, i):
            sch = ppl.scheduler
            try:
                if hasattr(sch, "sigmas"):
                    s = sch.sigmas.float()
                    cur = s[i].item()
                    nxt = s[i+1].item() if i+1 < len(s) else s[i].item()
                    var = max(nxt**2 - cur**2, 0.0)
                    return float(var**0.5)
                elif hasattr(sch, "alphas_cumprod"):
                    ac = sch.alphas_cumprod.float()
                    cur = ac[i].item()
                    nxt = ac[i+1].item() if i+1 < len(ac) else ac[i].item()
                    sig2_cur = max(1.0 - cur, 0.0)
                    sig2_nxt = max(1.0 - nxt, 0.0)
                    var = max(sig2_nxt - sig2_cur, 0.0)
                    return float(var**0.5)
            except Exception:
                pass
            ts = sch.timesteps
            t_cur  = float(ts[i].item())
            t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)
            return float(dt_unit**0.5)

        def _make_orthonormal_noise_like(x: torch.Tensor) -> torch.Tensor:
            B = x.size(0)
            M = x[0].numel()
            y = torch.randn(B, M, device=x.device, dtype=torch.float32)
            Q, _ = torch.linalg.qr(y.t(), mode='reduced')
            Qb = Q.t().to(dtype=x.dtype).contiguous().view_as(x)
            return Qb

        def _inject_init_ortho(latents: torch.Tensor, strength: float, mode: str = 'blend') -> torch.Tensor:
            if strength <= 0 or mode == 'off':
                return latents
            with torch.no_grad():
                U = _make_orthonormal_noise_like(latents)
                base_norm = (latents.flatten(1).norm(dim=1) + 1e-12)
                U_norm = (U.flatten(1).norm(dim=1) + 1e-12)
                U = U * (base_norm / U_norm).view(-1, 1, 1, 1)
                if mode == 'replace':
                    out = (1.0 - strength) * latents + strength * U
                else:
                    out = latents + strength * U
            return out

        # main loop: concept × prompt × guidance × seed
        for concept, prompt_list in concept_to_prompts.items():
            base_dir, imgs_root, eval_dir = _build_root_out(project_root, args.method, concept)
            _log(f"[OUT] base={base_dir}", True)

            for prompt_text in prompt_list:
                for g in (guidances if len(guidances)>0 else [args.guidance]):
                    for sd in (seeds if len(seeds)>0 else [args.seed]):

                        state: Dict[str, Optional[torch.Tensor]] = {
                            "prev_latents_vae_cpu": None,
                            "prev_ctrl_vae_cpu":   None,
                            "prev_dt_unit":        None,
                            "prev_prev_latents_vae_cpu": None,
                            "last_logdet":         None,
                            "gamma_auto_done":     False,
                        }

                        def diversity_callback(ppl, i, t, kw):
                            ts = ppl.scheduler.timesteps
                            t_cur  = float(ts[i].item())
                            t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
                            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
                            t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
                            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

                            if i == 0 and "latents" in kw and kw["latents"] is not None:
                                kw["latents"] = _inject_init_ortho(
                                    kw["latents"],
                                    strength=float(getattr(args, "init_ortho", 0.0)),
                                    mode=str(getattr(args, "init_ortho_mode", "blend")),
                                )

                            lat = kw.get("latents")
                            if lat is None:
                                return kw

                            gamma_sched = cfg.gamma0 * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
                            if gamma_sched <= 0:
                                state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                                state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
                                state["prev_dt_unit"] = dt_unit
                                return kw

                            lat_new = lat.clone()

                            lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
                            B = lat_vae_full.size(0)
                            chunk = 2 if B >= 2 else 1

                            prev_cpu = state.get("prev_latents_vae_cpu", None)

                            for s in range(0, B, chunk):
                                e = min(B, s + chunk)
                                z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

                                with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                                    imgs_chunk = (pipe.vae.decode(z / getattr(pipe.vae.config, "scaling_factor", 1.0), return_dict=False)[0].float().clamp(-1,1) + 1.0)/2.0

                                imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                                _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                                current_logdet = float(_logs.get("logdet", 0.0))

                                last_logdet = state.get("last_logdet", None)
                                if (last_logdet is not None) and (current_logdet < last_logdet):
                                    gamma_sched = 0.5 * gamma_sched
                                state["last_logdet"] = current_logdet

                                grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

                                grad_lat = torch.autograd.grad(
                                    outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
                                    retain_graph=False, create_graph=False, allow_unused=False
                                )[0]

                                v_est = None
                                if prev_cpu is not None:
                                    total_diff = z - prev_cpu[s:e].to(dev_vae, non_blocking=True)
                                    prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
                                    prev_dt   = state.get("prev_dt_unit", None)
                                    if (prev_ctrl is not None) and (prev_dt is not None):
                                        ctrl_prev = prev_ctrl[s:e].to(dev_vae, non_blocking=True)
                                        base_move_prev = total_diff - ctrl_prev
                                        v_est = base_move_prev / max(prev_dt, 1e-8)
                                    else:
                                        v_est = total_diff / max(dt_unit, 1e-8)

                                g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
                                div_disp = g_proj * dt_unit

                                brown_std = _brownian_std_from_scheduler(ppl, i)
                                eta = float(args.eta_sde)
                                base_brown = eta * brown_std

                                rho_t = float(args.rho)

                                base_disp = (v_est * dt_unit) if v_est is not None else z
                                base_norm = _bn(base_disp)

                                import torch as _t
                                target_brown = _t.full_like(base_norm, fill_value=max(base_brown, 0.0))
                                target_snr   = _t.clamp(base_norm * max(rho_t, 0.0), min=0.0)
                                target = _t.minimum(target_brown, target_snr)

                                xi = _t.randn_like(g_proj)
                                vnorm = _bn(v_est) if v_est is not None else None
                                if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                                    xi_eff = xi
                                else:
                                    xi_eff = project_partial_orth(xi, v_est, float(args.partial_ortho))

                                xi_norm = _bn(xi_eff)
                                noise_disp = xi_eff / (xi_norm.view(-1,1,1,1) + 1e-12) * target.view(-1,1,1,1)

                                disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                                div_raw   = _bn(div_disp)
                                scale     = _t.minimum(_t.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                                delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp
                                delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                                lat_new[s:e] = lat_new[s:e] + delta_tr

                                if "ctrl_cache" not in state:
                                    state["ctrl_cache"] = []
                                state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

                                if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
                                if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
                                del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

                            kw["latents"] = lat_new

                            state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                            state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")

                            if "ctrl_cache" in state:
                                import torch as _t
                                state["prev_ctrl_vae_cpu"] = _t.cat(state["ctrl_cache"], dim=0).to("cpu")
                                del state["ctrl_cache"]
                            state["prev_dt_unit"] = dt_unit
                            return kw

                        run_dir = _prompt_run_dir(imgs_root, prompt_text, int(sd), float(g), int(args.steps))
                        os.makedirs(run_dir, exist_ok=True)
                        _log(f"[RUN] concept='{concept}' | prompt='{prompt_text}' | seed={sd} | guidance={g} | steps={args.steps} -> {run_dir}", True)

                        generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
                        generator.manual_seed(int(sd))

                        latents_out = pipe(
                            prompt=prompt_text,
                            negative_prompt=(args.negative if args.negative else None),
                            height=args.height, width=args.width,
                            num_images_per_prompt=args.G,
                            num_inference_steps=args.steps,
                            guidance_scale=float(g),
                            generator=generator,
                            callback_on_step_end=diversity_callback,
                            callback_on_step_end_tensor_inputs=["latents"],
                            output_type="latent",
                            return_dict=False,
                        )[0]

                        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                        latents_final = latents_out.to(dev_vae, non_blocking=True)
                        if args.enable_vae_tiling:
                            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
                            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

                        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                            images = (pipe.vae.decode(latents_final / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0

                        from torchvision.utils import save_image
                        for i in range(images.size(0)):
                            fname = f"{i:02d}.png"
                            save_image(images[i].cpu(), os.path.join(run_dir, fname))

                        del latents_out, latents_final, images

        _log("Done.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
