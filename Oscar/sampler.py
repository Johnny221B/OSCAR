from typing import Optional, Dict, Tuple
import torch
from .config import DiversityConfig
from .clip_wrapper import CLIPWrapper
from .volume_objective import VolumeObjective
from .base_flow import BaseFlow
from .utils import make_t_steps, sched_factor, project_partial_orth, batched_norm

class DiverseFlowSampler:
    def __init__(self, base_flow: BaseFlow, clip_wrapper: CLIPWrapper, cfg: DiversityConfig):
        self.base = base_flow
        self.clip = clip_wrapper
        self.cfg = cfg
        if cfg.device is not None:
            self.device = cfg.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vol = VolumeObjective(self.clip, cfg)

    @torch.no_grad()
    def _heun_correct(self, x, v1, v2, dt, delta):
        return x + dt * (0.5*(v1 + v2) + delta)

    def sample(self, x0: torch.Tensor, cond: Optional[Dict]=None) -> Tuple[torch.Tensor, Dict]:
        cfg = self.cfg
        device = self.device
        x = x0.to(device)
        G = x.size(0)
        t_grid = make_t_steps(cfg.num_steps, cfg.t_start, cfg.t_end).to(device)
        logs_all = {k:[] for k in ["t","logdet","loss","min_angle_deg","mean_angle_deg","gamma_eff"]}
        last_delta = torch.zeros_like(x)

        for k in range(cfg.num_steps):
            t = float(t_grid[k].item())
            t_next = float(t_grid[k+1].item())
            dt = abs(t - t_next)

            with torch.no_grad():
                v = self.base.velocity(x, t, cond)

            use_div = (k % cfg.update_every == 0)
            gamma_sched = cfg.gamma0 * sched_factor(t, cfg.t_gate, cfg.sched_shape)
            gamma_eff_scalar = 0.0

            if (gamma_sched > 0.0) and use_div:
                x_pred = (x + dt * v).detach().requires_grad_(True)
                loss, grad_pred, vlogs = self.vol.volume_loss_and_grad(x_pred)
                # g_proj = project_partial_orth(grad_pred.detach(), v.detach(), cfg.partial_ortho)
                g_proj = project_partial_orth(grad_pred.detach(), v.detach(), cfg.partial_ortho)
                beta_gate = sched_factor(
                    t,
                    cfg.t_gate if cfg.noise_use_same_gate else cfg.noise_t_gate,
                    cfg.sched_shape
                )
                beta = cfg.noise_beta0 * beta_gate
                if beta > 0.0:
                    xi = torch.randn_like(g_proj)
                    xi = project_partial_orth(xi, v.detach(), 1.0)    
                    noise_vel = ((2.0 * beta) / max(dt, 1e-8))**0.5 * xi
                    h = g_proj + noise_vel
                else:
                    h = g_proj
                # v_norm = batched_norm(v.detach())
                # g_norm = batched_norm(g_proj.detach())
                # cap = cfg.gamma_max_ratio * v_norm
                # scale = torch.minimum(torch.ones_like(cap), cap / (g_norm + 1e-12))
                # delta = (gamma_sched * scale.view(-1,1,1,1)) * g_proj.detach()
                # gamma_eff_scalar = float(gamma_sched)
                # last_delta = delta.detach()
                v_norm = batched_norm(v.detach())
                h_norm = batched_norm(h.detach())
                cap = cfg.gamma_max_ratio * v_norm
                scale = torch.minimum(torch.ones_like(cap), cap / (h_norm + 1e-12))
                
                delta = (gamma_sched * scale.view(-1,1,1,1)) * h
                gamma_eff_scalar = float(gamma_sched)
                last_delta = delta.detach()
                
                logs_all["logdet"].append(vlogs["logdet"])
                logs_all["loss"].append(vlogs["loss"])
                logs_all["min_angle_deg"].append(vlogs["min_angle_deg"])
                logs_all["mean_angle_deg"].append(vlogs["mean_angle_deg"])
            else:
                delta = 0.1*last_delta if gamma_sched>0 else torch.zeros_like(x)
                logs_all["logdet"].append(float("nan"))
                logs_all["loss"].append(float("nan"))
                logs_all["min_angle_deg"].append(float("nan"))
                logs_all["mean_angle_deg"].append(float("nan"))

            with torch.no_grad():
                x_mid = x + dt * (v + delta)
                v2 = self.base.velocity(x_mid, t_next, cond)
                x = self._heun_correct(x, v, v2, dt, delta)

            logs_all["t"].append(t)
            logs_all["gamma_eff"].append(gamma_eff_scalar)

        return x.clamp(0,1), logs_all
