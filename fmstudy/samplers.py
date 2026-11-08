"""Integrators of the flows, samplers of DDPM and the straightness of the trajectories.

A step is one evaluation of the network: Euler and both DDIM samplers use one evaluation per step,
so the number of steps is also the number of function evaluations (NFE). Heun (two evaluations
per step, Karras et al., 2022) is used only to generate the pairs of the reflow step.
"""
import numpy as np
import torch


def _velocity(model, x, t):
    return model(x, torch.full((x.shape[0],), float(t), device=x.device))


@torch.no_grad()
def euler(model, x0, steps, return_path=False):
    """Explicit Euler from t = 0 to t = 1 with equal steps: x <- x + h v(x, t)."""
    x, h = x0, 1.0 / steps
    path, slopes = [x0], []
    for k in range(steps):
        v = _velocity(model, x, k * h)
        x = x + h * v
        if return_path:
            path.append(x)
            slopes.append(v)
    return (x, torch.stack(path), torch.stack(slopes)) if return_path else x


@torch.no_grad()
def heun(model, x0, nfe):
    """Heun's second-order method with nfe / 2 steps (two evaluations per step)."""
    assert nfe % 2 == 0 and nfe >= 2, "Heun needs an even number of evaluations"
    steps = nfe // 2
    x, h = x0, 1.0 / steps
    for k in range(steps):
        t = k * h
        d1 = _velocity(model, x, t)
        d2 = _velocity(model, x + h * d1, t + h)
        x = x + 0.5 * h * (d1 + d2)
    return x


def diffusion_steps(T, steps):
    """Evenly spaced subsequence of the T training steps that starts at T - 1 ("trailing").

    Lin et al. (2024, Sec. 3.3 and Table 2): with T = 1000 and 10 steps it evaluates the network
    at 999, 899, ..., 99, and the last step returns the estimate of the data. A grid that also
    contains step 0 would spend its last evaluation on a step of size beta_1, so that two steps
    would act as one. With steps = T it is the full sequence T - 1, ..., 0.
    """
    return (np.round(np.arange(T, 0, -T / steps)) - 1).astype(int)


@torch.no_grad()
def ddim(model, noise, steps, schedule, eta=0.0, generator=None):
    """Generalised DDIM sampler of Song et al. (2021), equations (12) and (16).

    x_prev = sqrt(ab_prev) x0_hat + sqrt(1 - ab_prev - sigma^2) eps_hat + sigma z, with
    sigma = eta sqrt((1 - ab_prev) / (1 - ab_t)) sqrt(1 - ab_t / ab_prev).
    eta = 0 gives the deterministic DDIM. eta = 1 gives the stochastic sampler of DDPM on the
    chosen subsequence, which with all T steps is Algorithm 2 of Ho et al. (2020) with the
    posterior variance sigma_t^2 = beta_tilde_t (their Section 3.2). The last step returns x0_hat.
    """
    T = schedule.T
    alpha_bar = schedule.alpha_bar.detach().cpu().double().numpy()
    seq = diffusion_steps(T, steps)
    x = noise
    for i, t in enumerate(seq):
        ab_t = float(alpha_bar[t])
        ab_prev = float(alpha_bar[seq[i + 1]]) if i + 1 < len(seq) else 1.0
        s = torch.full((x.shape[0],), t / (T - 1), device=x.device)
        eps = model(x, s)
        x0_hat = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
        sigma = eta * np.sqrt((1 - ab_prev) / (1 - ab_t)) * np.sqrt(max(1 - ab_t / ab_prev, 0.0))
        x = np.sqrt(ab_prev) * x0_hat + np.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
        if sigma > 0:
            z = torch.randn(x.shape, generator=generator).to(x.device)
            x = x + sigma * z
    return x


@torch.no_grad()
def straightness(model, x0, steps=100):
    """Straightness S of Liu et al. (2023) and relative path length, along Euler paths.

    S = int_0^1 E || (x(1) - x(0)) - v(x(t), t) ||^2 dt is approximated by the mean over the
    Euler steps. S = 0 means straight trajectories travelled at constant speed. The length ratio
    divides the length of the path by the length of the straight segment between its ends.
    """
    x1, path, slopes = euler(model, x0, steps, return_path=True)
    disp = x1 - x0
    s_value = (disp[None] - slopes).flatten(2).pow(2).sum(-1).mean().item()
    seg = (path[1:] - path[:-1]).flatten(2).norm(dim=-1).sum(0)
    ratio = (seg / disp.flatten(1).norm(dim=-1).clamp_min(1e-12)).mean().item()
    return s_value, ratio
