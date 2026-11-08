"""Training objectives of the four methods and the exact velocity field of a Gaussian mixture.

Convention for the flows: t in [0, 1], x0 ~ N(0, sigma0^2 I) is the noise (sigma0 = 1 unless an
experiment changes it) and x1 ~ p_data is the data, with the straight path
x_t = (1 - t) x0 + t x1 (Liu et al., 2023; Lipman et al., 2023).
Convention for DDPM (Ho et al., 2020): the data sit at step 0 and the noise at step T - 1.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def ot_coupling(x0, x1):
    """Exact optimal assignment within a minibatch with squared Euclidean cost (Tong et al., 2024).

    Returns the noise reordered so that x0[i] is paired with x1[i].
    """
    cost = torch.cdist(x0.flatten(1).float(), x1.flatten(1).float()).pow(2)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    perm = np.empty_like(cols)
    perm[cols] = rows  # data point j receives the noise point matched to it
    return x0[torch.as_tensor(perm, device=x0.device)]


def flow_matching_loss(model, x1, x0=None, coupling="independent", sigma0=1.0, t=None):
    """Regression of the velocity x1 - x0 along the straight path, equation (1) of the report.

    With coupling="ot" the noise of the minibatch is first reordered by ot_coupling (FM-OT).
    The optional t fixes the times, which the validation loss uses to compare iterations.
    """
    if x0 is None:
        x0 = sigma0 * torch.randn_like(x1)
        if coupling == "ot":
            x0 = ot_coupling(x0, x1)
    if t is None:
        t = torch.rand(x1.shape[0], device=x1.device)
    tt = t.view(-1, *([1] * (x1.dim() - 1)))
    xt = (1 - tt) * x0 + tt * x1
    return (model(xt, t) - (x1 - x0)).pow(2).mean()


class DDPMSchedule:
    """Linear variance schedule of Ho et al. (2020): beta from 1e-4 to 0.02 over T = 1000 steps."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        self.alpha_bar = torch.cumprod(1 - self.betas, 0).float().to(device)

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def time_input(self, t):
        """Network time input s in [0, 1] for an integer step t."""
        return t.float() / (self.T - 1)


def ddpm_loss(model, x_data, schedule, t=None, eps=None):
    """Noise-prediction loss L_simple of Ho et al. (2020), Algorithm 1."""
    n = x_data.shape[0]
    if t is None:
        t = torch.randint(0, schedule.T, (n,), device=x_data.device)
    ab = schedule.alpha_bar[t].view(-1, *([1] * (x_data.dim() - 1)))
    if eps is None:
        eps = torch.randn_like(x_data)
    xt = ab.sqrt() * x_data + (1 - ab).sqrt() * eps
    return (model(xt, schedule.time_input(t)) - eps).pow(2).mean()


class PairDataset:
    """Fixed (noise, sample) pairs produced by a trained flow, used by the reflow step."""

    def __init__(self, x0, x1):
        assert x0.shape == x1.shape
        self.x0, self.x1 = x0, x1

    def batch(self, batch_size, generator=None):
        idx = torch.randint(0, self.x0.shape[0], (batch_size,), generator=generator)
        return self.x0[idx], self.x1[idx]


class MixtureVelocity(torch.nn.Module):
    """Exact marginal velocity of flow matching with independent pairs for a Gaussian mixture.

    With x0 ~ N(0, s0^2 I) and x1 ~ sum_k pi_k N(mu_k, s^2 I), the point x_t = (1 - t) x0 + t x1
    follows sum_k pi_k N(t mu_k, c_t I) with c_t = t^2 s^2 + (1 - t)^2 s0^2. The posterior mean is
    E[x1 | x_t] = sum_k r_k(x_t) (mu_k + t s^2 / c_t (x_t - t mu_k)), with responsibilities r_k,
    and v*(x_t, t) = (E[x1 | x_t] - x_t) / (1 - t). No network is trained: this field isolates
    the error of the integrator from the error of learning.
    """

    def __init__(self, mixture, sigma0=1.0):
        super().__init__()
        self.mu = torch.as_tensor(mixture.means, dtype=torch.float64)
        self.log_w = torch.log(torch.as_tensor(mixture.weights, dtype=torch.float64))
        self.s2 = float(mixture.std) ** 2
        self.s02 = float(sigma0) ** 2

    def posterior_mean(self, x, t):
        x = x.double()
        c = t ** 2 * self.s2 + (1 - t) ** 2 * self.s02
        diff = x[:, None, :] - t * self.mu[None]
        logits = self.log_w[None] - diff.pow(2).sum(-1) / (2 * c)
        r = torch.softmax(logits, dim=1)
        means = self.mu[None] + (t * self.s2 / c) * diff
        return (r[:, :, None] * means).sum(1)

    def forward(self, x, s):
        t = float(s[0])
        assert t < 1.0, "the Euler grid never evaluates the field at t = 1"
        return ((self.posterior_mean(x, t) - x.double()) / (1 - t)).to(x.dtype)
