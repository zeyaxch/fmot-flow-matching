"""Tests with known results. Run with: python -m unittest discover -s tests -v"""
import itertools
import math
import unittest

import numpy as np
import torch

from fmstudy import metrics, samplers
from fmstudy.data import get_mixture, mnist_partitions
from fmstudy.methods import DDPMSchedule, MixtureVelocity, flow_matching_loss, ot_coupling
from fmstudy.models import MLP2D
from fmstudy.utils import ROOT

MU = torch.tensor([2.0, -1.0])
SIGMA = torch.tensor([[1.0, 0.6], [0.6, 0.5]])


def gaussian_velocity(x, t):
    """Exact marginal velocity (2) between N(0, I) and N(MU, SIGMA) with independent pairs.

    x_t is Gaussian with mean t*MU and covariance C_t = (1-t)^2 I + t^2 SIGMA, and
    Cov(x1 - x0, x_t) = t SIGMA - (1-t) I, so v*(x, t) = MU + (t SIGMA - (1-t) I) C_t^{-1} (x - t MU).
    """
    t = float(t[0]) if torch.is_tensor(t) else float(t)
    eye = torch.eye(2)
    c_t = (1 - t) ** 2 * eye + t ** 2 * SIGMA
    a = (t * SIGMA - (1 - t) * eye) @ torch.linalg.inv(c_t)
    return MU + (x - t * MU) @ a.T


class ExactModel(torch.nn.Module):
    def forward(self, x, s):
        return gaussian_velocity(x, s)


class TestKnownResults(unittest.TestCase):
    def test_exact_velocity_transports_gaussian(self):
        """Heun with the exact field maps N(0, I) to N(MU, SIGMA)."""
        torch.manual_seed(0)
        x1 = samplers.heun(ExactModel(), torch.randn(200_000, 2), nfe=100)
        self.assertLess((x1.mean(0) - MU).abs().max().item(), 0.02)
        self.assertLess((torch.cov(x1.T) - SIGMA).abs().max().item(), 0.02)

    def test_one_euler_step_gives_the_mean(self):
        """With independent pairs, v*(x, 0) = E[x1] - x, so one Euler step returns E[x1]."""
        x1 = samplers.euler(ExactModel(), torch.randn(1000, 2), 1)
        self.assertLess((x1 - MU).abs().max().item(), 1e-5)

    def test_mixture_field_matches_monte_carlo(self):
        """The closed-form field of a mixture equals E[x1 - x0 | x_t] estimated by sampling."""
        mix = get_mixture("3unequal")
        field = MixtureVelocity(mix, sigma0=0.7)
        rng = np.random.default_rng(0)
        x1, _ = mix.sample(400_000, rng)
        x0 = 0.7 * rng.standard_normal(x1.shape)
        t = 0.6
        xt = (1 - t) * x0 + t * x1
        for target in ([0.0, 0.5], [0.5, -0.5]):  # points with enough samples nearby
            near = np.linalg.norm(xt - np.array(target), axis=1) < 0.05
            mc = (x1 - x0)[near].mean(0)
            exact = field(torch.tensor([target], dtype=torch.float32), torch.tensor([t])).numpy()[0]
            self.assertLess(np.abs(mc - exact).max(), 0.05)

    def test_last_euler_step_lands_on_the_posterior_mean(self):
        """x_t + (1 - t) v*(x_t, t) = E[x1 | x_t]: the last step of Euler returns the posterior mean."""
        mix = get_mixture("8gaussians")
        field = MixtureVelocity(mix, sigma0=1.0)
        x = torch.randn(500, 2, dtype=torch.float64)
        t = 0.5
        step = x + (1 - t) * field(x, torch.tensor([t]))
        self.assertLess((step - field.posterior_mean(x, t)).abs().max().item(), 1e-10)

    def test_many_euler_steps_on_the_exact_field_reach_the_modes(self):
        """With enough steps, the exact field puts almost every sample on a mode."""
        mix = get_mixture("8gaussians")
        x = samplers.euler(MixtureVelocity(mix, 1.0), torch.randn(2000, 2), 1000).numpy()
        self.assertGreater(metrics.placement_statistics(x, mix)["on_mode"], 0.95)

    def test_learned_velocity_matches_exact(self):
        """Flow matching learns the closed-form velocity (end-to-end test of the loss)."""
        torch.manual_seed(2)
        model = MLP2D(hidden=128, layers=3)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        chol = torch.linalg.cholesky(SIGMA)
        for _ in range(3000):
            x1 = MU + torch.randn(256, 2) @ chol.T
            loss = flow_matching_loss(model, x1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        errors = []
        for t in (0.1, 0.5, 0.9):
            x = t * MU + torch.randn(2000, 2)
            with torch.no_grad():
                v = model(x, torch.full((2000,), t))
            v_star = gaussian_velocity(x, t)
            errors.append(((v - v_star).norm(dim=1) / v_star.norm(dim=1)).median().item())
        self.assertLess(max(errors), 0.1)


class TestIntegrators(unittest.TestCase):
    def test_orders_of_convergence(self):
        """Euler is first order and Heun second order on dx/dt = A x."""
        a = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

        class Linear(torch.nn.Module):
            def forward(self, x, s):
                return x @ a.T

        x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        a = a.double()
        exact = x0 @ torch.linalg.matrix_exp(a).T
        err = {"euler": [(samplers.euler(Linear(), x0, n) - exact).norm().item() for n in (20, 40)],
               "heun": [(samplers.heun(Linear(), x0, n) - exact).norm().item() for n in (20, 40)]}
        self.assertAlmostEqual(math.log2(err["euler"][0] / err["euler"][1]), 1.0, delta=0.15)
        self.assertAlmostEqual(math.log2(err["heun"][0] / err["heun"][1]), 2.0, delta=0.15)

    def test_diffusion_grid_uses_every_step(self):
        """K steps give K distinct evaluations that start at T - 1, and T steps give all of them."""
        for steps in (1, 2, 3, 5, 10, 100):
            seq = samplers.diffusion_steps(1000, steps)
            self.assertEqual(len(np.unique(seq)), steps)
            self.assertEqual(seq[0], 999)
            self.assertTrue((np.diff(seq) < 0).all())
        self.assertTrue((samplers.diffusion_steps(1000, 1000) == np.arange(999, -1, -1)).all())
        self.assertEqual(list(samplers.diffusion_steps(1000, 2)), [999, 499])

    def test_ddim_with_exact_noise_predictor(self):
        """DDIM with the exact epsilon of Gaussian data recovers the data distribution."""
        sched = DDPMSchedule()
        m, s = 1.5, 0.3
        ab = sched.alpha_bar

        class ExactEps(torch.nn.Module):
            def forward(self, x, u):
                t = (u * (sched.T - 1)).round().long()
                a = ab[t].view(-1, 1)
                return (1 - a).sqrt() * (x - a.sqrt() * m) / (a * s ** 2 + 1 - a)

        torch.manual_seed(3)
        x = samplers.ddim(ExactEps(), torch.randn(100_000, 2), 100, sched)
        self.assertLess(abs(x.mean().item() - m), 0.02)
        self.assertLess(abs(x.std().item() - s), 0.02)
        one = samplers.ddim(ExactEps(), torch.randn(1000, 2), 1, sched)
        self.assertLess((one - m).abs().max().item(), 0.05)  # one step lands near the mean
        # With all 1000 steps the stochastic sampler (eta = 1) is also exact. With fewer steps
        # it shrinks the spread (std 0.27 instead of 0.30 with 100 steps), a discretisation error.
        for steps in (1000,):
            st = samplers.ddim(ExactEps(), torch.randn(50_000, 2), steps, sched, eta=1.0,
                               generator=torch.Generator().manual_seed(5))
            self.assertLess(abs(st.mean().item() - m), 0.02)
            self.assertLess(abs(st.std().item() - s), 0.02)


class TestMetricsAndCoupling(unittest.TestCase):
    def test_w1_translation(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((300, 2))
        self.assertAlmostEqual(metrics.w1_exact(a, a), 0.0, places=12)
        shift = np.array([0.3, -0.4])
        self.assertAlmostEqual(metrics.w1_exact(a, a + shift), 0.5, places=9)

    def test_ot_coupling_is_optimal(self):
        torch.manual_seed(4)
        x0, x1 = torch.randn(6, 2), torch.randn(6, 2)
        paired = ot_coupling(x0, x1)
        cost = (paired - x1).pow(2).sum().item()
        best = min(((x0[list(p)] - x1).pow(2).sum().item())
                   for p in itertools.permutations(range(6)))
        self.assertAlmostEqual(cost, best, places=5)

    def test_mode_statistics(self):
        mix = get_mixture("3unequal")
        x, _ = mix.sample(20_000, np.random.default_rng(0))
        st = metrics.mode_statistics(x, mix)
        self.assertEqual(st["modes"], 3)
        self.assertGreater(st["hq_fraction"], 0.98)
        self.assertLess(st["weight_tv"], 0.02)
        collapsed = metrics.mode_statistics(np.zeros((100, 2)), mix)
        self.assertEqual(collapsed["modes"], 0)

    def test_placement_of_points(self):
        """A midpoint between neighbouring modes is on a segment, and the centre is elsewhere."""
        mix = get_mixture("8gaussians")
        mid = 0.5 * (mix.means[0] + mix.means[4])
        st = metrics.placement_statistics(np.stack([mid, mix.means[2], np.zeros(2)]), mix)
        self.assertAlmostEqual(st["on_segment"], 1 / 3)
        self.assertAlmostEqual(st["on_mode"], 1 / 3)
        self.assertAlmostEqual(st["elsewhere"], 1 / 3)

    def test_digit_statistics(self):
        probs = np.zeros((4, 10))
        probs[0, 3] = probs[1, 3] = probs[2, 4] = 1.0
        probs[3, 3], probs[3, 4] = 0.5, 0.5
        st = metrics.digit_statistics(probs, (3, 4), reference_share=0.5)
        self.assertAlmostEqual(st["ambiguous"], 0.25)
        self.assertAlmostEqual(st["share_3"], 2 / 3)
        self.assertAlmostEqual(st["share_error"], 1 / 6)

    def test_frechet_of_identical_sets(self):
        f = np.random.default_rng(1).standard_normal((2000, 8))
        self.assertLess(abs(metrics.frechet_distance(f, f)), 1e-6)


@unittest.skipUnless((ROOT / "data" / "mnist").exists(), "MNIST has not been downloaded")
class TestPartitions(unittest.TestCase):
    def test_partitions_are_disjoint_and_complete(self):
        """Training and validation split the official training images of the two digits."""
        parts = mnist_partitions(ROOT / "data" / "mnist", [3, 4], 0.1, 0)
        n_train, n_val = len(parts["train"][0]), len(parts["validation"][0])
        self.assertEqual(n_train + n_val, 6131 + 5842)
        self.assertEqual(n_val, round(0.1 * (6131 + 5842)))
        self.assertEqual(len(parts["test"][0]), 1010 + 982)
        self.assertTrue(set(parts["train"][1].tolist()) <= {3, 4})


if __name__ == "__main__":
    unittest.main()
