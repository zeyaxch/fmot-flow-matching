"""Evaluation metrics.

2D: exact empirical W1 (assignment problem, Peyré and Cuturi, 2019), mode coverage and fraction
of high-quality samples within 3 sigma (Metz et al., 2017; Srivastava et al., 2017), the error in
the mode weights and our classification of the remaining samples: on a segment between two
neighbouring modes or elsewhere. MNIST: Fréchet distance (Heusel et al., 2017) on the features of
our own classifier, the proportion of each digit and the fraction of ambiguous images.
"""
import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def w1_exact(a, b):
    """W1 between two uniform empirical measures with the same number of points."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    cost = cdist(a, b)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def mode_statistics(x, mixture, n_sigma=3.0):
    """Modes covered, fraction of high-quality samples and total variation of the mode weights.

    A sample is high quality if it lies within n_sigma standard deviations of its nearest mean.
    A mode is covered if at least one high-quality sample falls in it.
    """
    x = np.asarray(x, np.float64)
    d = cdist(x, mixture.means)
    nearest = d.argmin(1)
    hq = d[np.arange(len(x)), nearest] < n_sigma * mixture.std
    counts = np.bincount(nearest[hq], minlength=mixture.n_components)
    covered = int((counts > 0).sum())
    if hq.sum() > 0:
        weights = counts / counts.sum()
        tv = 0.5 * float(np.abs(weights - mixture.weights).sum())
    else:
        tv = 1.0
    return {"modes": covered, "hq_fraction": float(hq.mean()), "weight_tv": tv}


def neighbour_segments(mixture, tol=1.01):
    """Segments between neighbouring modes: pairs of means at the minimum distance (up to tol)."""
    d = cdist(mixture.means, mixture.means)
    np.fill_diagonal(d, np.inf)
    i, j = np.nonzero(np.triu(d <= tol * d.min()))
    return mixture.means[i], mixture.means[j]


def distance_to_segments(x, a, b):
    """Distance from each point of x to the nearest of the segments [a_m, b_m]."""
    x = np.asarray(x, np.float64)
    u = b - a
    lam = np.clip(((x[:, None, :] - a[None]) * u[None]).sum(-1) / (u * u).sum(-1)[None], 0.0, 1.0)
    proj = a[None] + lam[:, :, None] * u[None]
    return np.linalg.norm(x[:, None, :] - proj, axis=-1).min(1)


def placement_statistics(x, mixture, n_sigma=3.0):
    """Fractions of samples on a mode, on a segment between neighbouring modes, and elsewhere.

    A sample is on a mode if it lies within n_sigma standard deviations of a mean (the
    high-quality criterion), and on a segment if it is not on a mode but lies within the same
    distance of a segment between two neighbouring modes.
    """
    x = np.asarray(x, np.float64)
    band = n_sigma * mixture.std
    on_mode = cdist(x, mixture.means).min(1) < band
    a, b = neighbour_segments(mixture)
    on_seg = (~on_mode) & (distance_to_segments(x, a, b) < band)
    return {"on_mode": float(on_mode.mean()), "on_segment": float(on_seg.mean()),
            "elsewhere": float(1.0 - on_mode.mean() - on_seg.mean())}


def frechet_distance(feat_a, feat_b):
    """Fréchet distance between Gaussians fitted to two sets of features."""
    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    cov_a = np.cov(feat_a, rowvar=False)
    cov_b = np.cov(feat_b, rowvar=False)
    covmean = np.real(linalg.sqrtm(cov_a @ cov_b))
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))


def digit_statistics(probs, digits=(3, 4), reference_share=None, threshold=0.9):
    """Share of each digit among the clear images and fraction of ambiguous images.

    probs are the softmax outputs of the ten-class classifier. An image is a clear d if the
    classifier assigns it to d with probability at least threshold, and ambiguous otherwise
    (including images assigned to a digit outside the chosen ones).
    """
    probs = np.asarray(probs)
    pred, conf = probs.argmax(1), probs.max(1)
    clear = {d: (pred == d) & (conf >= threshold) for d in digits}
    n_clear = sum(int(c.sum()) for c in clear.values())
    first = float(clear[digits[0]].sum() / n_clear) if n_clear else float("nan")
    out = {"ambiguous": float(1.0 - n_clear / len(probs)), f"share_{digits[0]}": first}
    if reference_share is not None:
        out["share_error"] = abs(first - reference_share) if n_clear else 1.0
    return out
