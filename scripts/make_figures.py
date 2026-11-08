"""Every figure of the report, generated from the result files.

Colours: the five methods keep the same colour in every figure, and FM-OT, the method under
study, is drawn with a thicker line. The placement of the 2D samples (on a mode, on a segment
between neighbouring modes, elsewhere) also keeps the same three colours everywhere.

Usage: python scripts/make_figures.py [--results results] [--figures figures]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmstudy.data import get_mixture  # noqa: E402
from fmstudy.metrics import distance_to_segments, neighbour_segments  # noqa: E402
from fmstudy.utils import FIGURES, RESULTS, load_config  # noqa: E402

INK, MUTED, GRID, REAL = "#0b0b0b", "#52514e", "#e4e3df", "#d6d5cf"
# (method, sampler) -> label, colour, marker, line style. Validated categorical order.
SERIES = {("fmot", "euler"): ("FM-OT", "#2a78d6", "o", "-"),
          ("fm", "euler"): ("FM", "#eb6834", "s", "-"),
          ("rf2", "euler"): ("RF-2", "#1baf7a", "^", "-"),
          ("ddpm", "ddim"): (r"DDPM, $\eta=0$", "#eda100", "D", "--"),
          ("ddpm", "ddpm"): (r"DDPM, $\eta=1$", "#e87ba4", "v", ":")}
PLACEMENT = {"on mode": "#2a78d6", "on segment": "#eb6834", "elsewhere": "#52514e"}
SIGMA_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # sigma0 = 0.05, 0.3, 1, 3
DATASETS = {"8gaussians": "8 Gaussians", "25gaussians": "25 Gaussians", "3unequal": "3 unequal weights"}
LIMITS = {"8gaussians": 1.75, "25gaussians": 1.75, "3unequal": 1.95}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def save(fig, fig_dir, name):
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def grid_lines(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def placement_colours(x, mix):
    """Colour of each sample by its placement, with the definitions of metrics.placement_statistics."""
    band = 3 * mix.std
    on_mode = cdist(x, mix.means).min(1) < band
    a, b = neighbour_segments(mix)
    on_seg = ~on_mode & (distance_to_segments(x, a, b) < band)
    c = np.full(len(x), PLACEMENT["elsewhere"], dtype=object)
    c[on_seg] = PLACEMENT["on segment"]
    c[on_mode] = PLACEMENT["on mode"]
    return c


def scatter_panel(ax, test, x, mix, lim):
    a, b = neighbour_segments(mix)
    for p, q in zip(a, b):
        ax.plot([p[0], q[0]], [p[1], q[1]], c=GRID, lw=0.6, zorder=0)
    ax.scatter(test[:, 0], test[:, 1], s=0.6, c=REAL, lw=0, rasterized=True, zorder=1)
    ax.scatter(x[:, 0], x[:, 1], s=0.8, c=placement_colours(x, mix), lw=0, rasterized=True, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def placement_legend(fig, y=1.0):
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4, mfc=c, mec="none", label=k)
               for k, c in PLACEMENT.items()]
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=4, mfc=REAL, mec="none", label="test data"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=4, frameon=False,
               handletextpad=0.2, columnspacing=1.0)


def exact_samples(res, fig_dir, cfg, steps=(1, 2, 3, 5, 10, 1000), dataset="8gaussians"):
    """Euler on the exact field of FM for each width of the noise and number of steps."""
    mix, sigmas = get_mixture(dataset), cfg["toy"]["sigma0_sweep"]
    fig, axes = plt.subplots(len(sigmas), len(steps), figsize=(5.5, 3.75))
    for i, s0 in enumerate(sigmas):
        p = res / "2d_samples" / f"exact_{dataset}_s{s0}_b0.npz"
        if not p.exists():
            continue
        d = np.load(p)
        for j, k in enumerate(steps):
            ax = axes[i, j]
            scatter_panel(ax, d["test"], d[f"samples_exact_euler_{k}"], mix, LIMITS[dataset])
            if i == 0:
                ax.set_title(f"$K={k}$")
            if j == 0:
                ax.text(-0.08, 0.5, rf"$\sigma_0={s0:g}$", transform=ax.transAxes, rotation=90,
                        ha="right", va="center")
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    placement_legend(fig, 0.94)
    save(fig, fig_dir, f"exact_samples_{dataset}")


def main_samples(res, fig_dir, dataset="8gaussians", steps=(1, 2, 3, 5, 10, 1000), methods=None, suffix=""):
    """Samples of FM-OT and of the three standards with the same noise (sigma0 = 1, B = 256).

    `methods` keeps only those methods, as in the figure of the main text (exact field, FM-OT and FM)."""
    p = res / "2d_samples" / f"main_{dataset}_s1.0_b256.npz"
    if not p.exists():
        return
    d, mix = np.load(p), get_mixture(dataset)
    rows = [("exact", None)] if (res / "2d_samples" / f"exact_{dataset}_s1.0_b0.npz").exists() else []
    rows += [key for key in SERIES if methods is None or key[0] in methods]
    exact = np.load(res / "2d_samples" / f"exact_{dataset}_s1.0_b0.npz") if rows[0][0] == "exact" else None
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(5.5, 0.93 * len(rows) + 0.1))
    for i, (method, sampler) in enumerate(rows):
        for j, k in enumerate(steps):
            ax = axes[i, j]
            if method == "exact":
                x, label = exact[f"samples_exact_euler_{k}"], "exact field"
            else:
                x, label = d[f"samples_{method}_{sampler}_{k}"], SERIES[(method, sampler)][0]
            scatter_panel(ax, d["test"], x, mix, LIMITS[dataset])
            if i == 0:
                ax.set_title(f"$K={k}$")
            if j == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center")
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    placement_legend(fig, 0.955)
    save(fig, fig_dir, f"main_samples_{dataset}{suffix}")


def trajectories(res, fig_dir, dataset="8gaussians", n=150):
    """Euler trajectories of 100 steps from the noise to the data."""
    p = res / "2d_samples" / f"main_{dataset}_s1.0_b256.npz"
    if not p.exists():
        return
    d = np.load(p)
    exact = np.load(res / "2d_samples" / f"exact_{dataset}_s1.0_b0.npz")
    panels = [("exact field", exact["path_exact"], MUTED)]
    panels += [(SERIES[(m, "euler")][0], d[f"path_{m}"], SERIES[(m, "euler")][1]) for m in ("fmot", "fm", "rf2")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5, 1.55))
    for ax, (label, path, colour) in zip(axes, panels):
        path = path[:, :n]
        ax.plot(path[:, :, 0], path[:, :, 1], c=colour, lw=0.35, alpha=0.6, rasterized=True)
        ax.scatter(path[0, :, 0], path[0, :, 1], s=0.8, c=REAL, lw=0, zorder=3)
        ax.scatter(path[-1, :, 0], path[-1, :, 1], s=1.2, c=INK, lw=0, zorder=3)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label)
    fig.subplots_adjust(wspace=0.05)
    save(fig, fig_dir, f"trajectories_{dataset}")


def series_plot(ax, df, col, steps_col="steps", highlight=True):
    for (method, sampler), (label, colour, marker, ls) in SERIES.items():
        r = df[(df.method == method) & (df.sampler == sampler)].sort_values(steps_col)
        if not len(r):
            continue
        main = method == "fmot" and highlight
        ax.plot(r[steps_col], r[col], c=colour, ls=ls, lw=1.8 if main else 1.0, marker=marker,
                ms=3.2 if main else 2.6, label=label, zorder=4 if main else 3)


def reference_line(ax, y):
    """Dotted line of two real samples on a logarithmic axis, labelled below it with room to spare."""
    ax.axhline(y, c=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(1, y / 1.2, "real vs real", color=MUTED, fontsize=6, va="top")
    ax.set_ylim(bottom=y / 2.2)


def w1_curves(res, fig_dir):
    """W1 to the test data against the number of steps on the three mixtures, and FD on MNIST."""
    m2 = pd.read_csv(res / "2d_metrics.csv").drop_duplicates(
        ["experiment", "dataset", "sigma0", "batch_size", "method", "sampler", "steps"], keep="last")
    ref = pd.read_csv(res / "2d_reference.csv").drop_duplicates(["dataset"], keep="last").set_index("dataset")
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.75))
    for ax, (name, title) in zip(axes, DATASETS.items()):
        r = m2[(m2.experiment == "main") & (m2.dataset == name)]
        series_plot(ax, r, "w1")
        if name in ref.index:
            reference_line(ax, ref.loc[name, "w1_real_real"])
        ax.set_title(title)
    axes[0].set_ylabel("$W_1$ to the test data")
    p = res / "mnist_metrics.csv"
    if p.exists():
        mm = pd.read_csv(p).drop_duplicates(["method", "seed", "sampler", "steps"], keep="last")
        mref = pd.read_csv(res / "mnist_reference.csv").iloc[-1]
        series_plot(axes[3], mm, "fd")
        reference_line(axes[3], mref.fd_train_vs_test)
    axes[3].set_title("MNIST 3 / 4 (FD)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.set_xlabel("steps $K$")
        grid_lines(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, fig_dir, "w1_steps")


def sigma0_curves(res, fig_dir, cfg):
    """Fraction of samples off the modes against the steps, for each width of the noise."""
    m2 = pd.read_csv(res / "2d_metrics.csv").drop_duplicates(
        ["experiment", "dataset", "sigma0", "batch_size", "method", "sampler", "steps"], keep="last")
    m2 = m2[(m2.dataset == "8gaussians") & (m2.batch_size.isin([0, cfg["toy"]["batch_size"]]))]
    panels = [("exact field", ("exact",), "exact"), ("FM", ("main", "sigma0"), "fm"),
              ("FM-OT", ("main", "sigma0"), "fmot")]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6), sharey=True)
    for ax, (title, exps, method) in zip(axes, panels):
        for s0, colour in zip(cfg["toy"]["sigma0_sweep"], SIGMA_RAMP):
            r = m2[m2.experiment.isin(exps) & (m2.method == method) & np.isclose(m2.sigma0, s0)
                   & (m2.sampler == "euler")].sort_values("steps")
            if len(r):
                ax.plot(r.steps, 1 - r.on_mode, c=colour, lw=1.2, marker="o", ms=2.4, label=rf"$\sigma_0={s0:g}$")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.set_title(title)
        ax.set_xlabel("steps $K$")
        ax.set_ylim(-0.02, 1.02)
        grid_lines(ax)
    axes[0].set_ylabel("fraction off the modes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, fig_dir, "sigma0_off_mode")


def batch_curves(res, fig_dir, cfg, k_w1=2):
    """Effect of the minibatch size on FM and FM-OT: straightness, W1 with few steps and cost."""
    toy = cfg["toy"]
    geo = pd.read_csv(res / "2d_straightness.csv").drop_duplicates(
        ["experiment", "dataset", "sigma0", "batch_size", "method"], keep="last")
    m2 = pd.read_csv(res / "2d_metrics.csv").drop_duplicates(
        ["experiment", "dataset", "sigma0", "batch_size", "method", "sampler", "steps"], keep="last")
    tim = pd.read_csv(res / "2d_training.csv").drop_duplicates(
        ["experiment", "dataset", "sigma0", "batch_size", "method"], keep="last")

    def sel(df, method):
        return df[df.experiment.isin(["main", "batch"]) & (df.dataset == "8gaussians") & (df.method == method)
                  & np.isclose(df.sigma0, toy["sigma0"])].sort_values("batch_size")

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.6))
    for method in ("fmot", "fm"):
        label, colour, marker, _ = SERIES[(method, "euler")]
        lw = 1.8 if method == "fmot" else 1.0
        g = sel(geo, method)
        axes[0].plot(g.batch_size, g.straightness, c=colour, lw=lw, marker=marker, ms=3, label=label)
        w = sel(m2, method)
        w = w[(w.steps == k_w1) & (w.sampler == "euler")]
        axes[1].plot(w.batch_size, w.w1, c=colour, lw=lw, marker=marker, ms=3, label=label)
        t = sel(tim, method)
        axes[2].plot(t.batch_size, t.train_seconds / 60, c=colour, lw=lw, marker=marker, ms=3, label=label)
        if method == "fmot":
            axes[2].plot(t.batch_size, t.coupling_seconds / 60, c=colour, lw=1.0, ls="--", marker=marker,
                         ms=3, mfc="white", label="FM-OT, coupling only")
    titles = ("straightness $S$", f"$W_1$ with $K={k_w1}$", "training time (min)")
    for ax, title in zip(axes, titles):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(toy["batch_sweep"])
        ax.set_xticklabels([str(b) for b in toy["batch_sweep"]])
        ax.minorticks_off()
        ax.set_xlabel("minibatch size $B$")
        ax.set_title(title)
        grid_lines(ax)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, fig_dir, "batch_effect")


def mnist_samples(res, fig_dir, steps=(1, 2, 5, 10, 1000), n=16):
    """The first 16 generated images (same noise) of each method and number of steps."""
    rows = []
    for (method, sampler), (label, *_rest) in SERIES.items():
        p = res / "mnist_samples" / f"{method}_seed0.npz"
        if p.exists():
            rows.append((method, sampler, label, np.load(p)))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(5.5, 1.12 * len(rows)))
    axes = np.atleast_2d(axes)
    side = int(np.sqrt(n))
    for i, (method, sampler, label, d) in enumerate(rows):
        for j, k in enumerate(steps):
            ax = axes[i, j]
            imgs = d[f"{sampler}_{k}"][:n, 0]
            grid = imgs.reshape(side, side, 28, 28).transpose(0, 2, 1, 3).reshape(side * 28, side * 28)
            ax.imshow(255 - grid, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"$K={k}$")
            if j == 0:
                ax.set_ylabel(label)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, fig_dir, "mnist_samples")


def loss_curves(res, fig_dir):
    """Training and validation losses of the main 2D comparison and of MNIST (appendix)."""
    l2 = pd.read_csv(res / "2d_loss.csv")
    p = res / "mnist_loss.csv"
    lm = pd.read_csv(p) if p.exists() else pd.DataFrame()
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.6))
    for ax, name in zip(axes[:3], DATASETS):
        r = l2[(l2.experiment == "main") & (l2.dataset == name)]
        for m in ("fmot", "fm", "rf2", "ddpm"):
            label, colour, *_ = SERIES[(m, "ddim" if m == "ddpm" else "euler")]
            q = r[r.method == m].drop_duplicates("iteration", keep="last")
            ax.plot(q.iteration, q.train_loss, c=colour, lw=0.8, alpha=0.5)
            ax.plot(q.iteration, q.val_loss, c=colour, lw=1.2, label=label.split(",")[0])
        ax.set_title(DATASETS[name])
    if len(lm):
        for m in ("fmot", "fm", "rf2", "ddpm"):
            label, colour, *_ = SERIES[(m, "ddim" if m == "ddpm" else "euler")]
            q = lm[lm.method == m].drop_duplicates("iteration", keep="last")
            axes[3].plot(q.iteration, q.train_loss, c=colour, lw=0.8, alpha=0.5)
            axes[3].plot(q.iteration, q.val_loss_ema, c=colour, lw=1.2, label=label.split(",")[0])
        axes[3].set_title("MNIST 3 / 4")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.tick_params(axis="x", labelrotation=30)
        grid_lines(ax)
    axes[0].set_ylabel("loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, fig_dir, "loss_curves")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--figures", default=str(FIGURES))
    args = parser.parse_args()
    res, fig_dir = Path(args.results), Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    jobs = [lambda: exact_samples(res, fig_dir, cfg),
            lambda: main_samples(res, fig_dir),
            lambda: main_samples(res, fig_dir, methods=("fmot", "fm"), suffix="_core"),
            lambda: trajectories(res, fig_dir),
            lambda: w1_curves(res, fig_dir),
            lambda: sigma0_curves(res, fig_dir, cfg),
            lambda: batch_curves(res, fig_dir, cfg),
            lambda: mnist_samples(res, fig_dir),
            lambda: loss_curves(res, fig_dir)]
    for job in jobs:
        try:
            job()
        except FileNotFoundError as err:  # results of a later stage not yet available
            print("skipped:", err, flush=True)
    print("figures written to", fig_dir, flush=True)


if __name__ == "__main__":
    main()
