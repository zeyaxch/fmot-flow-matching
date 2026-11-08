"""Aggregate the raw results, test the hypotheses H1-H6 and write the tables and numbers of the report.

Every number quoted in the report comes from this script or from make_figures.py. The numbers go
to tables/numbers.tex as keyed values, read in the report with \\val{key}. With one seed per
configuration, two values differ relevantly only if the larger exceeds the smaller by more than
the fraction fixed in configs/experiments.json ("relevance"), twice the spread between three seeds
in preliminary runs of the study.

Usage: python scripts/analyze.py [--results results] [--tables ../informe/tables]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmstudy import metrics  # noqa: E402
from fmstudy.data import get_mixture  # noqa: E402
from fmstudy.methods import MixtureVelocity  # noqa: E402
from fmstudy.models import MLP2D  # noqa: E402
from fmstudy.utils import RESULTS, ROOT, RUNS, load_config, read_mnist_sweep  # noqa: E402

DATASETS = {"8gaussians": ("8g", "8 Gaussians"), "25gaussians": ("25g", "25 Gaussians"),
            "3unequal": ("3u", "3 unequal")}
# (method, sampler) -> (key, label). The key of the DDPM rows is the sampler.
ROWS = {("fmot", "euler"): ("fmot", "FM-OT"), ("fm", "euler"): ("fm", "FM"),
        ("rf2", "euler"): ("rf2", "RF-2"), ("ddpm", "ddim"): ("ddim", r"DDPM, $\eta=0$"),
        ("ddpm", "ddpm"): ("ddpm", r"DDPM, $\eta=1$")}
FLOWS = ("fmot", "fm", "rf2")
FEW = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------- formatting

def is_missing(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def sig(x, digits=3):
    """Significant digits, keeping trailing zeros and without scientific notation."""
    if is_missing(x):
        return "--"
    x = float(x)
    if x != 0 and abs(x) < 1e-4:
        mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
        return rf"${mantissa}\cdot10^{{{int(exponent)}}}$"
    if abs(x) >= 10 ** digits:
        return f"{x:.0f}"
    return f"{x:#.{digits}g}".rstrip(".")


def skey(x):
    """Value of a swept hyperparameter inside a key: 0.05, 0.3, 1, 3."""
    return f"{float(x):g}"


def dist(x):
    """W1 and FD: three significant digits, at most one decimal above 100."""
    if is_missing(x):
        return "--"
    return f"{x:.0f}" if abs(x) >= 100 else sig(x)


def pct(x):
    if is_missing(x):
        return "--"
    if 0 < x < 0.005:
        return r"$<$1\%"
    return f"{100 * x:.0f}\\%"


def seconds(x):
    if is_missing(x):
        return "--"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def tex_escape(text):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}")):
        text = text.replace(a, b)
    return text


def ratio_relevant(a, b, tol):
    """True if the larger of two positive values exceeds the smaller by more than tol."""
    lo, hi = sorted((float(a), float(b)))
    return hi > (1 + tol) * max(lo, 1e-12)


# ---------------------------------------------------------------- loading

def read(results, name):
    path = results / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def last(df):
    """Keep the last row of each configuration, in case a run was repeated."""
    keys = [c for c in ("experiment", "dataset", "sigma0", "batch_size", "seed", "method", "sampler",
                        "steps") if c in df.columns]
    return df.drop_duplicates(keys, keep="last") if len(df) else df


class Lookup:
    """Access to one metric of the 2D results by experiment, dataset, sigma0, batch and method."""

    def __init__(self, df):
        self.df = df

    def get(self, col, experiment, dataset, method, sampler, steps, sigma0=1.0, batch=None):
        d = self.df
        if not len(d):
            return float("nan")
        mask = ((d.experiment == experiment) & (d.dataset == dataset) & (d.method == method)
                & (d.sampler == sampler) & (d.steps == steps) & np.isclose(d.sigma0, sigma0))
        if batch is not None:
            mask &= d.batch_size == batch
        rows = d[mask]
        return float(rows[col].iloc[0]) if len(rows) else float("nan")


# ---------------------------------------------------------------- numbers

class Numbers:
    def __init__(self):
        self.values = {}

    def put(self, key, text):
        self.values[key] = text

    def write(self, path):
        """Write the values that the report reads with \\val, or all of them if there is no report."""
        report = path.parent.parent
        used = set()
        for tex in [report / "main.tex", *sorted((report / "sections").glob("*.tex"))]:
            if tex.exists():
                used |= set(re.findall(r"\\val\{([^}]*)\}", tex.read_text()))
        missing = sorted(used - set(self.values))
        if missing:
            print("values read by the report but not computed:", ", ".join(missing))
        keys = sorted(used & set(self.values)) if used else sorted(self.values)
        lines = ["% Generated by scripts/analyze.py. Do not edit by hand.",
                 r"\makeatletter",
                 r"\newcommand{\val}[1]{\ifcsname val@#1\endcsname\csname val@#1\endcsname"
                 r"\else\textbf{??}\PackageWarning{numbers}{Undefined value #1}\fi}",
                 r"\makeatother"]
        for k in keys:
            lines.append(rf"\expandafter\def\csname val@{k}\endcsname{{{self.values[k]}}}")
        path.write_text("\n".join(lines) + "\n")
        return len(keys)


def boundary_field(toy, times=(0.9, 0.99), lam=0.35):
    """Velocity towards the nearer mode on the segments between neighbouring modes.

    At x_t = t ((1 - lam) mu_a + lam mu_b), a point of the segment closer to mode a, we project the
    exact field of FM and the field learned by each FM network on the direction from b to a. A
    network that reproduces the separation between the modes should match the exact value.
    """
    mix = get_mixture("8gaussians")
    exact = MixtureVelocity(mix, toy["sigma0"])
    a, b = metrics.neighbour_segments(mix)
    d = torch.tensor((a - b) / np.linalg.norm(a - b, axis=1, keepdims=True), dtype=torch.float32)
    models = {"exact": exact}
    for batch in toy["batch_sweep"]:
        exp = "main" if batch == toy["batch_size"] else "batch"
        path = RUNS / "2d" / exp / f"8gaussians_s{toy['sigma0']}_b{batch}_seed0" / "fm.pt"
        if path.exists():
            model = MLP2D(toy["mlp_hidden"], toy["mlp_layers"], toy["time_dim"])
            model.load_state_dict(torch.load(path))
            models[f"fm-b{batch}"] = model.eval()
    out = {}
    for name, model in models.items():
        for t in times:
            x = torch.tensor(t * ((1 - lam) * a + lam * b), dtype=torch.float64 if name == "exact" else torch.float32)
            with torch.no_grad():
                v = model(x, torch.full((len(x),), t, dtype=x.dtype)).float()
            out[(name, t)] = float((v * d).sum(1).mean())
    return out


# ---------------------------------------------------------------- tables

def tabular(spec, header, body):
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    out += body
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def w1_main_table(look, ref, grid):
    header = "Data & Method & " + " & ".join(f"$K={k}$" for k in grid)
    body = []
    for i, (name, (_, label)) in enumerate(DATASETS.items()):
        if i:
            body.append(r"\midrule")
        real = ref.get(name, float("nan"))
        rows = list(ROWS.items())
        for j, ((method, sampler), (_, mlabel)) in enumerate(rows):
            first = rf"\multirow{{{len(rows)}}}{{*}}{{\shortstack[l]{{{label}\\(real {dist(real)})}}}}" if j == 0 else ""
            cells = [dist(look.get("w1", "main", name, method, sampler, k)) for k in grid]
            body.append(f"{first} & {mlabel} & " + " & ".join(cells) + r" \\")
    return tabular("@{}ll" + "r" * len(grid) + "@{}", header, body)


def placement_table(look, grid, dataset="8gaussians"):
    """Fraction of samples on a mode and on a segment between neighbouring modes."""
    header = "Method & " + " & ".join(f"$K={k}$" for k in grid)
    body = []
    for (method, sampler), (_, mlabel) in ROWS.items():
        cells = []
        for k in grid:
            on = look.get("on_mode", "main", dataset, method, sampler, k)
            seg = look.get("on_segment", "main", dataset, method, sampler, k)
            cells.append(f"{pct(on)} / {pct(seg)}")
        body.append(f"{mlabel} & " + " & ".join(cells) + r" \\")
    for s0 in (1.0,):
        cells = []
        for k in grid:
            on = look.get("on_mode", "exact", dataset, "exact", "euler", k, s0)
            seg = look.get("on_segment", "exact", dataset, "exact", "euler", k, s0)
            cells.append(f"{pct(on)} / {pct(seg)}")
        body.insert(0, r"Exact field of FM & " + " & ".join(cells) + r" \\")
    return tabular("@{}l" + "c" * len(grid) + "@{}", header, body)


def exact_table(look, sigmas, grid, dataset):
    """Exact field: fraction off the modes and, among those, fraction on a segment."""
    header = r"$\sigma_0$ & " + " & ".join(f"$K={k}$" for k in grid)
    body = []
    for s0 in sigmas:
        cells = []
        for k in grid:
            on = look.get("on_mode", "exact", dataset, "exact", "euler", k, s0)
            seg = look.get("on_segment", "exact", dataset, "exact", "euler", k, s0)
            off = 1 - on
            share = seg / off if off > 1e-9 else float("nan")
            cells.append(f"{pct(off)} ({pct(share)})" if off > 1e-9 else pct(off))
        body.append(f"{skey(s0)} & " + " & ".join(cells) + r" \\")
    return tabular("@{}l" + "c" * len(grid) + "@{}", header, body)


def sweep_table(m2, geo, tim, column, values, label, show_cost):
    """FM and FM-OT on the eight Gaussians for each value of sigma0 or of the minibatch size."""
    # S grows with the scale of the noise, so the sweep of sigma0 reports the scale-free length ratio.
    geo_col, geo_label = ("length_ratio", "length / chord") if column == "sigma0" else ("straightness", "$S$")
    cols = [geo_label, "$W_1$, $K=1$", "$K=2$", "$K=5$", "$K=1000$", "on mode, $K=5$", "on segment, $K=1000$"]
    if show_cost:
        cols += ["training (s)", "coupling (s)"]
    header = f"{label} & Method & " + " & ".join(cols)
    body = []
    for i, v in enumerate(values):
        if i:
            body.append(r"\midrule")
        for method in ("fmot", "fm"):
            def pick(df):
                if not len(df):
                    return df
                exp = df.experiment.isin(["main", "sigma0" if column == "sigma0" else "batch"])
                return df[(df.dataset == "8gaussians") & (df.method == method) & exp
                          & np.isclose(df[column], v)]
            mm = pick(m2)
            mm = mm[mm.sampler == "euler"] if len(mm) else mm

            def w(k, col="w1"):
                r = mm[mm.steps == k] if len(mm) else mm
                return float(r[col].iloc[0]) if len(r) else float("nan")

            g = pick(geo)
            s_value = float(g[geo_col].iloc[0]) if len(g) else float("nan")
            cells = [sig(s_value), dist(w(1)), dist(w(2)), dist(w(5)), dist(w(1000)), pct(w(5, "on_mode")),
                     pct(w(1000, "on_segment"))]
            if show_cost:
                t = pick(tim)
                cells += [seconds(float(t.train_seconds.iloc[0])) if len(t) else "--",
                          seconds(float(t.coupling_seconds.iloc[0])) if len(t) and method == "fmot" else "--"]
            first = f"{skey(v)}" if method == "fmot" else ""
            name = "FM-OT" if method == "fmot" else "FM"
            body.append(f"{first} & {name} & " + " & ".join(cells) + r" \\")
    n = 2 + len(cols)
    return tabular("@{}ll" + "r" * (n - 2) + "@{}", header, body)


def straightness_table(geo, mgeo):
    header = "Method & " + " & ".join(label for _, label in DATASETS.values()) + " & MNIST 3 / 4"
    body = []
    for method in FLOWS:
        cells = []
        for name in DATASETS:
            g = geo[(geo.experiment == "main") & (geo.dataset == name) & (geo.method == method)] if len(geo) else geo
            cells.append(f"{sig(g.straightness.iloc[0])} ({sig(g.length_ratio.iloc[0])})" if len(g) else "--")
        g = mgeo[mgeo.method == method] if len(mgeo) else mgeo
        cells.append(f"{sig(g.straightness.iloc[0])} ({sig(g.length_ratio.iloc[0])})" if len(g) else "--")
        body.append(f"{ROWS[(method, 'euler')][1]} & " + " & ".join(cells) + r" \\")
    g = geo[(geo.experiment == "exact") & np.isclose(geo.sigma0, 1.0)] if len(geo) else geo
    cells = []
    for name in DATASETS:
        r = g[g.dataset == name] if len(g) else g
        cells.append(f"{sig(r.straightness.iloc[0])} ({sig(r.length_ratio.iloc[0])})" if len(r) else "--")
    body.insert(0, "Exact field of FM & " + " & ".join(cells) + r" & -- \\")
    return tabular("@{}lcccc@{}", header, body)


def mnist_table(mm, grid, col, fmt):
    header = "Method & " + " & ".join(f"$K={k}$" for k in grid)
    body = []
    for (method, sampler), (_, mlabel) in ROWS.items():
        cells = []
        for k in grid:
            r = mm[(mm.method == method) & (mm.sampler == sampler) & (mm.steps == k)] if len(mm) else mm
            cells.append(fmt(float(r[col].iloc[0])) if len(r) else "--")
        body.append(f"{mlabel} & " + " & ".join(cells) + r" \\")
    return tabular("@{}l" + "r" * len(grid) + "@{}", header, body)


def mnist_batch_table(mbm, mbg, mbt, batches, ks=(1, 2, 5, 10, 1000)):
    """FM and FM-OT on MNIST 3 / 4 for each minibatch size: straightness, FD and training time."""
    header = ("$B$ & Method & $S$ & " + " & ".join(f"FD, $K={k}$" if i == 0 else f"$K={k}$" for i, k in enumerate(ks))
              + " & training (min)")
    body = []
    for i, b in enumerate(batches):
        if i:
            body.append(r"\midrule")
        for method in ("fmot", "fm"):
            def pick(df):
                return df[(df.batch_size == b) & (df.method == method)] if len(df) else df
            g, m, t = pick(mbg), pick(mbm), pick(mbt)
            cells = [sig(float(g.straightness.iloc[0])) if len(g) else "--"]
            for k in ks:
                r = m[m.steps == k] if len(m) else m
                cells.append(dist(float(r.fd.iloc[0])) if len(r) else "--")
            cells.append(f"{float(t.train_seconds.iloc[0]) / 60:.0f}" if len(t) else "--")
            first = str(b) if method == "fmot" else ""
            name = "FM-OT" if method == "fmot" else "FM"
            body.append(f"{first} & {name} & " + " & ".join(cells) + r" \\")
    return tabular("@{}ll" + "r" * (len(ks) + 2) + "@{}", header, body)


def compute_table(tim, m2, mtim, mm, mbt, mbm, default_batch):
    """Training, coupling, reflow pairs and sampling time of each run."""
    header = "Experiment & Method & Runs & Training & Coupling & Pairs & Sampling (min)"
    names = {"main": "2D main", "sigma0": r"2D $\sigma_0$", "batch": "2D $B$"}
    body = []

    def minutes(seconds, used):
        return f"{seconds / 60:.1f}" if used else "--"

    def row(name, method, runs, train, coupling, pairs, gen):
        label = ROWS[(method, "ddim" if method == "ddpm" else "euler")][1].split(",")[0]
        return (f"{name} & {label} & {runs} & {train / 60:.1f} & {coupling} & "
                f"{minutes(pairs, method == 'rf2')} & {gen / 60:.1f}" + r" \\")

    for exp in ("main", "sigma0", "batch"):
        t = tim[tim.experiment == exp] if len(tim) else tim
        g = m2[m2.experiment == exp] if len(m2) else m2
        for method in ("fmot", "fm", "rf2", "ddpm"):
            r = t[t.method == method] if len(t) else t
            if not len(r):
                continue
            gen = g[g.method == method].gen_seconds.sum() if len(g) else float("nan")
            body.append(row(names[exp], method, len(r), r.train_seconds.sum(),
                            minutes(r.coupling_seconds.sum(), method == "fmot"), r.pair_seconds.sum(), gen))
    if len(mtim):
        body.append(r"\midrule")
        for method in ("fmot", "fm", "rf2", "ddpm"):
            r = mtim[mtim.method == method]
            if not len(r):
                continue
            gen = mm[mm.method == method].gen_seconds.sum() if len(mm) else float("nan")
            body.append(row("MNIST 3 / 4", method, len(r), r.train_seconds.sum(), "--", r.pair_seconds.sum(), gen))
    extra = mbt[mbt.batch_size != default_batch] if len(mbt) else mbt
    for method in ("fmot", "fm"):
        r = extra[extra.method == method] if len(extra) else extra
        if not len(r):
            continue
        g = mbm[(mbm.method == method) & (mbm.batch_size != default_batch)]
        body.append(row("MNIST $B$", method, len(r), r.train_seconds.sum(), "--", r.pair_seconds.sum(),
                        g.gen_seconds.sum()))
    return tabular("@{}llrrrrr@{}", header, body)


def hyper_table(raw):
    """Every hyperparameter of configs/experiments.json with its value and its source (a longtable)."""
    sections = {"": "General", "ddpm": "DDPM", "toy": "2D", "mnist": "MNIST", "relevance": "Relevance"}
    body = []

    def walk(node, prefix):
        for k, v in node.items():
            if k == "_comment":
                continue
            if isinstance(v, dict) and "value" in v:
                val = v["value"]
                if isinstance(val, list):
                    val = ", ".join(str(x) for x in val)
                body.append(f"{sections[prefix]} & \\texttt{{{tex_escape(k)}}} & {tex_escape(str(val))} & "
                            f"{tex_escape(v['source'])}" + r" \\")
            elif isinstance(v, dict):
                walk(v, k)

    walk(raw, "")
    head = [r"\toprule", r"Part & Name & Value & Source \\", r"\midrule"]
    # A longtable, whose header is repeated on every page.
    first = [r"\begin{longtable}{@{}llp{2.6cm}p{7.2cm}@{}}",
             r"\caption{Every hyperparameter of the study, generated from \texttt{configs/experiments.json}.}",
             r"\label{tab:provenance}\\"]
    rows = first + head + [r"\endfirsthead"] + head + [r"\endhead"] + body + [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------- hypotheses

def verdict(ok, partly=False):
    return "supported" if ok else ("partly supported" if partly else "not supported")


def readme_summary(nums, hyp):
    """Markdown summary of the main results for the README, written from the same values as the report."""
    v = nums.values

    def g(key):
        return v.get(key, "--").replace(r"\%", "%").replace("$<$", "<")

    labels = [(key, label) for (_, _), (key, label) in ROWS.items()]
    names = {"ddim": "DDPM, eta = 0 (DDIM)", "ddpm": "DDPM, eta = 1"}
    steps = (1, 2, 3, 5, 10, 1000)
    out = ["W1 to the 2000 test points of the eight Gaussians (lower is better, one seed). "
           f"Two real samples of the same size reach {g('w1-real-8g')}.", "",
           "| Method | " + " | ".join(f"K = {k}" for k in steps) + " |",
           "|---|" + "---:|" * len(steps)]
    for key, label in labels:
        out.append(f"| {names.get(key, label)} | " + " | ".join(g(f"w1-8g-{key}-{k}") for k in steps) + " |")
    if any(k.startswith("fd-mnist-") for k in v):
        out += ["", "Fréchet distance on MNIST 3 / 4 in the features of our classifier "
                f"(2000 training images reach {g('fd-ref')}).", "",
                "| Method | " + " | ".join(f"K = {k}" for k in steps) + " |",
                "|---|" + "---:|" * len(steps)]
        for key, label in labels:
            out.append(f"| {names.get(key, label)} | " + " | ".join(g(f"fd-mnist-{key}-{k}") for k in steps) + " |")
    out += ["",
            f"* Few Euler steps leave the samples on the posterior mean of the data, between neighbouring modes, "
            f"also with the exact field and no network. With two steps FM-OT places {g('onmode-8g-fmot-2')} of its "
            f"samples on a mode of the eight Gaussians, against {g('onmode-8g-fm-2')} for FM.",
            f"* FM-OT lowers the straightness from {g('S-8g-fm')} to {g('S-8g-fmot')} on the eight Gaussians "
            f"(from {g('S-mnist-fm')} to {g('S-mnist-fmot')} on MNIST), and from two steps on it has the lowest W1 "
            f"in {g('fmot-lowest')} of the {g('fmot-cells')} combinations of mixture and steps.",
            f"* Its straightness falls from {g('S-8g-fmot-b16')} with B = 16 to {g('S-8g-fmot-b1024')} with "
            f"B = 1024, while one assignment costs {g('bench-8gaussians-b64')} ms with B = 64 and "
            f"{g('bench-8gaussians-b1024')} ms with B = 1024 (one thread).",
            "* Hypotheses: " + ", ".join(f"{k} {val}" for k, val in sorted(hyp.items())) + "."]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--tables", default=str(ROOT.parent / "informe" / "tables"))
    args = parser.parse_args()
    results, tables = Path(args.results), Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    raw = json.loads((ROOT / "configs" / "experiments.json").read_text())
    toy, mn, tol = cfg["toy"], cfg["mnist"], cfg["relevance"]
    grid = cfg["steps_grid"]
    nums = Numbers()

    m2 = last(read(results, "2d_metrics.csv"))
    geo = last(read(results, "2d_straightness.csv"))
    tim = last(read(results, "2d_training.csv"))
    ref2 = last(read(results, "2d_reference.csv"))
    mm = last(read(results, "mnist_metrics.csv"))
    mgeo = last(read(results, "mnist_straightness.csv"))
    mtim = last(read(results, "mnist_timing.csv"))
    mref = read(results, "mnist_reference.csv")
    # Minibatch sweep of FM and FM-OT on MNIST; the minibatch of the configuration is the main run.
    mb_sweep = mn["batch_sweep"]
    mbm, mbg, mbt = (last(read_mnist_sweep(results, name, mb_sweep, mn["batch_size"]))
                     for name in ("mnist_metrics", "mnist_straightness", "mnist_timing"))
    mbm, mbg, mbt = (df[df.method.isin(["fm", "fmot"])] if len(df) else df for df in (mbm, mbg, mbt))
    look = Lookup(m2)

    # Relevance thresholds.
    for k, v in tol.items():
        nums.put(f"tol-{k}", f"{100 * v:.0f}\\%")

    # 2D: every metric of every configuration as a keyed value.
    ref = {}
    for name, (key, _) in DATASETS.items():
        r = ref2[ref2.dataset == name] if len(ref2) else ref2
        ref[name] = float(r.w1_real_real.iloc[-1]) if len(r) else float("nan")
        nums.put(f"w1-real-{key}", dist(ref[name]))  # README
    if len(m2):
        for _, r in m2.iterrows():
            key = DATASETS[r.dataset][0]
            mkey = "exact" if r.method == "exact" else ROWS[(r.method, r.sampler)][0]
            if r.experiment == "main":
                base = f"{key}-{mkey}-{r.steps}"
            elif r.experiment == "exact":
                base = f"{key}-exact-s{skey(r.sigma0)}-{r.steps}"
            elif r.experiment == "sigma0":
                base = f"{key}-{mkey}-s{skey(r.sigma0)}-{r.steps}"
            else:
                base = f"{key}-{mkey}-b{int(r.batch_size)}-{r.steps}"
            nums.put(f"w1-{base}", dist(r.w1))
            nums.put(f"onmode-{base}", pct(r.on_mode))
            nums.put(f"onseg-{base}", pct(r.on_segment))
            nums.put(f"off-{base}", pct(1 - r.on_mode))
            off = 1 - r.on_mode
            nums.put(f"segshare-{base}", pct(r.on_segment / off) if off > 1e-9 else "--")
            nums.put(f"tv-{base}", sig(r.weight_tv, 2))
    if len(geo):
        for _, r in geo.iterrows():
            key = DATASETS[r.dataset][0]
            if r.experiment == "main":
                base = f"{key}-{r.method}"
            elif r.experiment == "exact":
                base = f"{key}-exact-s{skey(r.sigma0)}"
            elif r.experiment == "sigma0":
                base = f"{key}-{r.method}-s{skey(r.sigma0)}"
            else:
                base = f"{key}-{r.method}-b{int(r.batch_size)}"
            nums.put(f"S-{base}", sig(r.straightness))
            nums.put(f"ratio-{base}", sig(r.length_ratio))
    if len(tim):
        for _, r in tim.iterrows():
            key = DATASETS[r.dataset][0]
            base = (f"{key}-{r.method}" if r.experiment == "main" else
                    f"{key}-{r.method}-s{skey(r.sigma0)}" if r.experiment == "sigma0" else
                    f"{key}-{r.method}-b{int(r.batch_size)}")
            if r.method == "fmot" and r.train_seconds > 0:
                nums.put(f"couplingshare-{base}", pct(r.coupling_seconds / r.train_seconds))
        hours_2d = (tim.train_seconds.sum() + tim.pair_seconds.sum()
                    + (m2.gen_seconds.sum() if len(m2) else 0)) / 3600
        nums.put("hours-2d", f"{hours_2d:.1f}")

    # Amplification of the noise error by one DDIM step from the last training step.
    beta = np.linspace(cfg["ddpm"]["beta_start"], cfg["ddpm"]["beta_end"], cfg["ddpm"]["T"])
    ab = np.cumprod(1 - beta)[-1]
    nums.put("ddim-sqrt-ab", sig(np.sqrt(ab), 2))
    nums.put("ddim-amplification", f"{np.sqrt((1 - ab) / ab):.0f}")

    # Clean benchmark of the assignment (scripts/benchmark_coupling.py).
    bench = read(results, "coupling_benchmark.csv")
    if len(bench):
        bench = bench.drop_duplicates(["data", "batch_size"], keep="last")
        for _, r in bench.iterrows():
            nums.put(f"bench-{r.data}-b{int(r.batch_size)}", sig(r.median_ms, 2))
        for data, g in bench.groupby("data"):
            g = g[g.batch_size >= 64]
            slope = np.polyfit(np.log(g.batch_size), np.log(g.median_ms), 1)[0]
            nums.put(f"bench-slope-{data}", f"{slope:.1f}")

    for (name, t), v in boundary_field(toy).items():
        nums.put(f"bfield-{name}-{t:g}", sig(v, 2))
    for name, (key, _) in DATASETS.items():
        r = ref2[ref2.dataset == name] if len(ref2) else ref2
        if len(r):
            nums.put(f"onmode-real-{key}", pct(r.real_on_mode.iloc[-1]))
            nums.put(f"tv-real-{key}", sig(r.real_weight_tv.iloc[-1], 2))
        w2, w1000 = (look.get("w1", "main", name, "fmot", "euler", k) for k in (2, 1000))
        nums.put(f"fmot-gap-2-{key}", pct(w2 / w1000 - 1))

    # FM-OT against every other method and sampler from two steps on.
    if len(m2):
        cells = lowest = beaten = 0
        for name in DATASETS:
            for k in [k for k in grid if k >= 2]:
                a = look.get("w1", "main", name, "fmot", "euler", k)
                others = [look.get("w1", "main", name, m, sm, k) for (m, sm) in ROWS if m != "fmot"]
                if np.isnan(a) or any(np.isnan(o) for o in others):
                    continue
                cells += 1
                lowest += int(a <= min(others))
                beaten += sum(int(o < a and ratio_relevant(a, o, tol["w1"])) for o in others)
        nums.put("fmot-cells", str(cells))
        nums.put("fmot-lowest", str(lowest))
        nums.put("fmot-beaten", str(beaten))

    # MNIST.
    if len(mref):
        r = mref.iloc[-1]
        nums.put("clf-acc-all", f"{100 * r.classifier_accuracy_all:.1f}\\%")
        nums.put("clf-acc-digits", f"{100 * r.classifier_accuracy_digits:.1f}\\%")
        nums.put("fd-ref", dist(r.fd_train_vs_test))
        nums.put("real-ambiguous", pct(r.real_ambiguous))
        nums.put("test-share", pct(r.test_share))
    if len(mm):
        for _, r in mm.iterrows():
            base = f"mnist-{ROWS[(r.method, r.sampler)][0]}-{r.steps}"
            nums.put(f"fd-{base}", dist(r.fd))
            nums.put(f"amb-{base}", pct(r.ambiguous))
            nums.put(f"share-{base}", pct(r.share_3))
    if len(mgeo):
        for _, r in mgeo.iterrows():
            nums.put(f"S-mnist-{r.method}", sig(r.straightness))
    if len(mtim):
        gen = mm.gen_seconds.sum() if len(mm) else 0.0
        nums.put("hours-mnist", f"{(mtim.train_seconds.sum() + mtim.pair_seconds.sum() + gen) / 3600:.1f}")
        nums.put("hours-mnist-eval", f"{gen / 3600:.1f}")  # README
        f = mtim[mtim.method.isin(["fm", "fmot"])].set_index("method").train_seconds
        if {"fm", "fmot"} <= set(f.index):
            # Share of the FM-OT training spent in the assignment, from the clean benchmark.
            b = bench[(bench.data == "mnist34") & (bench.batch_size == 128)] if len(bench) else bench
            if len(b):
                it = mtim.set_index("method").iterations["fmot"]
                nums.put("mnist-ot-assign", pct(b.median_ms.iloc[0] * it / 1000 / f["fmot"]))

    for _, r in mbg.iterrows():
        nums.put(f"S-mnist-{r.method}-b{int(r.batch_size)}", sig(r.straightness))
    if len(mbg):
        s_b = mbg.pivot(index="batch_size", columns="method", values="straightness").dropna()
        for b, r in s_b.iterrows():
            nums.put(f"S-gain-mnist-b{int(b)}", sig(r["fm"] / r["fmot"], 2))
        if len(s_b) >= 2:
            for method in ("fmot", "fm"):
                slope = np.polyfit(np.log(s_b.index.to_numpy(float)), np.log(s_b[method].to_numpy()), 1)[0]
                nums.put(f"S-slope-mnist-{method}", f"{slope:.2f}")
    extra_t = mbt[mbt.batch_size != mn["batch_size"]] if len(mbt) else mbt
    if len(extra_t):
        gen_b = mbm[mbm.batch_size != mn["batch_size"]].gen_seconds.sum()
        nums.put("hours-mnist-batch", f"{(extra_t.train_seconds.sum() + gen_b) / 3600:.1f}")

    # ------------------------------------------------------------ hypotheses
    hyp = {}
    # H1: off-mode samples of few steps lie on the segments, with the exact field and with FM.
    if len(m2):
        checks = []
        for k in (3, 4, 5):
            for exp, method in (("exact", "exact"), ("main", "fm")):
                on = look.get("on_mode", exp, "8gaussians", method, "euler", k)
                seg = look.get("on_segment", exp, "8gaussians", method, "euler", k)
                if 1 - on >= 0.05:
                    checks.append(seg / (1 - on) >= 0.5)
        if checks:
            hyp["H1"] = verdict(all(checks), any(checks))

        # H2: the off-mode fraction at K = 5 is lowest at an intermediate sigma0 and a very
        # concentrated noise does not remove it (exact field and trained FM).
        def off(exp, method, s0, k=5):
            return 1 - look.get("on_mode", exp, "8gaussians", method, "euler", k, s0)
        res = []
        for exp, method in (("exact", "exact"), ("sigma0", "fm")):
            o = {s0: (off(exp, method, s0) if not (exp == "sigma0" and s0 == toy["sigma0"])
                      else off("main", "fm", s0)) for s0 in toy["sigma0_sweep"]}
            if any(np.isnan(v) for v in o.values()):
                continue
            best = min(o, key=o.get)
            res.append(best not in (min(o), max(o)) and o[min(o)] >= 0.05)
        if res:
            hyp["H2"] = verdict(all(res), any(res))

        # H3: the minibatch size does not change FM but straightens FM-OT, at a cost that grows
        # faster than linearly.
        sw = [b for b in toy["batch_sweep"]]
        s_fmot, s_fm, cost = [], [], []
        for b in sw:
            exp = "main" if b == toy["batch_size"] else "batch"
            for lst, method in ((s_fmot, "fmot"), (s_fm, "fm")):
                g = geo[(geo.experiment == exp) & (geo.dataset == "8gaussians") & (geo.method == method)
                        & (geo.batch_size == b)] if len(geo) else geo
                lst.append(float(g.straightness.iloc[0]) if len(g) else float("nan"))
            t = tim[(tim.experiment == exp) & (tim.dataset == "8gaussians") & (tim.method == "fmot")
                    & (tim.batch_size == b)] if len(tim) else tim
            cost.append(float(t.coupling_seconds.iloc[0]) / b if len(t) else float("nan"))
        if not any(np.isnan(s_fmot + s_fm)):
            fmot_down = all(a > b for a, b in zip(s_fmot, s_fmot[1:]))
            fm_flat = not ratio_relevant(min(s_fm), max(s_fm), tol["straightness"])
            superlinear = cost[-1] > cost[0]  # coupling time per point grows with the minibatch
            hyp["H3"] = verdict(fmot_down and fm_flat and superlinear, fmot_down)
            nums.put("H3-fmot-monotone", "yes" if fmot_down else "no")
            nums.put("S-slope-fmot", f"{np.polyfit(np.log(sw), np.log(s_fmot), 1)[0]:.1f}")
            nums.put("H3-fm-flat", "yes" if fm_flat else "no")

        # H4: with K <= 5, FM-OT beats FM and DDIM in W1 on every mixture.
        wins, total = 0, 0
        for name in DATASETS:
            for k in FEW:
                a = look.get("w1", "main", name, "fmot", "euler", k)
                for method, sampler in (("fm", "euler"), ("ddpm", "ddim")):
                    b = look.get("w1", "main", name, method, sampler, k)
                    if np.isnan(a) or np.isnan(b):
                        continue
                    total += 1
                    wins += int(a < b and ratio_relevant(a, b, tol["w1"]))
        if total:
            hyp["H4"] = verdict(wins == total, wins >= total / 2)
            nums.put("H4-wins", f"{wins}")
            nums.put("H4-total", f"{total}")

        # H5: with 1000 steps the stochastic DDPM sampler is at least as good as the best flow.
        res = []
        for name in DATASETS:
            d = look.get("w1", "main", name, "ddpm", "ddpm", 1000)
            best = min(look.get("w1", "main", name, m, "euler", 1000) for m in FLOWS)
            if not (np.isnan(d) or np.isnan(best)):
                res.append(d <= best or not ratio_relevant(d, best, tol["w1"]))
        if len(mm):
            r = mm[(mm.steps == 1000)]
            d = r[(r.method == "ddpm") & (r.sampler == "ddpm")].fd
            best = r[r.method.isin(FLOWS)].fd
            if len(d) and len(best):
                res.append(float(d.iloc[0]) <= best.min() or not ratio_relevant(float(d.iloc[0]), best.min(), tol["fd"]))
        if res:
            hyp["H5"] = verdict(all(res), any(res))

    # H6: minibatch OT straightens much less on MNIST than in 2D.
    if len(geo) and len(mgeo):
        g = geo[(geo.experiment == "main") & (geo.dataset == "8gaussians")].set_index("method").straightness
        mg = mgeo.set_index("method").straightness
        if {"fm", "fmot"} <= set(g.index) and {"fm", "fmot"} <= set(mg.index):
            r2d, rmn = g["fm"] / g["fmot"], mg["fm"] / mg["fmot"]
            nums.put("S-gain-8g", sig(r2d, 2))
            nums.put("S-gain-mnist", sig(rmn, 2))
            hyp["H6"] = verdict(r2d > (1 + tol["straightness"]) * rmn)
    for k, v in hyp.items():
        nums.put(f"verdict-{k}", v)
    (results / "hypotheses.json").write_text(json.dumps(hyp, indent=2) + "\n")

    # ------------------------------------------------------------ tables
    (tables / "w1_main_table.tex").write_text(w1_main_table(look, ref, grid))
    (tables / "placement_table.tex").write_text(placement_table(look, [1, 2, 3, 4, 5, 10, 1000]))
    for name, (key, _) in DATASETS.items():
        (tables / f"exact_table_{key}.tex").write_text(
            exact_table(look, toy["sigma0_sweep"], [1, 2, 3, 4, 5, 10, 100, 1000], name))
    (tables / "sigma0_table.tex").write_text(
        sweep_table(m2, geo, tim, "sigma0", toy["sigma0_sweep"], r"$\sigma_0$", False))
    (tables / "batch_table.tex").write_text(
        sweep_table(m2, geo, tim, "batch_size", toy["batch_sweep"], "$B$", True))
    (tables / "straightness_table.tex").write_text(straightness_table(geo, mgeo))
    (tables / "mnist_fd_table.tex").write_text(mnist_table(mm, grid, "fd", dist))
    (tables / "mnist_ambiguous_table.tex").write_text(mnist_table(mm, grid, "ambiguous", pct))
    (tables / "mnist_share_table.tex").write_text(mnist_table(mm, grid, "share_3", pct))
    (tables / "mnist_batch_table.tex").write_text(mnist_batch_table(mbm, mbg, mbt, mb_sweep))
    (tables / "compute_table.tex").write_text(compute_table(tim, m2, mtim, mm, mbt, mbm, mn["batch_size"]))
    if len(bench):
        rows = []
        for b in sorted(bench.batch_size.unique()):
            cells = []
            for data in ("8gaussians", "mnist34"):
                r = bench[(bench.data == data) & (bench.batch_size == b)]
                cells.append(sig(r.median_ms.iloc[0], 2) if len(r) else "--")
            rows.append(f"{int(b)} & " + " & ".join(cells) + r" \\")
        (tables / "bench_table.tex").write_text(tabular("@{}rrr@{}", r"$B$ & 8 Gaussians ($d=2$) & MNIST 3 / 4 ($d=784$)", rows))
    (tables / "hyper_table.tex").write_text(hyper_table(raw))
    written = nums.write(tables / "numbers.tex")
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- RESULTS -->", "<!-- /RESULTS -->"
    if start in text and end in text:
        i, j = text.index(start) + len(start), text.index(end)
        text = text[:i] + "\n" + readme_summary(nums, hyp) + "\n" + text[j:]
    start, end = "<!-- TIMES -->", "<!-- /TIMES -->"
    if start in text and end in text:
        i, j = text.index(start) + len(start), text.index(end)
        times = (f"On the Apple M1 of the study, the 2D experiments took {nums.values.get('hours-2d', '--')} hours "
                 f"on three CPU threads and the MNIST experiment {nums.values.get('hours-mnist', '--')} hours on "
                 f"its GPU, {nums.values.get('hours-mnist-eval', '--')} of them in sampling. The sweep of the minibatch "
                 f"on MNIST took another {nums.values.get('hours-mnist-batch', '--')} hours. The compute appendix of the "
                 "report gives the time of every run.")
        text = text[:i] + "\n" + times + "\n" + text[j:]
    readme.write_text(text)
    print(json.dumps(hyp, indent=2))
    print(f"{written} values written to {tables / 'numbers.tex'}")


if __name__ == "__main__":
    main()
