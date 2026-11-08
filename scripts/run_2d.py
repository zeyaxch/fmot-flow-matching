"""Experiments on the 2D mixtures: the main comparison and the two sweeps of FM-OT.

main    every mixture, sigma0 = 1 and minibatch 256: FM-OT against FM, RF-2 and DDPM.
sigma0  eight Gaussians, FM and FM-OT with very concentrated to very wide noise.
batch   eight Gaussians, FM and FM-OT with minibatches of 16 to 1024 points.

Every model starts from the same initialisation, sees fresh samples of the mixture at each
iteration and is evaluated on a fixed test sample. A fixed validation sample follows the loss
during training.

Usage: python scripts/run_2d.py --experiment main [--datasets 8gaussians] [--values 0.3 3.0]
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmstudy import metrics, samplers  # noqa: E402
from fmstudy.data import get_mixture  # noqa: E402
from fmstudy.methods import (DDPMSchedule, PairDataset, ddpm_loss,  # noqa: E402
                             flow_matching_loss, ot_coupling)
from fmstudy.models import MLP2D  # noqa: E402
from fmstudy.utils import (RESULTS, RUNS, Timer, append_rows, load_config,  # noqa: E402
                           seed_everything)

FLOWS = ("fm", "fmot", "rf2")
FIGURE_STEPS = (1, 2, 3, 5, 10, 1000)


def coupled_noise(x1, x0, coupling, batch_size):
    """Noise for a set of data points, reordered by minibatch OT in chunks of batch_size."""
    if coupling == "ot":
        x0 = torch.cat([ot_coupling(x0[i:i + batch_size], x1[i:i + batch_size])
                        for i in range(0, len(x1), batch_size)])
    return x0


def train(model, loss_fn, val_fn, iterations, lr, every, log, tag):
    """Adam loop shared by every method. Returns the training time in seconds."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    running = 0.0
    with Timer() as timer:
        for it in range(1, iterations + 1):
            loss = loss_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if it % every == 0:
                with torch.no_grad():
                    val = val_fn(model).item()
                log.append({**tag, "iteration": it, "train_loss": running / every, "val_loss": val})
                running = 0.0
    return timer.seconds


def run(cfg, dataset, sigma0, batch_size, seed, methods, experiment, out, iterations,
        evaluate_only=False):
    toy = cfg["toy"]
    mix = get_mixture(dataset)
    sched = DDPMSchedule(**cfg["ddpm"])
    n_test, lr, every = toy["n_test"], toy["lr"], toy["validate_every"]
    test, _ = mix.sample(n_test, np.random.default_rng(10_000))           # test partition
    val, _ = mix.sample(toy["n_validation"], np.random.default_rng(30_000))  # validation partition
    val = torch.from_numpy(val)
    tag = {"experiment": experiment, "dataset": dataset, "sigma0": sigma0, "batch_size": batch_size,
           "seed": seed}
    rng = np.random.default_rng(seed)  # training data: fresh samples at every iteration

    def data_batch():
        return torch.from_numpy(mix.sample(batch_size, rng)[0])

    run_dir = RUNS / "2d" / experiment / f"{dataset}_s{sigma0}_b{batch_size}_seed{seed}"
    if evaluate_only:
        # Re-evaluation of the saved models with the current samplers, without training again.
        models = {}
        for method in methods:
            models[method] = MLP2D(toy["mlp_hidden"], toy["mlp_layers"], toy["time_dim"])
            models[method].load_state_dict(torch.load(run_dir / f"{method}.pt"))
    else:
        if experiment == "main":
            real, _ = mix.sample(n_test, np.random.default_rng(20_000 + seed))
            append_rows(out / "2d_reference.csv", [{
                **tag, "w1_real_real": metrics.w1_exact(test, real),
                **{f"real_{k}": v for k, v in metrics.mode_statistics(real, mix).items()},
                **{f"real_{k}": v for k, v in metrics.placement_statistics(real, mix).items()}}])

        seed_everything(seed)
        init = MLP2D(toy["mlp_hidden"], toy["mlp_layers"], toy["time_dim"])
        models, log, timing = {}, [], []
        # Fixed noise and times of the validation loss, shared by every method and iteration.
        gval = torch.Generator().manual_seed(31_000)
        t_val = torch.rand(len(val), generator=gval)
        z_val = torch.randn(val.shape, generator=gval)
        t_int = torch.randint(0, sched.T, (len(val),), generator=gval)
        for method in [m for m in methods if m != "rf2"]:
            model = copy.deepcopy(init)
            torch.manual_seed(1000 + seed)
            coupling_time = [0.0]
            if method == "ddpm":
                def loss_fn(m=model):
                    return ddpm_loss(m, data_batch(), sched)

                def val_fn(m):
                    return ddpm_loss(m, val, sched, t=t_int, eps=z_val)
            else:
                coupling = "ot" if method == "fmot" else "independent"
                x0_val = coupled_noise(val, sigma0 * z_val, coupling, batch_size)

                def loss_fn(m=model, c=coupling, ct=coupling_time):
                    x1 = data_batch()
                    x0 = sigma0 * torch.randn_like(x1)
                    if c == "ot":
                        start = time.perf_counter()
                        x0 = ot_coupling(x0, x1)
                        ct[0] += time.perf_counter() - start
                    return flow_matching_loss(m, x1, x0=x0)

                def val_fn(m, x0_val=x0_val):
                    return flow_matching_loss(m, val, x0=x0_val, t=t_val)
            secs = train(model, loss_fn, val_fn, iterations, lr, every, log, {**tag, "method": method})
            timing.append({**tag, "method": method, "train_seconds": secs,
                           "coupling_seconds": coupling_time[0], "pair_seconds": 0.0})
            models[method] = model

        if "rf2" in methods:
            # Reflow (Liu et al., 2023): pairs (noise, FM sample) generated with Heun, then training
            # from the FM weights on those fixed pairs. 2000 extra pairs form its validation set.
            torch.manual_seed(2000 + seed)
            x0_pairs = sigma0 * torch.randn(toy["reflow_pairs"] + len(val), 2)
            with Timer() as t_pairs:
                x1_pairs = samplers.heun(models["fm"], x0_pairs, toy["reflow_pair_nfe"])
            pairs = PairDataset(x0_pairs[:-len(val)], x1_pairs[:-len(val)])
            x0_pv, x1_pv = x0_pairs[-len(val):], x1_pairs[-len(val):]
            gen = torch.Generator().manual_seed(3000 + seed)
            rf2 = copy.deepcopy(models["fm"])

            def rf2_loss(m=rf2):
                b0, b1 = pairs.batch(batch_size, gen)
                return flow_matching_loss(m, b1, x0=b0)

            def rf2_val(m):
                return flow_matching_loss(m, x1_pv, x0=x0_pv, t=t_val)

            torch.manual_seed(1000 + seed)
            secs = train(rf2, rf2_loss, rf2_val, iterations, lr, every, log, {**tag, "method": "rf2"})
            timing.append({**tag, "method": "rf2", "train_seconds": secs, "coupling_seconds": 0.0,
                           "pair_seconds": t_pairs.seconds})
            models["rf2"] = rf2

        run_dir.mkdir(parents=True, exist_ok=True)
        for method, model in models.items():
            torch.save(model.state_dict(), run_dir / f"{method}.pt")
        append_rows(out / "2d_loss.csv", log)
        append_rows(out / "2d_training.csv", timing)

    # Evaluation on a fixed noise sample shared by every method and number of steps.
    z = torch.randn(n_test, 2, generator=torch.Generator().manual_seed(4000 + seed))
    rows, geo, fig = [], [], {"test": test}
    for method, model in models.items():
        model.eval()
        configs = ([("ddim", k) for k in cfg["steps_grid"]] + [("ddpm", k) for k in cfg["steps_grid"]]
                   if method == "ddpm" else [("euler", k) for k in cfg["steps_grid"]])
        for sampler, steps in configs:
            with Timer() as t_gen:
                if sampler == "euler":
                    x = samplers.euler(model, sigma0 * z, steps)
                else:
                    g = torch.Generator().manual_seed(6000 + seed)
                    x = samplers.ddim(model, z, steps, sched, eta=0.0 if sampler == "ddim" else 1.0,
                                      generator=g)
            x = x.numpy()
            rows.append({**tag, "method": method, "sampler": sampler, "steps": steps,
                         "w1": metrics.w1_exact(x, test), **metrics.mode_statistics(x, mix),
                         **metrics.placement_statistics(x, mix), "gen_seconds": t_gen.seconds})
            if steps in FIGURE_STEPS:
                fig[f"samples_{method}_{sampler}_{steps}"] = x
        if method in FLOWS:
            s_value, ratio = samplers.straightness(model, sigma0 * z, toy["straightness_steps"])
            geo.append({**tag, "method": method, "straightness": s_value, "length_ratio": ratio})
            _, path, _ = samplers.euler(model, sigma0 * z[:toy["n_trajectories"]], 100, return_path=True)
            fig[f"path_{method}"] = path.numpy()
    append_rows(out / "2d_metrics.csv", rows)
    append_rows(out / "2d_straightness.csv", geo)
    fig_dir = out / "2d_samples"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{experiment}_{dataset}_s{sigma0}_b{batch_size}.npz"
    if evaluate_only and path.exists():  # keep the samples of the methods not evaluated again
        fig = {**dict(np.load(path)), **fig}
    np.savez_compressed(path, **fig)
    print(f"{experiment} {dataset} sigma0={sigma0} batch={batch_size}: done", flush=True)


def main():
    cfg = load_config()
    toy = cfg["toy"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "sigma0", "batch"], required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--values", nargs="+", type=float, default=None,
                        help="sigma0 values or minibatch sizes of the sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    parser.add_argument("--iterations", type=int, default=toy["iterations"])
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--methods", nargs="+", default=None, help="subset of the methods of the experiment")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="evaluate the saved models again without training them")
    args = parser.parse_args()

    torch.set_num_threads(toy["threads"])
    out = Path(args.results)
    for seed in args.seeds:
        if args.experiment == "main":
            for name in args.datasets or toy["datasets"]:
                run(cfg, name, toy["sigma0"], toy["batch_size"], seed,
                    tuple(args.methods or ("fm", "fmot", "ddpm", "rf2")), "main", out,
                    args.iterations, args.evaluate_only)
        elif args.experiment == "sigma0":
            values = args.values or [v for v in toy["sigma0_sweep"] if v != toy["sigma0"]]
            for s0 in values:
                run(cfg, "8gaussians", float(s0), toy["batch_size"], seed, tuple(args.methods or ("fm", "fmot")),
                    "sigma0", out, args.iterations, args.evaluate_only)
        else:
            values = args.values or [v for v in toy["batch_sweep"] if v != toy["batch_size"]]
            for b in values:
                run(cfg, "8gaussians", toy["sigma0"], int(b), seed, tuple(args.methods or ("fm", "fmot")),
                    "batch", out, args.iterations, args.evaluate_only)


if __name__ == "__main__":
    main()
