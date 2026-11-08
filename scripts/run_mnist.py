"""One MNIST model (FM-OT, FM, DDPM or RF-2) on the digits 3 and 4, then its evaluation.

The model trains on the training partition and follows its loss on the validation partition.
Training resumes from the last checkpoint if it is interrupted. The evaluation generates n_eval
images per sampler and number of steps from a fixed noise sample and compares them with the test
partition in the space of our classifier: Fréchet distance, share of each digit and fraction of
ambiguous images.

Usage: python scripts/run_mnist.py --method fm --seed 0
A minibatch other than the one of the configuration adds _b<B> to the run and to its result files.
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
from fmstudy.data import mnist_partitions  # noqa: E402
from fmstudy.methods import (DDPMSchedule, PairDataset, ddpm_loss,  # noqa: E402
                             flow_matching_loss, ot_coupling)
from fmstudy.models import MNISTClassifier, SmallUNet, count_parameters  # noqa: E402
from fmstudy.utils import (DATA, RESULTS, RUNS, Timer, append_rows, get_device,  # noqa: E402
                           load_config, seed_everything, synchronize)


def ema_update(ema_params, model_params, decay):
    """Exponential moving average in one fused call (one kernel per group of tensors)."""
    with torch.no_grad():
        torch._foreach_lerp_(ema_params, model_params, 1 - decay)


def reflow_pairs(cfg, device, seed, n_pairs, tag=""):
    """Pairs (noise, image) produced by the FM model of the same seed, cached on disk.

    The last 1000 pairs are kept apart as the validation set of RF-2.
    """
    path = RUNS / "mnist" / f"rf2_pairs_seed{seed}{tag}.pt"
    if path.exists():
        d = torch.load(path)
        return d["x0"], d["x1"], d["seconds"]
    fm = SmallUNet(cfg["unet_channels"]).to(device)
    fm.load_state_dict(torch.load(RUNS / "mnist" / f"fm_seed{seed}{tag}" / "ema.pt", map_location=device))
    fm.eval()
    gen = torch.Generator().manual_seed(5000 + seed)
    x0 = torch.randn(n_pairs + 1000, 1, 28, 28, generator=gen)
    x1 = []
    with Timer(device) as t:
        for i in range(0, len(x0), cfg["eval_batch"]):
            x1.append(samplers.heun(fm, x0[i:i + cfg["eval_batch"]].to(device), cfg["reflow_pair_nfe"]).cpu())
    x1 = torch.cat(x1)
    torch.save({"x0": x0, "x1": x1, "seconds": t.seconds}, path)
    return x0, x1, t.seconds


def main():
    full = load_config()
    cfg = full["mnist"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fm", "fmot", "ddpm", "rf2"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=cfg["iterations"])
    parser.add_argument("--n-eval", type=int, default=cfg["n_eval"])
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--pairs", type=int, default=cfg["reflow_pairs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"],
                        help="minibatch of training and of the FM-OT coupling")
    parser.add_argument("--tag", default="", help="suffix for smoke tests")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="evaluate the saved weights again without training them")
    args = parser.parse_args()
    batch = args.batch_size
    if batch != cfg["batch_size"]:
        args.tag += f"_b{batch}"

    device = get_device()
    out = Path(args.results)
    run_dir = RUNS / "mnist" / f"{args.method}_seed{args.seed}{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sched = DDPMSchedule(**full["ddpm"], device=device)

    parts = mnist_partitions(DATA / "mnist", cfg["digits"], cfg["val_fraction"], cfg["split_seed"])
    x_train = parts["train"][0].to(device)
    x_val = parts["validation"][0].to(device)
    seed_everything(args.seed)
    model = SmallUNet(cfg["unet_channels"]).to(device)
    pair_seconds = 0.0
    if args.method == "rf2":
        x0p, x1p, pair_seconds = reflow_pairs(cfg, device, args.seed, args.pairs, args.tag)
        pairs = PairDataset(x0p[:-1000].to(device), x1p[:-1000].to(device))
        x_val, x0_val_pairs = x1p[-1000:].to(device), x0p[-1000:].to(device)
        model.load_state_dict(torch.load(RUNS / "mnist" / f"fm_seed{args.seed}{args.tag}" / "ema.pt",
                                         map_location=device))
    ema = copy.deepcopy(model)
    if args.evaluate_only:
        # Re-evaluation of the saved weights with the current samplers, without training again.
        ema.load_state_dict(torch.load(run_dir / "ema.pt", map_location=device))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        gen = torch.Generator().manual_seed(100 + args.seed)
        start, elapsed = 0, 0.0

        ckpt_path = run_dir / "checkpoint.pt"
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"])
            # The generator states must stay on the CPU, although map_location moves every tensor.
            gen.set_state(ck["gen"].cpu())
            torch.set_rng_state(ck["torch_rng"].cpu())
            start, elapsed = ck["iteration"], ck["elapsed"]
            print(f"resuming from iteration {start}", flush=True)

        # Fixed noise and times of the validation loss, shared by every iteration.
        gval = torch.Generator().manual_seed(31_000)
        t_val = torch.rand(len(x_val), generator=gval).to(device)
        z_val = torch.randn(x_val.shape, generator=gval).to(device)
        t_int = torch.randint(0, sched.T, (len(x_val),), generator=gval).to(device)
        if args.method == "rf2":
            z_val = x0_val_pairs
        elif args.method == "fmot":
            z_val = torch.cat([ot_coupling(z_val[i:i + batch], x_val[i:i + batch])
                               for i in range(0, len(x_val), batch)])

        @torch.no_grad()
        def val_fn(m):
            m.eval()
            if args.method == "ddpm":
                v = ddpm_loss(m, x_val, sched, t=t_int, eps=z_val)
            else:
                v = flow_matching_loss(m, x_val, x0=z_val, t=t_val)
            m.train()
            return v.item()

        def loss_fn():
            if args.method == "rf2":
                idx = torch.randint(0, pairs.x0.shape[0], (batch,), generator=gen).to(device)
                return flow_matching_loss(model, pairs.x1[idx], x0=pairs.x0[idx])
            idx = torch.randint(0, x_train.shape[0], (batch,), generator=gen).to(device)
            x = x_train[idx]
            if args.method == "ddpm":
                return ddpm_loss(model, x, sched)
            return flow_matching_loss(model, x, coupling="ot" if args.method == "fmot" else "independent")

        log, running = [], torch.zeros((), device=device)
        model.train()
        ema_params, model_params = list(ema.parameters()), list(model.parameters())
        # The accelerator is synchronised only when the clock is read, so that the CPU can queue the
        # next iteration while the GPU computes the current one.
        synchronize(device)
        block_start = time.perf_counter()
        for it in range(start + 1, args.iterations + 1):
            loss = loss_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            ema_update(ema_params, model_params, cfg["ema"])
            running += loss.detach()
            if it % cfg["validate_every"] == 0:
                log.append({"method": args.method, "seed": args.seed, "iteration": it,
                            "train_loss": running.item() / cfg["validate_every"],
                            "val_loss": val_fn(model), "val_loss_ema": val_fn(ema)})
                running.zero_()
            if it % cfg["checkpoint_every"] == 0 or it == args.iterations:
                synchronize(device)
                elapsed += time.perf_counter() - block_start
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                            "gen": gen.get_state(), "torch_rng": torch.get_rng_state(),
                            "iteration": it, "elapsed": elapsed}, ckpt_path)
                append_rows(out / f"mnist_loss{args.tag}.csv", log)
                log = []
                print(f"{args.method} seed {args.seed}: iteration {it}, {elapsed / 60:.1f} min", flush=True)
                block_start = time.perf_counter()
        torch.save(ema.state_dict(), run_dir / "ema.pt")
        append_rows(out / f"mnist_timing{args.tag}.csv", [{
            "method": args.method, "seed": args.seed, "iterations": args.iterations,
            "train_seconds": elapsed, "pair_seconds": pair_seconds,
            "parameters": count_parameters(model), "device": device.type}])

    # Evaluation with the EMA weights.
    ema.eval()
    clf = MNISTClassifier().to(device)
    clf.load_state_dict(torch.load(RUNS / "mnist" / "classifier.pt", map_location=device))
    clf.eval()
    ref = np.load(RUNS / "mnist" / "test_features.npz")
    share = float(ref["share"])
    noise = torch.randn(args.n_eval, 1, 28, 28, generator=torch.Generator().manual_seed(9000 + args.seed))
    if args.method == "ddpm":
        configs = [("ddim", k) for k in full["steps_grid"]] + [("ddpm", k) for k in full["steps_grid"]]
    else:
        configs = [("euler", k) for k in full["steps_grid"]]
    rows, grids = [], {}
    for sampler, steps in configs:
        feats, probs = [], []
        gen_noise = torch.Generator().manual_seed(6000 + args.seed)
        with Timer(device) as t:
            for i in range(0, args.n_eval, cfg["eval_batch"]):
                z = noise[i:i + cfg["eval_batch"]].to(device)
                with torch.no_grad():
                    if sampler == "euler":
                        x = samplers.euler(ema, z, steps)
                    else:
                        x = samplers.ddim(ema, z, steps, sched, eta=0.0 if sampler == "ddim" else 1.0,
                                          generator=gen_noise)
                    x = x.clamp(-1, 1)
                    logits, f = clf(x, return_features=True)
                feats.append(f.cpu())
                probs.append(logits.softmax(1).cpu())
                if i == 0:
                    grids[f"{sampler}_{steps}"] = ((x[:cfg["n_samples_saved"]].cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
        feats = torch.cat(feats).numpy().astype(np.float64)
        probs = torch.cat(probs).numpy()
        rows.append({"method": args.method, "seed": args.seed, "sampler": sampler, "steps": steps,
                     "fd": metrics.frechet_distance(feats, ref["features"]),
                     **metrics.digit_statistics(probs, cfg["digits"], share, cfg["ambiguity_threshold"]),
                     "n": args.n_eval, "gen_seconds": t.seconds})
        print(f"  {sampler} {steps} steps: FD {rows[-1]['fd']:.2f}, ambiguous {rows[-1]['ambiguous']:.3f}",
              flush=True)
    append_rows(out / f"mnist_metrics{args.tag}.csv", rows)
    if args.method != "ddpm":
        z = noise[:cfg["n_straightness"]].to(device)
        s_value, ratio = samplers.straightness(ema, z, cfg["straightness_steps"])
        append_rows(out / f"mnist_straightness{args.tag}.csv", [{
            "method": args.method, "seed": args.seed, "straightness": s_value, "length_ratio": ratio}])
    sample_dir = out / "mnist_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_dir / f"{args.method}_seed{args.seed}{args.tag}.npz", **grids)
    (run_dir / "done").touch()


if __name__ == "__main__":
    main()
