"""Classifier that defines the evaluation space on MNIST.

It trains a small CNN on the ten digits of the official training set, except the validation
images of the two digits of the study, which no model sees during training. It reports the test
accuracy, stores the 128-dimensional features of the test images of the two digits and computes
the reference values between real images (a sample of the training partition against the test
partition), which are the best values a generator can attain.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmstudy import metrics  # noqa: E402
from fmstudy.data import load_mnist, mnist_partitions  # noqa: E402
from fmstudy.models import MNISTClassifier  # noqa: E402
from fmstudy.utils import (DATA, RESULTS, RUNS, append_rows, get_device,  # noqa: E402
                           load_config, seed_everything)


@torch.no_grad()
def features_and_probs(model, x, device, batch=1000):
    model.eval()
    feats, probs = [], []
    for i in range(0, len(x), batch):
        logits, f = model(x[i:i + batch].to(device), return_features=True)
        feats.append(f.cpu())
        probs.append(logits.softmax(1).cpu())
    return torch.cat(feats).numpy().astype(np.float64), torch.cat(probs).numpy()


def main():
    cfg = load_config()["mnist"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg["classifier_epochs"])
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    device = get_device()
    seed_everything(0)
    digits = cfg["digits"]
    parts = mnist_partitions(DATA / "mnist", digits, cfg["val_fraction"], cfg["split_seed"])
    x_all, y_all = load_mnist(DATA / "mnist", "train")
    keep = torch.ones(len(x_all), dtype=torch.bool)
    keep[parts["validation_index"]] = False
    x_train, y_train = x_all[keep], y_all[keep]
    x_test_all, y_test_all = load_mnist(DATA / "mnist", "test")

    model = MNISTClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["classifier_lr"])
    gen = torch.Generator().manual_seed(0)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_train), generator=gen)
        for i in range(0, len(perm), cfg["classifier_batch"]):
            idx = perm[i:i + cfg["classifier_batch"]]
            loss = F.cross_entropy(model(x_train[idx].to(device)), y_train[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        _, p = features_and_probs(model, x_test_all, device)
        print(f"epoch {epoch + 1}: test accuracy {(p.argmax(1) == y_test_all.numpy()).mean():.4f}", flush=True)

    _, p_all = features_and_probs(model, x_test_all, device)
    x_test, y_test = parts["test"]
    feat_test, p_test = features_and_probs(model, x_test, device)
    out_dir = RUNS / "mnist"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "classifier.pt")
    share = float((y_test.numpy() == digits[0]).mean())
    np.savez_compressed(out_dir / "test_features.npz", features=feat_test, labels=y_test.numpy(),
                        share=share)

    x_tr, _ = parts["train"]
    idx = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(1))[:cfg["n_eval"]]
    feat_ref, p_ref = features_and_probs(model, x_tr[idx], device)
    real = metrics.digit_statistics(p_ref, digits, share, cfg["ambiguity_threshold"])
    test_stats = metrics.digit_statistics(p_test, digits, share, cfg["ambiguity_threshold"])
    append_rows(Path(args.results) / "mnist_reference.csv", [{
        "n_train": len(parts["train"][0]), "n_validation": len(parts["validation"][0]),
        "n_test": len(x_test), "n_classifier_train": len(x_train),
        "classifier_accuracy_all": float((p_all.argmax(1) == y_test_all.numpy()).mean()),
        "classifier_accuracy_digits": float((p_test.argmax(1) == y_test.numpy()).mean()),
        "test_share": share, "test_ambiguous": test_stats["ambiguous"],
        "fd_train_vs_test": metrics.frechet_distance(feat_ref, feat_test),
        "real_ambiguous": real["ambiguous"], "real_share_error": real["share_error"],
        "n_reference": len(idx)}])
    print(f"accuracy {(p_test.argmax(1) == y_test.numpy()).mean():.4f} on the digits {digits}", flush=True)


if __name__ == "__main__":
    main()
