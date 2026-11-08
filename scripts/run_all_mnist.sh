#!/bin/sh
# Every MNIST run in order: classifier, FM-OT, then the three standards of comparison.
# RF-2 needs the FM model of the same seed, so it runs last. Each run resumes if interrupted.
# Usage: PY=python3 sh scripts/run_all_mnist.sh
set -e
PY=${PY:-python3}
cd "$(dirname "$0")/.."
[ -f runs/mnist/classifier.pt ] || $PY scripts/train_classifier.py
for m in fmot fm ddpm rf2; do
  [ -f runs/mnist/${m}_seed0/done ] || $PY scripts/run_mnist.py --method $m --seed 0
done
echo "all MNIST runs finished"
