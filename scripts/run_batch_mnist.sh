#!/bin/sh
# Minibatch sweep on MNIST: FM-OT and FM (the control, since with a fixed number of iterations a
# larger minibatch also shows more data) with B = 256 and 512. B = 128 is the main run.
# Usage: PY=python3 sh scripts/run_batch_mnist.sh
set -e
PY=${PY:-python3}
cd "$(dirname "$0")/.."
for b in 256 512; do
  for m in fmot fm; do
    [ -f runs/mnist/${m}_seed0_b${b}/done ] || $PY scripts/run_mnist.py --method $m --seed 0 --batch-size $b
  done
done
echo "minibatch sweep finished"
