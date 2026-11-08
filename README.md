# Flow matching with minibatch optimal transport: straight paths, few steps and samples between the modes

Course project of *Generative AI* (Máster en Inteligencia Artificial, UAM, 2026).
Authors: Zeya Xu Chen and Adrián Gonzalo San Martín.

The study is centred on FM-OT, flow matching whose noise and data are paired by optimal
transport inside each minibatch (Tong et al., 2024, Algorithm 3; Pooladian et al., 2023). We
explain why few integration steps leave the samples of flow matching on the segments between
neighbouring modes, measure how the minibatch size controls the straightness of FM-OT and its cost,
and compare FM-OT from 1 to 1000 steps with three standards trained with the same network,
initialisation and budget:

| Method | Training | Sampling |
|---|---|---|
| FM-OT | flow matching with exact minibatch OT pairs (Tong et al., 2024, Alg. 3), Jonker–Volgenant assignment (SciPy) | Euler |
| FM | flow matching with independent pairs (Lipman et al., 2023, Eq. 23; Liu et al., 2023, Eq. 1) | Euler |
| RF-2 | one reflow step on (noise, sample) pairs of FM, generated with Heun (Liu et al., 2023, Alg. 1) | Euler |
| DDPM | noise prediction, linear schedule from 1e-4 to 0.02, T = 1000 (Ho et al., 2020, Alg. 1) | DDIM with η = 0 and stochastic sampler with η = 1 (Song et al., 2021, Eqs. 12 and 16), on K steps from T − 1 with the trailing spacing (Lin et al., 2024, Sec. 3.3) |

The 2D experiments use three Gaussian mixtures, where W1, the modes and the exact velocity field
are computed in closed form. The image experiments use the digits 3 and 4 of MNIST and a Fréchet
distance in the feature space of our own classifier.

## Main results

<!-- RESULTS -->
W1 to the 2000 test points of the eight Gaussians (lower is better, one seed). Two real samples of the same size reach 0.0655.

| Method | K = 1 | K = 2 | K = 3 | K = 5 | K = 10 | K = 1000 |
|---|---:|---:|---:|---:|---:|---:|
| FM-OT | 0.179 | 0.0967 | 0.0895 | 0.0864 | 0.0862 | 0.0870 |
| FM | 1.34 | 0.602 | 0.347 | 0.172 | 0.122 | 0.110 |
| RF-2 | 0.133 | 0.128 | 0.129 | 0.126 | 0.126 | 0.125 |
| DDPM, eta = 0 (DDIM) | 11.0 | 1.02 | 0.954 | 0.582 | 0.296 | 0.218 |
| DDPM, eta = 1 | 11.0 | 1.20 | 0.788 | 0.481 | 0.254 | 0.158 |

Fréchet distance on MNIST 3 / 4 in the features of our classifier (2000 training images reach 6.62).

| Method | K = 1 | K = 2 | K = 3 | K = 5 | K = 10 | K = 1000 |
|---|---:|---:|---:|---:|---:|---:|
| FM-OT | 676 | 145 | 79.9 | 45.8 | 28.6 | 18.1 |
| FM | 1691 | 461 | 175 | 74.4 | 34.4 | 15.6 |
| RF-2 | 74.4 | 51.1 | 44.4 | 38.1 | 32.6 | 25.6 |
| DDPM, eta = 0 (DDIM) | 948 | 586 | 275 | 126 | 67.3 | 41.6 |
| DDPM, eta = 1 | 948 | 682 | 352 | 135 | 47.2 | 18.6 |

* Few Euler steps leave the samples on the posterior mean of the data, between neighbouring modes, also with the exact field and no network. With two steps FM-OT places 71% of its samples on a mode of the eight Gaussians, against 0% for FM.
* FM-OT lowers the straightness from 1.15 to 0.0154 on the eight Gaussians (from 54.6 to 28.7 on MNIST), and from two steps on it has the lowest W1 in 18 of the 21 combinations of mixture and steps.
* Its straightness falls from 0.123 with B = 16 to 0.00454 with B = 1024, while one assignment costs 0.24 ms with B = 64 and 148 ms with B = 1024 (one thread).
* Hypotheses: H1 partly supported, H2 supported, H3 supported, H4 supported, H5 partly supported, H6 supported.
<!-- /RESULTS -->

## Data, partitions and hyperparameters

| | Training | Validation | Test | Generated |
|---|---|---|---|---|
| Each 2D mixture | fresh samples at every iteration (20 000 × 256) | 2000, fixed | 2000, fixed | 2000 per sampler and step count |
| MNIST, digits 3 and 4 | 10 776 (5514 / 5262) | 1197 (617 / 580) | 1992 (1010 / 982) | 2000 per sampler and step count |
| Classifier of the MNIST metrics | 58 803 (ten digits, minus our validation) | – | 10 000 (official test) | – |

The validation partitions only follow the loss during training (`results/*_loss.csv`). No
checkpoint or hyperparameter is selected with them, and the test partitions are used only in the
final evaluation. Every hyperparameter, with its value and the exact place where it comes from
(paper, published code or our decision), is in [`configs/experiments.json`](configs/experiments.json)
and in Appendix A of the report. The main ones:

| | 2D | MNIST 3 / 4 | Origin |
|---|---|---|---|
| Network | MLP, 4 × 256 SiLU, 215 042 parameters | U-Net, 2 resolutions, 32 channels, 410 657 parameters | ours; Ronneberger et al. (2015) |
| Optimiser | Adam, rate 1e-3 | Adam, rate 2e-4, clipping 1, weight average 0.999 | Kingma and Ba (2015); Ho et al. (2020), App. B and code |
| Minibatch (also the OT problem) | 256 | 128 | gan_toy.py of Gulrajani et al. (2017); Ho et al. (2020), App. B |
| Iterations | 20 000 | 10 000 | ours (Apple M1 budget) |
| Noise | N(0, σ0² I), σ0 = 1 | N(0, I) | Lipman et al. (2023); Tong et al. (2024) |
| Sampling steps | 1, 2, 3, 4, 5, 10, 100, 1000 | same | supervisor |
| Seeds | 1 | 1 | ours; relevance threshold from preliminary runs with 3 seeds |

## Repository layout

```
fmstudy/            library: data, networks, objectives, integrators and metrics
  data.py           Gaussian mixtures in 2D, MNIST reader and partitions
  models.py         MLP for 2D, small U-Net and classifier for MNIST
  methods.py        flow matching loss, minibatch OT coupling, exact field of a mixture, DDPM
  samplers.py       Euler, Heun (reflow pairs only), generalised DDIM with eta, straightness
  metrics.py        exact W1, modes, placement between modes, Fréchet distance, digit statistics
configs/            hyperparameters, each one with its origin
scripts/            experiments, analysis and figures (one script per stage)
tests/              sixteen tests with closed-form results
notebooks/          executed tutorial on FM-OT
results/            raw results (CSV), sample files and hypotheses.json
figures/            every figure of the report, made by scripts/make_figures.py
runs/               final weights of every model, the classifier and its test features
```

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

We used Python 3.14.0 and PyTorch 2.14.0 on an Apple M1 with 8 GB (MPS backend for MNIST, CPU
for 2D). Any machine with PyTorch 2.2 or later works, and CUDA is used automatically if available.

## Data

The 2D mixtures are generated by the code. MNIST is read from the original IDX files of the
official PyTorch mirror, without torchvision:

```bash
mkdir -p data/mnist && cd data/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -O https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
done
```

## Reproducing every result

```bash
.venv/bin/python -m unittest discover -s tests  # sixteen tests with closed-form results
PY=.venv/bin/python sh scripts/run_all_2d.sh     # exact field, main comparison, sweeps of sigma0 and B
PY=.venv/bin/python sh scripts/run_all_mnist.sh  # classifier, FM-OT, FM, DDPM and RF-2 on MNIST 3 / 4
PY=.venv/bin/python sh scripts/run_batch_mnist.sh  # FM-OT and FM on MNIST 3 / 4 with B = 256 and 512
.venv/bin/python scripts/run_mnist.py --method ddpm --evaluate-only  # evaluate the weights of runs/ again (also run_2d.py)
.venv/bin/python scripts/benchmark_coupling.py   # cost of one assignment, on an idle machine
.venv/bin/python scripts/analyze.py              # tables, values of the report, hypotheses, this README
.venv/bin/python scripts/make_figures.py         # every figure
```

<!-- TIMES -->
On the Apple M1 of the study, the 2D experiments took 2.0 hours on three CPU threads and the MNIST experiment 4.6 hours on its GPU, 1.5 of them in sampling. The sweep of the minibatch on MNIST took another 8.7 hours. The compute appendix of the report gives the time of every run.
<!-- /TIMES -->

`runs/` holds the final weights of the 24 models in 2D, the moving averages of the eight MNIST
models and the classifier with its test features, so `--evaluate-only` reproduces every
evaluation without training. The training scripts overwrite these weights, and
`run_all_mnist.sh` reuses the classifier if it exists.

Every random draw uses a fixed seed. The MNIST runs save a checkpoint every 2000 iterations and
resume automatically.
On MPS, PyTorch does not guarantee bitwise determinism, so MNIST numbers may change slightly in
the last digits between machines.

## Traceability

* Every hyperparameter in `configs/experiments.json` records its origin.
* Every number in the report is produced by `scripts/analyze.py` (LaTeX tables and keyed values written to
  `../informe/tables`, the folder of the report next to this repository) and every figure by `scripts/make_figures.py`.
* The hypotheses and the test that decides each one were fixed before the final runs, and negative
  results are reported as such in `results/hypotheses.json` and in the report.

## License and use of AI

Code under the MIT license. MNIST is distributed by its authors (LeCun et al., 1998).
The use of generative AI tools in this project is declared in the report.
