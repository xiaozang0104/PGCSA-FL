# PGCSA-FL

Reference implementation of **PGCSA-FL**: clients compress their model update with a sparse
Johnson–Lindenstrauss projection (`k = 256` instead of `d` parameters) and add Gaussian noise
for differential privacy, so a round carries only `k` values per client; the server estimates
a low-dimensional subspace of the received projections, scores each client by its projection
energy, filters anomalous updates with mirrored median-anchored thresholds, averages the
surviving compressed updates and reconstructs that single average back to the full dimension.

## Files

| Group | Files |
|---|---|
| our scheme | `client_test.py`, `server_test.py`, `common_client.py`, `common_server.py` |
| data, models, attacks | `dataset.py`, `utils.py`, `model.py`, `model_MNIST2.py`, `model_cifar10.py`, `common_attack.py` |
| drivers | `run_experiments.py` (all methods), `generate_table.py` (LaTeX tables) |
| baselines | `*_aggregator.py` (13 rules) and `run_*.py` (one entry per method) |
| sweeps | `run_r_sweep.py`, `run_k_sweep.py`, `run_beta_sweep.py`, `run_threshold_sweep.py`, `run_c_compare_cifar10.py` |
| complexity | `bench_table_methods.py` (server cost + uplink)
| figures | `plot_stacked.py`, `plot_panels.py`, `plot_r_sweep.py` |

## Install and run

```bash
pip install -r requirements.txt
python run_experiments.py                     # accuracy of all methods  -> results/*.csv
python generate_table.py                      # LaTeX tables from results/*.csv
```

Datasets (MNIST, GTSRB, CIFAR-10) are loaded by `dataset.py`; put them next to the scripts
(see the paths at the top of that file).

## Complexity measurements

```bash
# server-side cost and uplink volume per round
python bench_table_methods.py --ns 20,50,100,200 \
       --ds 100000,222410,455000,800000,1000000 --repeats 5 --out bench_final.csv
python plot_stacked.py --csv bench_final.csv --outdir figs --layout wide
```

## Default configuration

20 clients, 100 rounds, 5 local epochs, batch 64, SGD with lr 0.01 and momentum 0.9, Dirichlet
non-IID with α = 0.5, `k = 256` with row sparsity `p = 1/√k` and non-zeros `±√(1/p)`,
Gaussian noise σ = 0.1, `r = 2`, `β = 0.1`, thresholds
`τ_lo = s_med + (s_med − s_max)`, `τ_hi = s_med + (s_med − s_min)`; seeds 42/123/2026, accuracy
averaged over the last 20 rounds.

## Notes

* The baselines are faithful re-implementations of the **defense logic** of their papers. The
  cryptographic front ends of LSFL, Split, ShieldFL and Qin (masking, secret sharing, Paillier)
  are not part of the released reference implementations and are not executed; Qin's client
  masking and LSFL's two-share splitting are.
* All costs are measured in a single process with one thread on synthetic updates of the real
  shape, so they reflect the aggregation rule itself, not the surrounding framework.
