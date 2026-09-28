# PGCSA-FL

Reference implementation of **PGCSA-FL**: clients compress their model update with a sparse
Johnson–Lindenstrauss projection (`k = 256` instead of `d` parameters) and add Gaussian noise
for differential privacy, so a round carries only `k` values per client; the server estimates
a low-dimensional subspace of the received projections, scores each client by its projection
energy, filters anomalous updates with mirrored median-anchored thresholds, averages the
surviving compressed updates and reconstructs that single average back to the full dimension.

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/                        # all code
│   ├── client_test.py, server_test.py          # our client and server
│   ├── common_client.py, common_server.py      # base client and training loop
│   ├── common_attack.py                        # the four poisoning attacks
│   ├── dataset.py, utils.py, model*.py         # data, helpers, the three models
│   ├── run_experiments.py, generate_table.py   # drivers: run everything / build LaTeX tables
│   ├── *_aggregator.py, run_*.py               # the 13 baselines and their entry points
│   ├── run_*_sweep.py, run_c_compare_cifar10.py# sensitivity and proportion experiments
│   ├── bench_table_methods.py                  # per-round server cost and uplink volume
│   └── plot_stacked.py, plot_panels.py, plot_r_sweep.py, make_fig_excel.py
├── results/                    # created at run time (git-ignored)
└── figs/                       # created at run time (git-ignored)
```

## Install and run

```bash
pip install -r requirements.txt

python src/run_experiments.py      # accuracy of all methods  -> results/*.csv
python src/generate_table.py       # LaTeX tables from results/*.csv
```

Datasets (MNIST, GTSRB, CIFAR-10) are loaded by `src/dataset.py`; put them in the repository
root (see the paths at the top of that file). Run the commands from the repository root.

## Complexity measurements

The two figures of the complexity section (per-round uplink volume and per-round
gradient filtration–aggregation time) are produced by

```bash
# per-round server-side cost and uplink volume of every method
python src/bench_table_methods.py --project src --ns 20,50,100,200 \
       --ds 100000,222410,455000,800000,1000000 --repeats 5 --out bench_final.csv

# the figures themselves (both panels in one row), and the Excel used to draw them
python src/plot_stacked.py --csv bench_final.csv --outdir figs --layout wide
python src/plot_panels.py  --csv bench_final.csv --outdir figs
python src/make_fig_excel.py bench_final.csv figs/fig_data.xlsx
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
