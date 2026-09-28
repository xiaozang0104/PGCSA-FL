# -*- coding: utf-8 -*-
"""
bench_table_methods.py
======================
Per-round COMMUNICATION and COMPUTATION cost of the nine methods of Table IV
(FLTrust, Dnc, PFLF, LSFL, ESFL, Split, FoundationFL, Qin, Ours) as a function
of the number of participating clients N.

* COMPUTATION  : measured wall-clock of the real server-side aggregation code
                 (imported from this project), serial, warm-up + repeats.
* COMMUNICATION: analytic uplink volume per round = N * (upload dimension) * 4 B,
                 following the notation of Table IV (float32; downlink broadcast
                 is the same for every method and is therefore not counted).

Run it from the project directory:

    python bench_table_methods.py                 # N = 20,50,100,200 ; d = 222410
    python bench_table_methods.py --ns 20 --repeats 3
    python bench_table_methods.py --d 454922 --out bench_mnist.csv

Notes / caveats printed at the end of the run (please read them before using the
numbers in the paper).
"""
import argparse, csv, os, platform, statistics as st, sys, time, traceback


def _arg(name, default=None):
    """read one option before numpy is imported (needed for thread pinning)"""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


# ---- pin the BLAS/OpenMP threads BEFORE numpy is imported --------------------
# 1 thread is the most reproducible setting; --threads 0 leaves the environment
# untouched.  The value is FORCED (not setdefault), because containers often ship
# a broken OMP_NUM_THREADS (e.g. 0, which libgomp rejects).  The same setting must
# be used for every method and reported in the figure caption.
_THREADS = _arg("--threads", "1")
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_ENV_BEFORE = {v: os.environ.get(v) for v in _THREAD_ENV}
if _THREADS not in ("0", "none", "None", ""):
    for _v in _THREAD_ENV:
        os.environ[_v] = _THREADS
    _EFFECTIVE_THREADS = _THREADS
else:                                    # leave the environment alone, but warn if broken
    _EFFECTIVE_THREADS = os.environ.get("OMP_NUM_THREADS", "env default")
    try:
        if int(_EFFECTIVE_THREADS) <= 0:
            sys.stderr.write(
                f"warning: OMP_NUM_THREADS={_EFFECTIVE_THREADS} is not a valid thread "
                "count; rerun with --threads 1 for a reproducible setting.\n")
    except ValueError:
        pass

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_project():
    """Find the project root (the directory that holds the aggregator modules).

    Search order:  --project <path>  >  $PGCSA_ROOT  >  current working directory
                   >  the directory this script lives in.
    A candidate is accepted when it contains server_test.py (or fltrust_aggregator.py).
    """
    cands = []
    arg = _arg("--project")
    if arg:
        cands.append(arg)
    if os.environ.get("PGCSA_ROOT"):
        cands.append(os.environ["PGCSA_ROOT"])
    cands += [os.getcwd(), HERE]
    seen = set()
    for c in cands:
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if any(os.path.exists(os.path.join(c, f))
               for f in ("server_test.py", "fltrust_aggregator.py")):
            return c
    return os.path.abspath(cands[0]) if cands else HERE


PROJECT = _resolve_project()
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:                                        # silence the server's loguru output
    from loguru import logger
    logger.remove()
except Exception:
    pass

# --------------------------------------------------------------------------
# real code paths (all imported from the project)
# --------------------------------------------------------------------------
try:
    from fltrust_aggregator import fltrust_aggregate
    from dnc_aggregator import dnc_aggregate
    from lsfl_aggregator import lsfl_aggregate
    from split_aggregator import split_aggregate
    from foundationfl_aggregator import foundationfl_aggregate
    from qin_aggregator import qin_aggregate, generate_jl_matrix as qin_jl
    from server_test import BatchAggrServer
    from client_test import generate_jl_matrix as pgcsa_jl
except ModuleNotFoundError as e:
    sys.stderr.write(
        "\n[bench_table_methods] cannot import the project modules: %s\n"
        f"  project root used : {PROJECT}\n"
        f"  working directory : {os.getcwd()}\n"
        f"  script directory  : {HERE}\n"
        "  -> either run the script from the project directory (the one that contains\n"
        "     server_test.py / fltrust_aggregator.py), or pass\n"
        "         --project <path-to-project>\n"
        "     A candidate is only accepted when server_test.py or fltrust_aggregator.py\n"
        "     is present in it.\n\n" % e)
    raise SystemExit(2)

IMPORT_ERRORS = {}
try:
    from run_pflf import PFLFServer
except Exception as e:                      # pragma: no cover
    PFLFServer, IMPORT_ERRORS["PFLF"] = None, f"{type(e).__name__}: {e}"
try:
    from run_esfl import ESFLServer
except Exception as e:                      # pragma: no cover
    ESFLServer, IMPORT_ERRORS["ESFL"] = None, f"{type(e).__name__}: {e}"


def _mean_fallback(deltas):
    """PFLF/ESFL server-side aggregation is literally np.mean of the updates; used only
    when their module cannot be imported (e.g. torch missing on the benchmark machine)."""
    return np.mean(np.stack(deltas, axis=0), axis=0)

MAL_RATIO = 0.2          # malicious fraction used by every aggregation rule
K_DEFAULT = 256          # compressed dimension of our scheme
QIN_K = 64               # Qin's JL dimension (Table IV / run_experiments.py)

# upload dimension per client, per Table IV  ('k' = compressed, 'd' = full model)
UPLOAD_DIM = {
    "FLTrust": "d", "Dnc": "d", "PFLF": "d", "LSFL": "d", "ESFL": "k",
    "Split": "d", "FoundationFL": "d", "Qin": "d",
    "Ours": "k",
}


def make_inputs(N, d, k, seed=0):
    """unit-norm updates of dimension d, normalized compressed updates of dimension k,
    one dense JL matrix (as implemented) and one genuinely sparse JL matrix (Table IV)."""
    rng = np.random.RandomState(seed)
    G = rng.standard_normal((N, d)).astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    ref = G.mean(axis=0).astype(np.float32)
    Z = rng.standard_normal((N, k)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    A = pgcsa_jl(d, k, rng)                                   # dense k x d (as implemented)
    p = 1.0 / np.sqrt(k)
    A_sp = sp.random(k, d, density=p, format="csr",
                     data_rvs=lambda s: rng.choice([1, -1], size=s)) * np.sqrt(1.0 / p)
    return dict(G=G, ref=ref, Z=Z, A=A, A_sp=A_sp)


def build_calls(inp, N, d, k):
    """name -> callable() performing one server-side aggregation round"""
    G, ref, Z, A = inp["G"], inp["ref"], inp["Z"], inp["A"]
    deltas = [G[i] for i in range(N)]
    projs = [Z[i] for i in range(N)]
    f = max(1, int(N * MAL_RATIO))
    calls = {}

    calls["FLTrust"] = lambda: fltrust_aggregate(deltas, ref)
    calls["Dnc"] = lambda: dnc_aggregate(deltas, f, n_iters=1, c=1, b=10000)
    calls["LSFL"] = lambda: lsfl_aggregate(deltas, k_frac=0.7)
    calls["Split"] = lambda: split_aggregate(deltas)
    calls["FoundationFL"] = lambda: foundationfl_aggregate(deltas, m=N // 2, c=None,
                                                           rule="trimmed_mean")
    calls["Qin"] = lambda: qin_aggregate(deltas, ref, jl_matrix=qin_jl(d, QIN_K, seed=42))
    impl = {}
    if PFLFServer is not None:
        calls["PFLF"] = lambda: PFLFServer._aggregate(None, deltas)
        impl["PFLF"] = "project"
    else:
        calls["PFLF"] = lambda: _mean_fallback(deltas)
        impl["PFLF"] = "mean-fallback"
    if ESFLServer is not None:
        calls["ESFL"] = lambda: ESFLServer._aggregate(None, deltas)
        impl["ESFL"] = "project"
    else:
        calls["ESFL"] = lambda: _mean_fallback(deltas)
        impl["ESFL"] = "mean-fallback"

    # our scheme: the pipeline of Step 5 / Table IV with the compressed-domain averaging,
    # i.e. detect on the compressed updates, average the accepted ones, and reconstruct
    # that single average with the pseudoinverse A^+ = A^T (A A^T)^{-1} (a k x k solve,
    # precomputed once, followed by the sparse product with ||A||_0 = d sqrt(k) non-zeros).
    At = inp["A_sp"].T.tocsr()
    Ginv = np.linalg.inv((inp["A_sp"] @ inp["A_sp"].T).toarray())

    def ours_c2():
        srv = BatchAggrServer(num_clients=N, reduced_dim=k, noise_std=0.1,
                              attack_mode="none", malicious_ratio=MAL_RATIO,
                              threshold_multiplier=2.0, data_distribution="iid",
                              dirichlet_alpha=0.5, dataset="GTSRB", r=2, beta=0.1, seed=42)
        srv.jl_matrix = A
        out = srv.detect_anomalies(projs)
        outliers = set(int(x) for x in out) if isinstance(out, (list, tuple, set, np.ndarray)) else set()
        good = [i for i in range(N) if i not in outliers] or list(range(N))
        z_bar = np.mean([projs[i] for i in good], axis=0)
        return np.asarray(At @ (Ginv @ z_bar)).ravel()

    calls["Ours"] = ours_c2
    impl.setdefault("Ours", "project (avg+recon)")
    return calls, impl


def timeit(fn, warmup=2, repeats=5, budget_s=3.0):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
        if sum(ts) / 1e3 > budget_s and len(ts) >= 3:
            break
    return st.median(ts), (st.stdev(ts) if len(ts) > 1 else 0.0), min(ts), len(ts)


def reference_task(reps=3):
    """fixed-size fp32 matmul, timed at the start and at the end of the run:
    if the two values differ by more than ~10% the machine drifted (thermal or a
    noisy neighbour) and the run should be repeated."""
    rng = np.random.RandomState(0)
    A = rng.standard_normal((1500, 1500)).astype(np.float32)
    B = rng.standard_normal((1500, 1500)).astype(np.float32)
    A @ B
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        A @ B
        ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)


def prewarm(calls):
    """call every method once (untimed) so lazy imports / first allocations do not
    land inside the first timed method"""
    for fn in calls.values():
        try:
            fn()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None,
                    help="path to the PGCSA-FL project (default: $PGCSA_ROOT or the current "
                         "working directory). Put this option FIRST when the project is elsewhere.")
    ap.add_argument("--ns", default="20,50,100,200", help="client counts, comma separated")
    ap.add_argument("--ds", default="", help="gradient dimensions for the d-sweep "
                                            "(fixed N = first entry of --ns), comma separated")
    ap.add_argument("--d", type=int, default=222410, help="gradient dimension of the N-sweep")
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="compressed dimension")
    ap.add_argument("--float-bytes", type=int, default=4, help="4 = float32")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", default="1",
                    help="BLAS/OpenMP threads; 1 is the most reproducible (default), "
                         "0 leaves the environment untouched. Must be the same for every method.")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_table_methods.csv"))
    args = ap.parse_args()
    Ns = [int(x) for x in args.ns.split(",") if x.strip()]
    ds = [int(x) for x in args.ds.split(",") if x.strip()]

    print("=" * 100)
    print("PER-ROUND COMMUNICATION AND COMPUTATION COST  (serial run)")
    print(f"  machine : {platform.node()} | {platform.processor() or platform.machine()} | "
          f"python {platform.python_version()}")
    print(f"  config  : d={args.d}  k={args.k}  float={args.float_bytes}B  "
          f"malicious={MAL_RATIO:.0%}  repeats={args.repeats}  "
          f"threads={_EFFECTIVE_THREADS} (requested {args.threads}; "
          f"OMP_NUM_THREADS before the run: {_ENV_BEFORE.get('OMP_NUM_THREADS')})")
    print(f"  N grid  : {Ns}" + (f"   d grid: {ds} (at N={Ns[0]})" if ds else ""))
    for m, e in IMPORT_ERRORS.items():
        print(f"  !! {m} could not be imported -> {e}")
    ref_start = reference_task()
    print(f"  reference task (fp32 1500x1500 matmul) before the run: {ref_start:.2f} ms")
    print("=" * 100)

    rows = []
    FIELDS = ["sweep", "method", "N", "d", "k", "comp_median_ms", "comp_std_ms",
              "comp_min_ms", "repeats", "comm_uplink_MB", "impl", "status"]

    def dump():                       # write after every grid point -> survives interruptions
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    grid = [("N", N, args.d) for N in Ns] + [("d", Ns[0], d) for d in ds]
    for sweep, N, d in grid:
        inp = make_inputs(N, d, args.k)
        calls, impl = build_calls(inp, N, d, args.k)
        prewarm(calls)                    # untimed: lazy imports / first allocations
        for name, fn in calls.items():
            dim = args.k if UPLOAD_DIM.get(name, "d") == "k" else d
            comm_mb = N * dim * args.float_bytes / 1e6
            try:
                med, sd, mn, n = timeit(fn, repeats=args.repeats)
                rows.append(dict(sweep=sweep, method=name, N=N, d=d, k=args.k,
                                 comp_median_ms=round(med, 3), comp_std_ms=round(sd, 3),
                                 comp_min_ms=round(mn, 3), repeats=n,
                                 comm_uplink_MB=round(comm_mb, 4),
                                 impl=impl.get(name, "project"), status="ok"))
            except Exception as e:
                rows.append(dict(sweep=sweep, method=name, N=N, d=d, k=args.k,
                                 comp_median_ms="", comp_std_ms="", comp_min_ms="",
                                 repeats=0, comm_uplink_MB=round(comm_mb, 4),
                                 impl=impl.get(name, "project"),
                                 status=f"ERROR: {type(e).__name__}: {e}"))
                print(f"  [{sweep}] N={N:<4} d={d:<7} {name:<24} ERROR {type(e).__name__}: {e}")
                traceback.print_exc(limit=1)
        done = [r for r in rows if r["N"] == N and r["d"] == d]
        tag = f"N = {N}" if sweep == "N" else f"d = {d} (N = {N})"
        print(f"\n--- {tag} ---")
        for r in sorted(done, key=lambda r: (r["comp_median_ms"] == "", r["comp_median_ms"] or 0)):
            c = f'{r["comp_median_ms"]:>10.3f} ms' if r["comp_median_ms"] != "" else "        n/a"
            print(f'   {r["method"]:<24} comp {c}   comm {r["comm_uplink_MB"]:>8.3f} MB/round')
        dump()
        sys.stdout.flush()

    dump()
    ref_end = reference_task()
    drift = (ref_end - ref_start) / ref_start * 100 if ref_start else 0.0
    print(f"\nreference task after the run: {ref_end:.2f} ms "
          f"(before: {ref_start:.2f} ms, drift {drift:+.1f}%)")
    if abs(drift) > 10:
        print("  !! the machine drifted by more than 10% during the run "
              "(thermal throttling or a noisy neighbour): repeat the run and use "
              "the second one, or compare the two.")
    print(f"written: {args.out}  ({len(rows)} rows)")

    meta = os.path.splitext(args.out)[0] + "_meta.txt"
    with open(meta, "w", encoding="utf-8") as fh:
        fh.write(f"machine      : {platform.node()} | "
                 f"{platform.processor() or platform.machine()} | {platform.platform()}\n"
                 f"python       : {platform.python_version()}\n"
                 f"project root : {PROJECT}\n"
                 f"config       : d={args.d} k={args.k} float={args.float_bytes}B "
                 f"malicious={MAL_RATIO:.0%} repeats={args.repeats} "
                 f"threads={args.threads} "
                 f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'env default')}) "
                 f"N grid={Ns} d grid={ds}\n"
                 f"drift check  : fp32 1500x1500 matmul {ref_start:.2f} ms before, "
                 f"{ref_end:.2f} ms after -> {drift:+.1f}%\n"
                 f"scope        : server-side aggregation only, mechanism level, serial; "
                 f"local training / data loading / network excluded\n"
                 f"communication: uplink only, float32; downlink broadcast identical for all\n")
        for m, e in IMPORT_ERRORS.items():
            fh.write(f"import note  : {m} -> {e} (timed with the mean fallback)\n")
    print(f"written: {meta}")

    print("""
-------------------------------------------------------------------------------
Accounting (copy the wording into the figure caption)
  * COMPUTATION: server-side aggregation only, mechanism level; local training,
    data loading and the network are excluded; serial single process, warm-up
    discarded, median of the repeats.
  * COMMUNICATION: uplink model updates only, float32; the downlink broadcast
    (d floats) is identical for every method and is therefore not counted.
  * ESFL and Qin follow the accounting of Table IV (ESFL uploads k floats,
    Qin uploads d masked floats).
Two things to watch
  1) "Ours" = the pipeline written in Table IV / Step 5 with the compressed-domain
     averaging (detect on the compressed updates, average the accepted ones,
     reconstruct that average once), timed with a genuinely sparse A so that it
     matches the O(k N + d sqrt(k)) cost of the table.
  2) The implemented JL matrix is DENSE (client_test.generate_jl_matrix ends with
     .toarray()), so the client-side projection costs O(k d), not O(d sqrt(k));
     fix that before plotting any client-side cost.
-------------------------------------------------------------------------------""")


if __name__ == "__main__":
    main()
