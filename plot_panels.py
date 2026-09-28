# -*- coding: utf-8 -*-
"""
plot_panels.py — the four sub-panels of the two complexity figures, as separate PDFs,
matching the style of the combined figures (same series, colours, axis labels and ticks).

    python plot_panels.py --csv bench_server.csv --outdir figs

Outputs (each ~3.4 in wide, for \\subfloat with width=0.48\\textwidth inside figure*):
    fig_cal_a.pdf    computation vs N        fig_cal_b.pdf    computation vs d
    fig_comm_a.pdf   communication vs N      fig_comm_b.pdf   communication vs d
"""
import argparse, csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_OURS, C_BASE, C_BAND = "#d62728", "#1f77b4", "#9ecae1"
SERIES = [("Ours", "Ours", C_OURS, "X", "-", 2.4, 10),
          ("FLTrust", "FLTrust", "#1f77b4", "o", "-", 1.3, 5),
          ("Dnc", "Dnc", "#2ca02c", "s", "-", 1.3, 5),
          ("PFLF", "PFLF", "#17becf", "v", "-", 1.3, 5),
          ("LSFL", "LSFL", "#9467bd", "^", "-", 1.3, 5),
          ("ESFL", "ESFL", "#8c564b", "D", "-", 1.3, 5),
          ("Split", "Split", "#e377c2", "P", "-", 1.4, 5),
          ("FoundationFL", "FoundationFL", "#7f7f7f", "*", "-", 1.4, 5),
          ("Qin", "Qin", "#bcbd22", "<", "-", 1.3, 5)]
BAND = []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "ok":
                continue
            rows.append(dict(sweep=r.get("sweep", "N"), method=r["method"], N=int(r["N"]),
                             d=int(r["d"]), ms=fnum(r["comp_median_ms"]),
                             sd=fnum(r["comp_std_ms"]) or 0.0,
                             comm=fnum(r.get("comm_uplink_MB"))))
    return rows


def series(rows, sweep, method, key):
    xkey = "N" if sweep == "N" else "d"
    pts = sorted([r for r in rows if r["sweep"] == sweep and r["method"] == method],
                 key=lambda r: r[xkey])
    xs = [p[xkey] for p in pts]
    if sweep == "d":
        xs = [x / 1000.0 for x in xs]
    return xs, [p[key] for p in pts], [p["sd"] for p in pts]


def panel(ax, rows, sweep, comm):
    if comm:
        # every method gets its own curve and legend entry; the full-dimensional methods
        # coincide by construction (they all upload N x d values) and ESFL coincides with ours
        for m, lab, c, mk, ls, lw, z in SERIES:
            xs, ys, _ = series(rows, sweep, m, "comm")
            if ys:
                ax.plot(xs, ys, color=c, marker=mk, linestyle="-",
                        linewidth=2.2 if m == "Ours" else lw, markersize=5,
                        label=lab if m == "Ours" else m, zorder=z)
    else:
        for i, m in enumerate(BAND):
            xs, ys, _ = series(rows, sweep, m, "ms")
            if ys:
                ax.plot(xs, ys, color="#b0b0b0", marker="o", linestyle="-", linewidth=1.0,
                        markersize=3.5,
                        label="other baselines" if i == 0 else "_nolegend_", zorder=2)
        for m, lab, c, mk, ls, lw, z in SERIES:
            xs, ys, er = series(rows, sweep, m, "ms")
            if ys:
                ax.errorbar(xs, ys, yerr=er, color=c, marker=mk, linestyle=ls, linewidth=lw,
                            markersize=5, capsize=2, label=lab, zorder=z)
    ax.set_yscale("log")
    if sweep == "d":
        ticks = sorted({r["d"] / 1000.0 for r in rows if r["sweep"] == "d"})
        ax.set_xticks(ticks)
        ax.set_xticklabels([("1M" if round(t) >= 1000 else f"{round(t):g}k") for t in ticks])
    ax.set_xlabel("Number of participating clients $N$" if sweep == "N"
                  else "Gradient dimension $d$", fontsize=9.5)
    ax.set_ylabel("Uplink volume per round (bytes)" if comm
                  else "Aggregation time per round (ms)", fontsize=9.5)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.45)
    ax.tick_params(labelsize=8.5)
    # state the quantity that is held fixed in this panel (corner chosen to avoid the data)
    if comm and sweep == "d":          # ours sits at the bottom here -> use the top-left
        ax.text(0.02, 0.97, "$N=20$ fixed", transform=ax.transAxes, ha="left", va="top",
                fontsize=7, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))
    else:
        ax.text(0.98, 0.02, "$d=222{,}410$ fixed" if sweep == "N" else "$N=20$ fixed",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))


def build(rows, outdir, name, sweep, comm):
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    panel(ax, rows, sweep, comm)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2 if comm else 3, fontsize=6.4, frameon=False,
              handletextpad=0.35, columnspacing=0.8, handlelength=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("written:", os.path.join(outdir, f"{name}.pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    out = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out, exist_ok=True)
    rows = load(args.csv)
    if not rows:
        sys.exit("no usable rows")
    build(rows, out, "fig_cal_a", "N", comm=False)
    build(rows, out, "fig_cal_b", "d", comm=False)
    build(rows, out, "fig_comm_a", "N", comm=True)
    build(rows, out, "fig_comm_b", "d", comm=True)


if __name__ == "__main__":
    main()
