# -*- coding: utf-8 -*-
"""
plot_stacked.py — single-column, two-row (stacked) versions of the two complexity figures.

    python plot_stacked.py --csv bench_server.csv --outdir figs

Produces
    fig_cal_stack.pdf/png    computation: (top) vs N, (bottom) vs d
    fig_comm_stack.pdf/png   communication: (top) vs N, (bottom) vs d

Design choices requested by the authors:
  * one column wide, the two panels stacked in two rows;
  * a compact legend of one or two rows: only the curves that the text discusses are
    labelled, the remaining baselines appear as a light band (their min--max range) and
    are described in the body text.
"""
import argparse, csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_OURS = "#d62728"
C_RECON = "#d62728"
C_DNC = "#2ca02c"
C_QIN = "#bcbd22"
C_SPLIT = "#e377c2"
C_FF = "#7f7f7f"
C_BAND = "#9ecae1"
C_BASE = "#1f77b4"

# every method gets its own colour / marker (no grouping)
ALL_METHODS = ["Ours", "FLTrust", "Dnc", "PFLF", "LSFL", "ESFL", "Split", "FoundationFL", "Qin"]
STYLE = {
    "Ours":         ("#d62728", "X", 2.4, 10),
    "FLTrust":      ("#1f77b4", "o", 1.3, 4),
    "Dnc":          ("#2ca02c", "s", 1.3, 4),
    "PFLF":         ("#17becf", "v", 1.3, 4),
    "LSFL":         ("#9467bd", "^", 1.3, 4),
    "ESFL":         ("#8c564b", "D", 1.3, 4),
    "Split":        ("#e377c2", "P", 1.4, 4),
    "FoundationFL": ("#7f7f7f", "*", 1.4, 4),
    "Qin":          ("#bcbd22", "<", 1.3, 4),
}


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
                             comm=fnum(r.get("comm_uplink_MB"))))
    return rows


def series(rows, sweep, xkey, method=None, key="ms"):
    sel = [r for r in rows if r["sweep"] == sweep and (method is None or r["method"] == method)]
    pts = sorted(sel, key=lambda r: r[xkey])
    xs = [p[xkey] for p in pts]
    if sweep == "d":
        xs = [x / 1000.0 for x in xs]
    return xs, [p[key] for p in pts]


def plot_panel(ax, rows, sweep, ylab, comm=False, first=False):
    xkey = "N" if sweep == "N" else "d"
    if comm:
        # one curve and one legend entry per method; the full-dimensional methods coincide
        # by construction (all upload N x d values) and ESFL coincides with ours (N x k)
        for m in ALL_METHODS:
            xs, ys = series(rows, sweep, xkey, m, "comm")
            if not ys:
                continue
            c, mk, lw, z = STYLE.get(m, (None, "o", 1.2, 3))
            ax.plot(xs, ys, color=c, marker=mk, linestyle="-",
                    linewidth=2.4 if m == "Ours" else lw, markersize=4, label=m, zorder=z)
    else:
        for m in ALL_METHODS:                       # every method gets its own curve + entry
            xs, ys = series(rows, sweep, xkey, m)
            if not ys:
                continue
            c, mk, lw, z = STYLE.get(m, (None, "o", 1.2, 3))
            ax.plot(xs, ys, color=c, marker=mk, linestyle="-", linewidth=lw,
                    markersize=4, label=m, zorder=z)
    ax.set_yscale("log")
    if sweep == "d":
        ticks = sorted({r["d"] / 1000.0 for r in rows if r["sweep"] == "d"})
        ax.set_xticks(ticks)
        ax.set_xticklabels([("1M" if round(t) >= 1000 else f"{round(t):g}k") for t in ticks])
    ax.set_xlabel("Number of clients $N$" if sweep == "N" else "Gradient dimension $d$",
                  fontsize=9 if False else 11, labelpad=1.5)
    ax.set_ylabel(ylab, fontsize=9, labelpad=1.5)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=8, pad=1.5, length=2.5)
    if comm and sweep == "d":          # ours sits at the bottom here -> use the top-left
        ax.text(0.02, 0.97, "$N=20$ fixed", transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.0))
    else:
        ax.text(0.98, 0.02, "$d=222{,}410$ fixed" if sweep == "N" else "$N=20$ fixed",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.0))
    return ax


def build(rows, outdir, name, ylab, comm):
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.3))
    plot_panel(axes[0], rows, "N", ylab, comm=comm)
    plot_panel(axes[1], rows, "d", ylab, comm=comm)
    h, l = axes[0].get_legend_handles_labels()
    ncol = 2 if comm else 4
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=5.9, frameon=False,
               bbox_to_anchor=(0.5, -0.045), handletextpad=0.3, columnspacing=0.7,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.045 if comm else 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("written:", os.path.join(outdir, f"{name}.pdf"))


def build_wide(rows, outdir, name, ylab, comm):
    """one row: two panels side by side (the layout the authors want for fig_cal/fig_comm)"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    plot_panel(axes[0], rows, "N", ylab, comm=comm)
    plot_panel(axes[1], rows, "d", ylab, comm=comm)
    h, l = axes[0].get_legend_handles_labels()
    ncol = 2 if comm else 5
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=6.3, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handletextpad=0.35, columnspacing=0.8,
               handlelength=1.6)
    fig.tight_layout(rect=(0, 0.10 if comm else 0.13, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("written:", os.path.join(outdir, f"{name}.pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--layout", choices=["both", "wide", "stack"], default="both")
    args = ap.parse_args()
    out = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out, exist_ok=True)
    rows = load(args.csv)
    if not rows:
        sys.exit("no usable rows")
    if args.layout in ("both", "wide"):
        build_wide(rows, out, "fig_cal", "Aggregation time (ms)", comm=False)
        build_wide(rows, out, "fig_comm", "Uplink volume / round (bytes)", comm=True)
    if args.layout in ("both", "stack"):
        build(rows, out, "fig_cal_stack", "Aggregation time (ms)", comm=False)
        build(rows, out, "fig_comm_stack", "Uplink volume / round (bytes)", comm=True)


if __name__ == "__main__":
    main()
