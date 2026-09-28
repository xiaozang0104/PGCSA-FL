"""Effect of r on accuracy and on loss (GTSRB, non-IID, targeted adaptive attack).

Data : results/par_sweep/r_sweep_attack_GTSRB_r{1,2,3,5,10,15,30}_mal{0.2,0.4}_seed{42,123,2026}_non_iid.csv
       (100 rounds, 3 seeds; avg20 of these files is the r block of Table II)
Output: pictures/accuracy of r.pdf  and  pictures/loss of r.pdf   (the two files referenced by
        experiment.tex) plus .svg/.png previews.

Design: one curve per r, mean over the three seeds, the adopted value r = 2 drawn as a thick
solid red line and every other value as its own dash pattern, so the figure also works when the
paper is printed in black and white.  Four representative values of r are drawn (all seven make
the panel unreadable because a single run that collapses for a few rounds drags the three-seed
mean down by 30 pp); the complete sweep is reported in Table II of the paper.
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = r"D:\Pycharm_Project\PGCSA-FL\results\par_sweep"
OUTDIR = r"D:\Pycharm_Project\PGCSA-FL\pictures"

# ---------------------------------------------------------------- style knobs
MAL = "0.2"                   # threat level used by the figure caption (GTSRB, non-IID, T-Adaptive)
R_ALL = [1, 2, 3, 5, 10, 15, 30]      # complete sweep (reported in Table II)
R_VALUES = [1, 2, 10, 30]             # four representative values drawn in the figure: with all
                                      # seven the panel becomes unreadable, because a single run
                                      # that collapses for a few rounds drags the three-seed mean
                                      # down by 30 pp and produces vertical streaks
ADOPTED = 2                   # highlighted value
SEEDS = [42, 123, 2026]
SHOW_BANDS = False            # True -> shaded +-1 std band over the three seeds (busy: the band
                              # reaches +-25 pp in the rounds where one run collapses)
BAND_ALPHA = 0.12
FIGSIZE = (3.6, 2.9)
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 7
LEGEND_NCOL = 2
LINE_WIDTH = 1.5
ADOPTED_LINE_WIDTH = 2.8
MARKER_SIZE = 4.0
MARKER_EVERY = 12
ADOPTED_COLOR = "#d62728"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
# the adopted value keeps the solid style; the others get one dash pattern each
LINESTYLES = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (6, 2, 1, 2)), (0, (9, 3)),
              (0, (2, 1.4)), (0, (6, 2, 1, 2, 1, 2)), (0, (1, 3))]
MARKERS = ["X", "o", "s", "^", "d", "*", "h", "P"]
YLIM_ACC = (0, 100)
YLIM_LOSS = (0, 4.2)
XLABEL = "Communication round"
# -----------------------------------------------------------------------------


def series(v, mal, seed, col):
    p = os.path.join(SWEEP, f"r_sweep_attack_GTSRB_r{v}_mal{mal}_seed{seed}_non_iid.csv")
    with open(p, newline="") as f:
        return [float(r[col]) for r in csv.DictReader(f)]


def mean_std(v, mal, col):
    runs = [series(v, mal, s, col) for s in SEEDS]
    scale = 100.0 if col == "accuracy" else 1.0
    n = len(runs[0])
    mean = [st.mean(run[i] for run in runs) * scale for i in range(n)]
    std = [st.stdev([run[i] for run in runs]) * scale for i in range(n)]
    return mean, std


def draw(ax, col, ylabel, ylim, legend_loc, legend_anchor):
    rounds = list(range(1, 101))
    order = sorted(R_VALUES, key=lambda v: v != ADOPTED)      # adopted value drawn last (on top)
    for k, v in enumerate(order):
        mean, std = mean_std(v, MAL, col)
        adopted = (v == ADOPTED)
        color = ADOPTED_COLOR if adopted else COLORS[k % len(COLORS)]
        ls = "-" if adopted else LINESTYLES[(k + 1) % len(LINESTYLES)]
        if SHOW_BANDS:
            ax.fill_between(rounds, [m - s for m, s in zip(mean, std)],
                            [m + s for m, s in zip(mean, std)],
                            color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2 if adopted else 1)
        ax.plot(rounds, mean, label=f"$r={v}$" + (" (adopted)" if adopted else ""),
                color=color, linestyle=ls,
                linewidth=ADOPTED_LINE_WIDTH if adopted else LINE_WIDTH,
                marker=MARKERS[k % len(MARKERS)], markersize=MARKER_SIZE if adopted else MARKER_SIZE - 1,
                markevery=MARKER_EVERY, markeredgecolor="white" if adopted else color,
                markeredgewidth=0.6 if adopted else 0,
                zorder=5 if adopted else 3)
        print(f"  r={v:<3} {col}: round 100 = {mean[-1]:.2f}  sd(last 20) = "
              f"{st.mean(std[-20:]):.2f}")
    ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xlim(1, 100)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, fontsize=LEGEND_FONTSIZE,
              ncol=LEGEND_NCOL, frameon=True, framealpha=0.85, labelspacing=0.2,
              handlelength=1.8, handletextpad=0.4, borderpad=0.25, columnspacing=0.8)


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), format=ext, dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


os.makedirs(OUTDIR, exist_ok=True)
for col, ylabel, ylim, name, loc, anchor in [
        ("accuracy", "Accuracy (%)", YLIM_ACC, "accuracy of r", "lower right", (0.99, 0.02)),
        ("loss", "Loss", YLIM_LOSS, "loss of r", "upper right", (0.99, 0.99))]:
    print(f"\n=== {name} (threat {MAL}) ===")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw(ax, col, ylabel, ylim, loc, anchor)
    fig.tight_layout(pad=0.4)
    save(fig, name)

print("\nfigures written to", OUTDIR)
