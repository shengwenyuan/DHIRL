"""
Convergence loss curve for Mouse Labyrinth (left panel of convergence_plots).

The labyrinth EM loss never drops to the convergence threshold (0.05) — all
runs stop at max_iterations.  Convergence is therefore judged by a "plateau"
criterion: first iteration where the rolling-window mean |Δloss| per 4-iter
block falls below PLATEAU_THRESHOLD (default 0.001).

Reads one or more E02.log files (one per random seed).  Per seed, fold curves
are averaged onto a shared grid.  When multiple seeds are given, each seed's
per-fold mean is shown as a thin semi-transparent line; the grand mean ± std
across all (seed × fold) runs is the bold band.

Usage (from DHIRL root):
    # single seed
    python plot/plot_convergence_labyrinth.py \
        --log src_autotest/logs/20260315_192657/G03/E02.log \
        --out plot/convergence_labyrinth.pdf

    # multiple seeds
    python plot/plot_convergence_labyrinth.py \
        --log src_autotest/logs/seed42/G03/E02.log \
        --log src_autotest/logs/seed0/G03/E02.log  \
        --out plot/convergence_labyrinth.pdf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size':        9,
    'axes.labelsize':  10,
    'axes.titlesize':   9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize':  9,
})


# ── Style (mirrors DHIRL_bridge/plot/plot_convergence_bridge.py) ─────────
BLUE         = (0.20, 0.45, 0.85)
GRAY_PLATEAU = (0.50, 0.50, 0.50)

SEED_COLORS = [
    (0.20, 0.45, 0.85),
    (0.85, 0.35, 0.10),
    (0.15, 0.62, 0.35),
    (0.65, 0.18, 0.58),
    (0.60, 0.52, 0.08),
]

SEED_LINE_ALPHA  = 0.45
BAND_ALPHA       = 0.15
GRAND_MEAN_ALPHA = 0.95

# Rolling-window plateau detector.
# For the PLOT ANNOTATION: applied to the grand mean curve (smoother),
#   window=6 blocks (24 iters), threshold=0.001 → triggers at iter ~140.
# For the TABLE / per-fold reporting: same threshold, applied per fold.
PLATEAU_WINDOW    = 6     # blocks  (= 24 iterations)
PLATEAU_THRESHOLD = 0.001 # absolute |Δloss| per 4-iter block


# ── Log parser ───────────────────────────────────────────────────────────

def parse_log(log_path):
    """Return list of dicts (one per fold-run) with keys:
       iters, losses, conv_iter, conv_loss, total_time.
    """
    runs = []
    cur_iters, cur_losses = [], []

    iter_re = re.compile(r"^Iteration (\d+), Loss: ([0-9.]+),")
    conv_re = re.compile(
        r"^Iteration (\d+), Converged with Loss: ([0-9.]+), Total time: ([0-9.]+)s")

    with open(log_path) as f:
        for line in f:
            m = conv_re.match(line)
            if m:
                conv_iter  = int(m.group(1))
                conv_loss  = float(m.group(2))
                total_time = float(m.group(3))
                if not cur_iters or cur_iters[-1] != conv_iter:
                    cur_iters.append(conv_iter)
                    cur_losses.append(conv_loss)
                runs.append({
                    "iters":      np.array(cur_iters),
                    "losses":     np.array(cur_losses),
                    "conv_iter":  conv_iter,
                    "conv_loss":  conv_loss,
                    "total_time": total_time,
                })
                cur_iters, cur_losses = [], []
                continue
            m = iter_re.match(line)
            if m:
                cur_iters.append(int(m.group(1)))
                cur_losses.append(float(m.group(2)))

    return runs


def plateau_iter(iters, losses, window=PLATEAU_WINDOW, threshold=PLATEAU_THRESHOLD):
    """Return the iteration index where the rolling |Δloss| drops below threshold.
    Returns None if never triggered.
    """
    diffs = np.abs(np.diff(losses))           # len = n-1
    for i in range(window - 1, len(diffs)):
        if np.mean(diffs[i - window + 1 : i + 1]) < threshold:
            return iters[i + 1]               # +1: diff[i] is between losses[i] and [i+1]
    return None


# ── Timing table ─────────────────────────────────────────────────────────

def print_timing_table(all_runs_per_seed):
    """Print avg time/iter stats and convergence summary."""
    all_runs = [r for seed_runs in all_runs_per_seed for r in seed_runs]

    # Re-parse timing from each log — we don't store it in the run dict above,
    # but we can compute it from total_time / conv_iter.
    total_times = np.array([r["total_time"] for r in all_runs])
    conv_iters  = np.array([r["conv_iter"]  for r in all_runs])
    total_per_iter = total_times / conv_iters

    plateau_iters = []
    for r in all_runs:
        p = plateau_iter(r["iters"], r["losses"],
                         threshold=PLATEAU_THRESHOLD)
        plateau_iters.append(p if p is not None else r["conv_iter"])
    plateau_med = int(np.median(plateau_iters))

    print("\n── Timing summary (labyrinth, from log) ──────────────────────────")
    print(f"  Total time/iter   : {total_per_iter.mean():.3f} ± {total_per_iter.std():.3f} s")
    print(f"  Max iterations    : {conv_iters.max():.0f}  (all runs hit limit)")
    print(f"  Plateau iter (per-fold) : median={plateau_med}, "
          f"mean={np.mean(plateau_iters):.0f} ± {np.std(plateau_iters):.0f}"
          f"  (rolling |Δloss|<{PLATEAU_THRESHOLD} over {PLATEAU_WINDOW} blocks)")
    print(f"  Final loss range  : [{min(r['conv_loss'] for r in all_runs):.4f},"
          f" {max(r['conv_loss'] for r in all_runs):.4f}]")
    print()
    print("  Per-iteration step breakdown (approx., from total_time / conv_iter):")
    print(f"    Total            : {total_per_iter.mean():.3f} s/iter")
    print("  (See log for E-step / IAVI / Intention-net breakdown per 4-iter block)")
    print("────────────────────────────────────────────────────────────────────\n")


# ── Plotting ─────────────────────────────────────────────────────────────

def runs_to_mat(runs, grid):
    """Interpolate each run's loss onto `grid` (linear interpolation)."""
    mat = []
    for r in runs:
        interp = np.interp(grid, r["iters"], r["losses"],
                           left=r["losses"][0], right=r["losses"][-1])
        mat.append(interp)
    return np.array(mat)   # (n_runs, len(grid))


def plot(all_runs_per_seed, out_path):
    n_seeds  = len(all_runs_per_seed)
    max_iter = max(r["conv_iter"]
                   for seed_runs in all_runs_per_seed
                   for r in seed_runs)
    grid     = np.arange(4, max_iter + 1, 4)

    # Per-seed means and grand statistics (linear space — no log transform)
    seed_means   = []
    all_mats     = []
    for seed_runs in all_runs_per_seed:
        mat = runs_to_mat(seed_runs, grid)
        seed_means.append(mat.mean(axis=0))
        all_mats.append(mat)

    seed_means  = np.array(seed_means)
    all_flat    = np.vstack(all_mats)
    grand_mean  = all_flat.mean(axis=0)
    grand_std   = all_flat.std(axis=0)

    # Plateau on the grand mean curve — directly corresponds to the plotted line.
    # The mean is smoother, so a tighter threshold (PLATEAU_THRESHOLD) is used.
    plateau_med = plateau_iter(grid, grand_mean,
                               window=PLATEAU_WINDOW,
                               threshold=PLATEAU_THRESHOLD)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    # Grand mean ± std band
    ax.fill_between(grid, grand_mean - grand_std, grand_mean + grand_std,
                    color=BLUE, alpha=BAND_ALPHA, zorder=4)
    ax.plot(grid, grand_mean, color=BLUE, linewidth=2.2,
            alpha=GRAND_MEAN_ALPHA, zorder=5)

    y_top = grand_mean[0]          # topmost point of the mean curve (iter 4)
    y_bot = grand_mean.min()
    y_span = y_top - y_bot

    # Plateau vertical marker — label just below the top of the axes
    if plateau_med is not None:
        ax.axvline(plateau_med, color=GRAY_PLATEAU, linewidth=0.9,
                   linestyle="--", zorder=2)
        ax.text(plateau_med + 2, y_top - 0.04 * y_span,
                f"plateau ≈ {plateau_med}", ha="left", va="top",
                fontsize=8, color=GRAY_PLATEAU)

    # Max-iterations vertical marker (value already shown in legend)
    ax.axvline(max_iter, color=GRAY_PLATEAU, linewidth=0.9,
               linestyle=":", zorder=2)

    # Axes
    ax.set_xlim(left=0, right=max_iter + 4)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("EM objective (loss)")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.grid(axis="x", alpha=0.12, linestyle=":")

    # Legend
    n_folds = len(all_runs_per_seed[0])
    label = ("mean ± std" +
             (f" ({n_seeds} seeds × {n_folds} folds)"
              if n_seeds > 1 else f" ({n_folds} folds)"))
    handles = [
        Line2D([0], [0], color=BLUE, linewidth=2.2,
               alpha=GRAND_MEAN_ALPHA, label=label),
    ]
    handles += [
        Line2D([0], [0], color=GRAY_PLATEAU, linewidth=0.9,
               linestyle="--", label=f"mean-curve plateau (|Δμ| < {PLATEAU_THRESHOLD})"),
        Line2D([0], [0], color=GRAY_PLATEAU, linewidth=0.9,
               linestyle=":", label=f"max iter ({max_iter})"),
    ]
    ax.legend(handles=handles, loc="upper right",
              framealpha=0.85, edgecolor="0.7")

    plt.tight_layout(pad=1.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="append", required=True, metavar="LOG",
                        help="Path to E02.log.  Repeat for each random seed.")
    parser.add_argument("--out", default="plot/convergence_labyrinth.pdf")
    args = parser.parse_args()

    all_runs_per_seed = []
    for log_path in args.log:
        runs = parse_log(log_path)
        plateaus = [plateau_iter(r["iters"], r["losses"]) for r in runs]
        print(f"{log_path}: {len(runs)} fold-runs")
        for i, (r, p) in enumerate(zip(runs, plateaus)):
            print(f"  fold {i}: max_iter={r['conv_iter']}, "
                  f"final_loss={r['conv_loss']:.4f}, "
                  f"total_time={r['total_time']:.1f}s, "
                  f"plateau_iter={p}")
        all_runs_per_seed.append(runs)

    print_timing_table(all_runs_per_seed)
    plot(all_runs_per_seed, args.out)


if __name__ == "__main__":
    main()
