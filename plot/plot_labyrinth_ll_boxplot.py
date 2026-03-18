import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

# ── Configure entries here ───────────────────────────────────────────────────
# Each entry is either:
#   ("display name", "path/to/file.csv")   – reads test_ll from first 5 rows
#   ("display name", None, mean, std)      – synthetic box from mean ± std
ENTRIES = [
    ("MaxEnt IRL",  os.path.join(ROOT_DIR, "outputs/labyrinth_train/ll_maxent_irl.csv")),
    # ("MaxCausalEnt IRL",  os.path.join(ROOT_DIR, "outputs/labyrinth_train/ll_max_causal_entropy.csv")),
    ("IAVI",  os.path.join(ROOT_DIR, "outputs/labyrinth_train/ll_iavi.csv")),
    # ("HIAVI",  os.path.join(ROOT_DIR, "outputs/labyrinth_train/ll_hiavi.csv")),
    ("SWIRL(S-2)",     None, -0.7287, 0.00367),
    ("PGIAVI", os.path.join(ROOT_DIR, "src_autotest/outputs/20260315_192657/G03/E02/ll.csv")),
]
# ─────────────────────────────────────────────────────────────────────────────

BRIGHT_COLORS = [
    '#e74c3c',  # bright red
    '#3498db',  # bright blue
    '#2ecc71',  # bright green
    '#e67e22',  # bright orange
    '#9b59b6',  # bright purple
    '#1abc9c',  # bright teal
    '#f1c40f',  # bright yellow
    '#e91e63',  # bright pink
    '#00bcd4',  # bright cyan
    '#ff5722',  # deep orange
]

OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs/labyrinth_train')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'll_boxplot.pdf')


def load_data(entry):
    """Return a 1-D array of values for one entry."""
    if len(entry) == 2:
        _, path = entry
        df = pd.read_csv(path)
        return np.array(df['test_ll'].iloc[:5])
    else:
        # synthetic: (name, None, mean, std) → 5 pseudo-values that reproduce mean & std
        _, _, mean, std = entry
        # build 5 values with exact mean and std
        vals = np.array([mean - std, mean - std/2, mean, mean + std/2, mean + std])
        # rescale to hit exactly the requested mean and std
        vals = (vals - vals.mean()) / vals.std(ddof=1) * std + mean
        return vals


def plot_boxplot(entries):
    data   = [load_data(e) for e in entries]
    labels = [e[0] for e in entries]

    fig, ax = plt.subplots()

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        vert=True,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops={'linewidth': 1.2, 'color': 'black', 'linestyle': '-'},
        medianprops={'linewidth': 0.0},
    )

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(BRIGHT_COLORS[i % len(BRIGHT_COLORS)])
        box.set_alpha(0.6)
        box.set_edgecolor('black')

    ax.set_ylabel('Test LL', fontsize=18)
    ax.grid(axis='y', alpha=0.75)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=200)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    if not ENTRIES:
        print("No entries configured. Edit the ENTRIES list at the top of this file.")
    else:
        plot_boxplot(ENTRIES)
