"""
Boxplot comparison of test log-likelihood across all methods
for the gridworld-frustration case (outputs/train).
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def plot_boxplot(files, labels):
    data = [pd.read_csv(os.path.abspath(f))['test_ll'] for f in files]
    data = [d[:5] for d in data if not d.empty]
    bp = plt.boxplot(
        data,
        tick_labels=labels,
        vert=True,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops={
            'linewidth': 1.2,
            'color': 'black',
            'linestyle': '-',
        },
        medianprops={'linewidth': 0.0},
    )

    # Classic, readable color palette
    classic_colors = [
        '#2e86ab',   # steel blue
        '#a23b72',   # raspberry
        '#f18f01',   # amber
        '#c73e1d',   # brick red
        '#3b1f2b',   # dark plum
    ]
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(classic_colors[i % len(classic_colors)])
        box.set_alpha(0.7)
        box.set_edgecolor('black')

    plt.ylabel('Test LL', fontsize=21)
    # plt.title('Gridworld: Frustration', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.gca().tick_params(axis='both', labelsize=16)
    # Bold the first column (pgiavi)
    for i, tick in enumerate(plt.gca().get_xticklabels()):
        tick.set_fontweight('bold' if i == 0 else 'normal')
    plt.tight_layout()
    out_path = os.path.join(data_dir, 'gridworld_frustration_ll_boxplot.pdf')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Saved: {out_path}')


# Display names: file base name (without ll_) -> label on plot (capital, pgiql -> PRISM)
LABEL_RENAME = {
    'max_causal_entropy': 'MaxCE',
    'max_entropy': 'MaxEnt',
    'pgiql': 'PRISM',
    'hiavi': 'HIAVI',
    'iavi': 'IAVI',
}

if __name__ == '__main__':
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../outputs/train'
    )
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    # Put pgiql first, then sort the rest alphabetically
    pgiql_files = [f for f in files if 'pgiql' in os.path.basename(f)]
    other_files = sorted([f for f in files if f not in pgiql_files])
    files = pgiql_files + other_files
    # Labels: strip path and .csv, remove "ll_", apply renames
    raw = [os.path.splitext(os.path.basename(f))[0] for f in files]
    raw = [r.replace('ll_', '') if r.startswith('ll_') else r for r in raw]
    labels = [LABEL_RENAME.get(r, r.upper()) for r in raw]
    plot_boxplot(files, labels)
