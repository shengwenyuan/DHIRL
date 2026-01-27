import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_boxplot(files, labels):
    data = [pd.read_csv(os.path.abspath(f))['test_ll'] for f in files]
    data = [d[:5] for d in data if not d.empty]
    # Create boxplot with filled boxes and bold mean line
    bp = plt.boxplot(
        data,
        tick_labels=labels,
        vert=True,
        patch_artist=True,            # enable facecolor fill
        showmeans=True,               # show mean
        meanline=True,                # render mean as a line
        meanprops={
            'linewidth': 1.2,
            'color': 'black',
            'linestyle': '-'  # solid mean line
        },
        # boxprops={'linewidth': 1.0, 'edgecolor': 'black'},
        # whiskerprops={'linewidth': 1.0},     # hide whiskers
        # capprops={'linewidth': 1.0},         # hide caps
        medianprops={'linewidth': 0.0},      # hide median
    )

    # Use a custom dark, muted color palette
    dark_colors = [
        '#1f3b75',  # dark blue
        '#3b2466',  # indigo
        '#5a2a83',  # deep purple
        '#1b5e5a',  # teal
        '#2f5d3a',  # forest green
        '#6b1d3b',  # maroon
        '#5a3a2e',  # brown
        '#34495e',  # slate
        '#2c3e50',  # dark slate
        '#4b3869',  # muted violet
    ]
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(dark_colors[i % len(dark_colors)])
        box.set_alpha(0.6)
        box.set_edgecolor('black')

    plt.ylabel('Test LL', fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    # plt.show()
    plt.savefig(data_dir + '/ll_boxplot.pdf', bbox_inches='tight', dpi=200)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../outputs/labyrinth_train')
files = glob.glob(os.path.join(data_dir, 'll_pgiql_*.csv'))
files.sort()
labels = [os.path.splitext(os.path.basename(f))[0] for f in files]
labels = [label.split('_')[-1] for label in labels]
plot_boxplot(files, labels)