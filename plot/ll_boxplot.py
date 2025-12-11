import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_boxplot(files, labels):
    data = [pd.read_csv(os.path.abspath(f))['test_ll'] for f in files]
    data = [d[:5] for d in data if not d.empty]
    # data = data[1:]
    # labels = labels[1:]
    plt.boxplot(data, labels=labels, vert=True)
    plt.ylabel('Test LL')
    plt.grid(axis='y', alpha=0.3)
    # plt.show()
    plt.savefig(data_dir + '/ll_boxplot.pdf')

data_dir = os.path.abspath('outputs/labyrinth_train')
files = glob.glob(os.path.join(data_dir, 'll_pgiql_*.csv'))
files.sort()
labels = [os.path.splitext(os.path.basename(f))[0] for f in files]
labels = [label.split('_')[-1] for label in labels]
plot_boxplot(files, labels)