import matplotlib.pyplot as plt
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Model label, checkpoint folder, trajectory index (None for random)
CONFIGS = [
    ('L1 2.08', 'src_autotest/outputs/20260315_141343/G00/E06/238/fold_0', 0),
    ('L1 2.12', 'src_autotest/outputs/20260315_141343/G00/E07/238/fold_0', 0),
    # ('L1 2.16', 'src_autotest/outputs/20260315_141343/G00/E08/238/fold_0', 0),
    ('L1 2.22', 'src_autotest/outputs/20260315_161732/G01/E01/238/fold_0', 0),
    # ('L1 2.26', 'src_autotest/outputs/20260315_161732/G01/E02/238/fold_0', 0),
    ('L1 2.32', 'src_autotest/outputs/20260315_161732/G01/E03/238/fold_0', 0),
    # ('KL 1.43', 'src_autotest/outputs/20260315_161732/G01/E04/238/fold_0', 0),
    # ('KL 1.48', 'src_autotest/outputs/20260315_161732/G01/E05/238/fold_0', 0),
    # ('KL 1.36', 'src_autotest/outputs/20260315_051654/G01/E05/238/fold_0', 0),
    # ('Max Ent',  'outputs/labyrinth_train/maxent_irl/237/fold_0', 0),
]

plot_folder = os.path.join(ROOT, 'outputs/labyrinth_train')
# ============================================================

n_steps = 500
behaviors = ['explore', 'water', 'home', 'water port visit', 'home visit']
colors = ["#C9927B", "#0567B7", "#805A3D", '#FFA500', '#DC143C']
markers = ['s', 's', 's', 'o', 'x']

np.random.seed(42)

models = []
zs_list = []
traj_list = []
for label, ckpt_rel, traj_idx in CONFIGS:
    ckpt_folder = os.path.join(ROOT, ckpt_rel)
    f_mapping = np.load(ckpt_folder + '/f_test.npy')
    learnt_zs = np.argmax(f_mapping, axis=-1)
    if traj_idx is None:
        traj_idx = np.random.randint(0, len(learnt_zs))
    print(f"[{label}] trajectory index: {traj_idx}")
    zs_list.append(learnt_zs[traj_idx])
    models.append(label)

    test_trajs = np.load(ckpt_folder + '/test_trajs.npy')
    traj_list.append(test_trajs[traj_idx])


# - - - plotting - - -
fig, axes = plt.subplots(len(models), 1, figsize=(12, 4), sharex=True)
# fig.suptitle('', fontsize=14)

for i, (model, ax, model_zs, traj) in enumerate(zip(models, axes, zs_list, traj_list)):
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_yticks([])
    
    j = 0
    while j < n_steps:
        start = j
        while j < n_steps and model_zs[j] == model_zs[start]:
            if traj[j][0] == 116:  # water port visit
                ax.plot(j + 0.5, 0.5, marker=markers[-2], 
                        color=colors[-2],
                        markersize=6, markeredgewidth=0.5)
            elif traj[j][0] == 0:  # home visit
                ax.plot(j + 0.5, 0.5, marker=markers[-1], 
                        color=colors[-1],
                        markersize=6, markeredgewidth=0.8)
            j += 1
        ax.axvspan(start, j, color=colors[model_zs[start]], alpha=0.7)
    
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, n_steps])
    # ax.tick_params(axis='x', which='both', length=4, labelsize=10)
    # ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
    # ax.text(-0.02, 0.5, '0', transform=ax.transAxes, ha='right', va='center')
    # ax.text(1.02, 0.5, str(n_steps), transform=ax.transAxes, ha='left', va='center')

legend_elements = [plt.Line2D([0], [0], marker=markers[i], color='w',
                             markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=8,
                             label=behaviors[i])
                    for i in range(len(behaviors))]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=True, 
           facecolor='white', edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(plot_folder + '/trajectory_segment.pdf')
plt.close()