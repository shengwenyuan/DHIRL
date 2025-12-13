import matplotlib.pyplot as plt
import numpy as np
import os
np.random.seed(42)

models = ['3hid-lstm', '3hid-lstm', '3hid-lstm', '3hid-lstm']
n_steps = 500
behaviors = ['water', 'home', 'explore', 'water port visit', 'home visit']
colors = ['#008B8B', '#8B4513', '#BC8F8F', '#FFA500', '#DC143C']  # Teal, Brown, Tan, Orange, Crimson
markers = ['s', 's', 's', 'o', 'x']  # Square, Square, Square, Circle, Cross

plot_folder = os.path.abspath('outputs/labyrinth_train')
ckpt_folder = os.path.abspath('outputs/labyrinth_train/pgiql/237/fold_1')
data = []
for _ in models:
    f_mapping = np.load(ckpt_folder + '/f_test.npy')
    learnt_zs = np.argmax(f_mapping, axis=-1)
    seq = learnt_zs[np.random.randint(len(learnt_zs))]  # Take a random trajectory for plotting
    data.append(seq)


# - - - plotting - - -
fig, axes = plt.subplots(len(models), 1, figsize=(12, 4), sharex=True)
# fig.suptitle('', fontsize=14)

for i, (model, ax, model_data) in enumerate(zip(models, axes, data)):
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_yticks([])
    
    j = 0
    while j < n_steps:
        start = j
        while j < n_steps and model_data[j] == model_data[start]:
            j += 1
        ax.axvspan(start, j, color=colors[model_data[start]], alpha=0.7)
    
    # for j in range(n_steps):
    #     if model_data[j] in [3, 4]:  # water port visit or home visit
    #         ax.plot(j + 0.5, 0.5, marker=markers[model_data[j]], 
    #                color='red' if model_data[j] == 4 else 'orange',
    #                markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, n_steps])
    ax.tick_params(axis='x', which='both', length=4, labelsize=10)
    # ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
    # ax.text(-0.02, 0.5, '0', transform=ax.transAxes, ha='right', va='center')
    # ax.text(1.02, 0.5, str(n_steps), transform=ax.transAxes, ha='left', va='center')

legend_elements = [plt.Line2D([0], [0], marker=markers[i], color='w', 
                             markerfacecolor=colors[i], markersize=8, 
                             label=behaviors[i]) 
                    for i in range(len(behaviors))]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=True, 
           facecolor='white', edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(plot_folder + '/trajectory_segment.pdf')
plt.close()