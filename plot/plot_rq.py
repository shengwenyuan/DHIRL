"""
This plotting script is modified based on https://github.com/BRAINML-GT/SWIRL
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from plot_labyrinth import plot_trajs, PlotMazeFunction
from mpl_toolkits.axes_grid1 import make_axes_locatable

env_folder = os.path.abspath('data/labyrinth/data')
maze_info = np.load(env_folder + '/maze_info.npz', allow_pickle=True)
m_wa, m_ru, m_xc, m_yc = maze_info['m_wa'], maze_info['m_ru'], maze_info['m_xc'], maze_info['m_yc']
xy_list = np.load(env_folder + '/xy_list500new.npy', allow_pickle=True)

ckpt_folder = 'outputs/labyrinth_train/pgiql/237/fold_4'
num_folds = 5
num_states = 127
num_actions = 4
num_latents = 3
rewards = []
qvalues = []
for i in range(num_latents):
    reward = np.load(ckpt_folder + f'/r_{i}.npy')
    qvalue = np.load(ckpt_folder + f'/q_{i}.npy')
    rewards.append(reward)
    qvalues.append(qvalue)
rewards = np.array(rewards)  # (num_latents, num_states, num_actions)
qvalues = np.array(qvalues)  # (num_latents, num_states, num_actions)
f_mapping = np.load(ckpt_folder + '/f_train.npy')
f_mapping = np.concatenate((f_mapping, np.load(ckpt_folder + '/f_test.npy')), axis=0)  # (num_trajs, seq_len, num_latents)

plot_folder = os.path.abspath('outputs/labyrinth_train')
from scipy.special import softmax

# - - - plot on maze - - -
fig, axes = plt.subplots(1, 3, figsize=(19,6), dpi=400)
title_list = ['Water', 'Home', 'Explore']
color_list = ['blue', 'brown', 'lightblue']
color_options = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    (0.814, 0.661, 0.885, 0.9)
]
for i in range(num_latents):
    policy = softmax(qvalues[i], axis=-1)  # (num_states, num_actions)
    converted_map = np.sum(rewards[i] * policy, -1)
    PlotMazeFunction(converted_map, title_list[i], m_wa, m_ru, m_xc, m_yc, numcol='blue', figsize=6, selected_color=color_options[i], axes=axes[i])

norm = plt.Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1

# for i in range(num_latents):
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1, 1), color_options[i]])
#     sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
#     divider = make_axes_locatable(axes[i])
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     cbar = fig.colorbar(sm, cax=cax, ticks=[0, 1])
#     cbar.ax.tick_params(labelsize=12)

plt.savefig(plot_folder + '/all_reward_maps_labyrinth.pdf')
# plt.savefig(plot_folder + '/all_reward_maps_labyrinth.pdf', bbox_inches='tight')


learnt_zs = np.argmax(f_mapping, axis=-1)  # (num_trajs, seq_len)
fig, axs = plt.subplots(1, 3, figsize=(18,6), dpi=400)
axs, lines_list = plot_trajs(m_wa, learnt_zs, xy_list, axs=axs)
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(lines_list[-1], cax=cax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Start', 'End'])
cbar.ax.tick_params(labelsize=18)

plt.savefig(plot_folder + '/all_trajs_labyrinth.pdf', bbox_inches='tight')