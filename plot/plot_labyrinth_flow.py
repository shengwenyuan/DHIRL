"""
Plot V(s) heatmap with greedy flow arrows on the labyrinth maze.
Replaces the expected-reward map with value-function + policy direction.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from plot_labyrinth import PlotMazeFunction, plot_flow_arrows, plot_path_line
from align_latents import align_latents

env_folder = os.path.abspath('data/labyrinth/data')
maze_info = np.load(env_folder + '/maze_info.npz', allow_pickle=True)
m_wa, m_ru, m_xc, m_yc = maze_info['m_wa'], maze_info['m_ru'], maze_info['m_xc'], maze_info['m_yc']
trans_probs = np.load(env_folder + '/trans_probs.npy')  # (127, 4, 127)

ckpt_folder = 'src_autotest/outputs/20260315_192657/G03/E02/238/fold_0' # L1 2.22 KL 1.48
# ckpt_folder = 'src_autotest/outputs/20260315_161732/G01/E05/238/fold_0' # KL 1.48
# ckpt_folder = 'src_autotest/outputs/20260315_161732/G01/E01/238/fold_0' # L1 2.22
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

perm = np.argsort(align_latents(ckpt_folder))  # perm[semantic_z] -> raw_z
rewards = rewards[perm]
qvalues = qvalues[perm]

plot_folder = os.path.abspath('outputs/labyrinth_train')
from scipy.special import softmax

# - - - plot V(s) + flow arrows on maze - - -
fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=400)
title_list = ['explore', 'water', 'home']
color_options = [
    (201/255, 146/255, 123/255, 1.0),  # explore: "#C9927B"
    (5/255, 103/255, 183/255, 1.0),    # water:   "#0567B7"
    (128/255, 90/255, 61/255, 1.0),    # home:    "#805A3D"
]
landmark_map = {
    0: {},                # explore: no specific goal
    1: {116: '*'},        # water port
    2: {0: 'o'},          # home
}
home_to_water_path = {0, 2, 6, 13, 28, 57}  # path from home to water
near_home_states = set(range(1, 15))        # states 1-14 near home

highlight_map = {
    'explore': None,
    'water': home_to_water_path,
    'home': near_home_states,
}

for i in range(num_latents):
    r_max = np.max(rewards[i], axis=-1) - np.mean(rewards[i], axis=-1)  # max_a r(s,a) - mean_a r(s,a)
    PlotMazeFunction(r_max, title_list[i], m_wa, m_ru, m_xc, m_yc,
                     numcol=None, figsize=6,
                     selected_color=color_options[i], axes=axes[i],
                     landmarks=landmark_map[i])
    hl = highlight_map[title_list[i]]
    plot_flow_arrows(axes[i], qvalues[i], trans_probs, m_ru, m_xc, m_yc,
                     arrow_color='black', arrow_alpha=0.3, arrow_scale=0.35,
                     highlight_states=hl, highlight_color='gold')
    if hl:
        plot_path_line(axes[i], hl, qvalues[i], trans_probs, m_ru, m_xc, m_yc)

plt.savefig(plot_folder + '/all_vflow_maps_labyrinth.pdf', bbox_inches='tight')
print(f'Saved to {plot_folder}/all_vflow_maps_labyrinth.pdf')
