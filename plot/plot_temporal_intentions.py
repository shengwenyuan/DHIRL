"""
Generates Figure 2(c): temporal intention posterior and frustration counter.

Loads the exact network posterior P(k | history_t) saved during training,
then applies the Hungarian algorithm to align network latents to ground-truth
intention labels (0 = goal, 1 = abandon).
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import KFold
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.insert(0, ROOT)

from env.gridworld import GridWorld

OUTPUTS_TRAIN = os.path.join(ROOT, 'outputs', 'train')
OUT_FIG = os.path.join(ROOT, 'outputs', 'train', 'gridworld_temporal_intentions.pdf')
NUM_TRAJS = 1024
TRAJ_IDX = 308        # trajectory with clear frustration build-up → switch at step 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_frustration(traj, latents, barrier_ints):
    counter = np.zeros(len(traj), dtype=int)
    frust = 0
    for t, ((s, a, ns), lat) in enumerate(zip(traj, latents)):
        switched = t > 0 and lat != latents[t - 1]
        if s in barrier_ints:
            frust += 1
            if switched:
                frust = 0
        elif switched:
            frust = 0
        counter[t] = frust
    return counter


def hungarian_align(alpha, latents):
    """Permute alpha columns so col 0 = goal (gt=0), col 1 = abandon (gt=1)."""
    pred = np.argmax(alpha, axis=1)
    gt = np.array(latents)
    cost = np.zeros((2, 2), dtype=int)
    for k in range(2):
        for j in range(2):
            cost[k, j] = np.sum((pred == k) & (gt == j))
    _, col_ind = linear_sum_assignment(-cost)
    return alpha[:, np.argsort(col_ind)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    env = GridWorld()
    barrier_ints = set(env.state_to_int(b) for b in env.barriers)

    data_dir = os.path.join(ROOT, 'data', 'gridworld')
    with open(os.path.join(data_dir, 'trajs_frustration.json')) as f:
        traj = json.load(f)[TRAJ_IDX]
    with open(os.path.join(data_dir, 'latents_frustration.json')) as f:
        latents = json.load(f)[TRAJ_IDX]

    T = len(traj)
    steps = np.arange(T)

    # Load exact network posterior for TRAJ_IDX from its held-out fold
    splits = list(KFold(n_splits=5, shuffle=True, random_state=10015).split(range(NUM_TRAJS)))
    alpha = None
    for fi, (_, test_idxes) in enumerate(splits):
        if TRAJ_IDX not in test_idxes:
            continue
        f_path = os.path.join(OUTPUTS_TRAIN, 'pgiql', str(NUM_TRAJS), f'fold_{fi}', 'f_test.npy')
        local_idx = int(np.where(test_idxes == TRAJ_IDX)[0][0])
        alpha = np.load(f_path, allow_pickle=True)[local_idx]
        print(f'Loaded network posterior: traj {TRAJ_IDX}, fold {fi}, local {local_idx}')
        break
    assert alpha is not None and alpha.shape == (T, 2), \
        f'Posterior not found or wrong shape: {None if alpha is None else alpha.shape}'

    # Align network latents to ground-truth labels via Hungarian algorithm
    alpha = hungarian_align(alpha, latents)

    frustration = compute_frustration(traj, latents, barrier_ints)
    switch_pts = [t for t in range(1, T) if latents[t] != latents[t - 1]]
    barrier_visits = [t for t, (s, a, _) in enumerate(traj) if s in barrier_ints]

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(3.5, 3.2), dpi=200, constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.5], hspace=0.06, figure=fig)
    ax_post = fig.add_subplot(gs[0])
    ax_frust = fig.add_subplot(gs[1], sharex=ax_post)

    # Background shading per intention phase
    seg_color = {0: '#dbeafe', 1: '#fee2e2'}   # goal=blue, abandon=red
    seg_start = 0
    for t in range(T):
        end_seg = (t == T - 1) or (latents[t] != latents[t + 1])
        if end_seg:
            for ax in (ax_post, ax_frust):
                ax.axvspan(seg_start - 0.5, t + 0.5,
                           color=seg_color[latents[t]], alpha=0.35, lw=0)
            seg_start = t + 1

    # Vertical line at switch point(s)
    for sp in switch_pts:
        for ax in (ax_post, ax_frust):
            ax.axvline(sp - 0.5, color='#374151', lw=1.4,
                       ls=(0, (4, 3)), zorder=4)

    # Posterior curves
    ax_post.plot(steps, alpha[:, 0], color='#1d4ed8', lw=2.0,
                 label="$P(k{=}'goal'\\mid\\Phi_t)$", zorder=3)
    ax_post.plot(steps, alpha[:, 1], color='#dc2626', lw=2.0, ls='--',
                 label="$P(k{=}'abandon'\\mid\\Phi_t)$", zorder=3)
    ax_post.set_ylim(-0.06, 1.06)
    ax_post.set_yticks([0, 0.5, 1.0])
    ax_post.set_yticklabels(['0', '0.5', '1'], fontsize=9)
    ax_post.set_ylabel('intention\nposterior', fontsize=9)
    ax_post.legend(fontsize=7.5, loc='center right', framealpha=0.85,
                   handlelength=1.6)
    ax_post.tick_params(labelbottom=False, bottom=False)
    ax_post.spines[['top', 'right']].set_visible(False)

    # Frustration counter (step function; tick marks at barrier visits)
    ax_frust.step(steps, frustration, where='mid',
                  color='#7c3aed', lw=2.0, zorder=3)
    for bv in barrier_visits:
        ax_frust.axvline(bv, color='#7c3aed', lw=0.8, ls=':', alpha=0.5, zorder=2)
    max_frust = int(frustration.max())
    ax_frust.set_ylim(-0.3, max_frust + 0.6)
    ax_frust.set_yticks(range(max_frust + 1))
    ax_frust.set_yticklabels([str(i) for i in range(max_frust + 1)], fontsize=9)
    ax_frust.set_ylabel('frustration\ncounter', fontsize=9)
    ax_frust.set_xlabel('time step', fontsize=9)
    ax_frust.set_xlim(-0.5, T - 0.5)
    ax_frust.set_xticks(range(0, T, 2))
    ax_frust.tick_params(labelsize=9)
    ax_frust.spines[['top', 'right']].set_visible(False)

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved {OUT_FIG}')


if __name__ == '__main__':
    run()
