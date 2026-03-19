"""
Hungarian-based latent-index alignment for the labyrinth env.

Canonical semantic ordering  (used by all plot scripts):
    0: explore   1: water (target s=116)   2: home (target s=0)

Note: plot uses yflip=True; "bottom" in the plot corresponds to high y in maze coords.
State positions resolved via m_ru[j][-1], not direct index into m_xc/m_yc.

Usage
-----
    from align_latents import align_latents

    inv_perm = align_latents(ckpt_folder)        # inv_perm[raw_z] → semantic_z
    perm     = np.argsort(inv_perm)              # perm[semantic_z] → raw_z
"""

import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

ROLES = ['explore', 'water', 'home']   # canonical index → name

_ENV_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'labyrinth', 'data')

def _build_regions(num_states: int = 127):
    """Return (home_states, water_states) derived from maze geometry.

    Regions defined by bounding boxes of corner states (resolved through m_ru):
      home  : x∈[4,10], y∈[4,10]  — 7×7 center  (corners s=75,88,101,114; contains s=0)
      water : x∈[8,14], y∈[8,14]  — bottom-right (corners s=111,116,121,126; contains s=116)
    """
    maze_info = np.load(os.path.join(_ENV_FOLDER, 'maze_info.npz'), allow_pickle=True)
    m_xc, m_yc, m_ru = maze_info['m_xc'], maze_info['m_yc'], maze_info['m_ru']
    home, water = [], []
    for j in range(num_states):
        x = int(m_xc[m_ru[j][-1]])
        y = int(m_yc[m_ru[j][-1]])
        if 4 <= x <= 10 and 4 <= y <= 10:
            home.append(j)
        if 8 <= x <= 14 and 8 <= y <= 14 and not (8 <= x <= 10 and 12 <= y <= 14):
            water.append(j)
    return np.array(home), np.array(water)

HOME_REGION, WATER_REGION = _build_regions()


def _norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def align_latents_by_value(ckpt_folder: str, num_latents: int = 3) -> np.ndarray:
    """Align by mean state-value V(s) = E_{a~π}[r(s,a)] over each region.

    Kept for reference; prefer align_latents() which uses trajectory posteriors.
    """
    V = np.array([
        np.sum(
            np.load(f"{ckpt_folder}/r_{i}.npy") *
            softmax(np.load(f"{ckpt_folder}/q_{i}.npy"), axis=-1),
            axis=-1,
        )
        for i in range(num_latents)
    ])  # (num_latents, num_states)

    home_score    = _norm(V[:, HOME_REGION].mean(axis=1))
    water_score   = _norm(V[:, WATER_REGION].mean(axis=1))
    explore_score = 1.0 - _norm(home_score + water_score)

    scores = np.stack([explore_score, water_score, home_score], axis=1)
    _, col_ind = linear_sum_assignment(-scores)
    return col_ind


def align_latents(ckpt_folder: str, num_latents: int = 3, split: str = 'test') -> np.ndarray:
    """Align by soft visitation frequency: mean posterior f(z|t) over region visits.

    For each latent z, its score for a role is the mean posterior probability f(z)
    at timesteps where the agent was inside that role's target region.
    This uses the same data as the pedigree plot (f_test.npy + test_trajs.npy),
    making alignment consistent with the displayed segmentation.

    Parameters
    ----------
    ckpt_folder : path containing f_{split}.npy and {split}_trajs.npy
    split       : 'test' (default) or 'train'

    Returns
    -------
    inv_perm : np.ndarray, shape (num_latents,)
        inv_perm[raw_z] = canonical semantic index
    """
    f      = np.load(f"{ckpt_folder}/f_{split}.npy")           # (T, L, Z)
    states = np.load(f"{ckpt_folder}/{split}_trajs.npy")[:, :, 0].astype(int)  # (T, L)

    home_mask  = np.isin(states, HOME_REGION)   # (T, L) bool
    water_mask = np.isin(states, WATER_REGION)

    # Penalise posterior OUTSIDE each region: a well-concentrated latent
    # has low mean posterior on the complement, so negate to get a score.
    home_score  = np.array([-f[:, :, z][~home_mask].mean()  for z in range(num_latents)])
    water_score = np.array([-f[:, :, z][~water_mask].mean() for z in range(num_latents)])

    home_score  = _norm(home_score)
    water_score = _norm(water_score)
    explore_score = 1.0 - _norm(home_score + water_score)

    # scores[latent, role], roles = ['explore', 'water', 'home']
    # scores = np.stack([explore_score, water_score, home_score], axis=1)
    scores = np.stack([home_score, water_score, explore_score], axis=1)
    _, col_ind = linear_sum_assignment(-scores)
    return col_ind  # inv_perm[raw_z] = semantic_z
