import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from src_visualRNN.algorithms_b import PGIAVI_B
from src_visualRNN.intention_b import StatesRNN


def collect_hidden_sequences(model, trajs, device, batch_size=1024):
    batch_states, batch_actions, mask = model.encode_batch_trajs(trajs)
    max_len = batch_states.shape[1]
    B, T = batch_states.shape

    all_h = []
    seq_lengths = mask.sum(dim=1).cpu().numpy().astype(int)

    with torch.no_grad():
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        mask = mask.to(device)
        _, h_sequence = model.target_intention_net.forward_with_hidden(
            batch_states, batch_actions, mask=mask, total_length=max_len
        )
    # h_sequence: (B, T, rnn_hidden_dim)
    h_sequence = h_sequence.cpu().numpy()
    mask_np = mask.cpu().numpy()

    trajectories = []
    for i in range(B):
        L = seq_lengths[i]
        h_i = h_sequence[i, :L, :]  # (L, rnn_hidden_dim)
        trajectories.append(h_i)
        for t in range(L):
            all_h.append(h_i[t])

    H_all = np.stack(all_h, axis=0)  # (N_total, rnn_hidden_dim)
    return H_all, trajectories, seq_lengths


def find_fixed_points(trajectories, threshold_percentile=5.0):
    deltas = []
    for h_seq in trajectories:
        if len(h_seq) < 2:
            continue
        diffs = np.linalg.norm(h_seq[1:] - h_seq[:-1], axis=1)
        deltas.extend(diffs.tolist())
    if not deltas:
        dim = trajectories[0].shape[1] if trajectories else 128
        return np.zeros((0, dim))
    threshold = np.percentile(deltas, threshold_percentile)
    fixed_point_h = []
    for h_seq in trajectories:
        if len(h_seq) < 2:
            continue
        diffs = np.linalg.norm(h_seq[1:] - h_seq[:-1], axis=1)
        for t in range(1, len(h_seq)):
            if diffs[t - 1] <= threshold:
                fixed_point_h.append(h_seq[t])
    return np.array(fixed_point_h) if fixed_point_h else np.zeros((0, trajectories[0].shape[1]))


def main():
    parser = argparse.ArgumentParser(description='Visualize RNN hidden space (PCA 3D)')
    parser.add_argument('--num_latents', type=int, default=3)
    parser.add_argument('--num_trajs', type=int, default=238)
    parser.add_argument('--max_trajs_plot', type=int, default=40, help='Max trajectories to draw in 3D')
    parser.add_argument('--fixed_point_percentile', type=float, default=5.0,
                        help='Percentile of step sizes below which we call h_t a fixed-point candidate')
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='outputs/labyrinth_testRNN/rnn_hidden_viz')
    parser.add_argument('--data_dir', type=str, default='data/labyrinth/data')
    args = parser.parse_args()

    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rand_seed)

    num_states = 127
    num_actions = 4
    num_folds = 5
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)

    with open(os.path.join(args.data_dir, 'trans_probs.npy'), 'rb') as f:
        P = np.load(f)
    with open(os.path.join(args.data_dir, 'trajs.js')) as f:
        trajs = json.load(f)
    trajs = trajs[: args.num_trajs]
    train_idx, test_idx = next(kf.split(trajs))
    train_trajs = [trajs[i] for i in train_idx]
    test_trajs = [trajs[i] for i in test_idx]

    print('Training PGIAVI_B to get a trained intention RNN...')
    model = PGIAVI_B(
        num_latents=args.num_latents,
        num_states=num_states,
        num_actions=num_actions,
        P=P,
        train_trajs=train_trajs,
        test_trajs=test_trajs,
        discount=0.9,
    )
    model.fit()

    print('Collecting hidden state sequences...')
    H_all, trajectories, seq_lengths = collect_hidden_sequences(model, train_trajs, device)
    print(f'Total hidden states: {H_all.shape[0]}, hidden_dim: {H_all.shape[1]}, num_trajs: {len(trajectories)}')

    # PCA 3D projection
    pca = PCA(n_components=3)
    pca.fit(H_all)
    H_3d = pca.transform(H_all)
    print(f'PCA explained variance ratio (top 3): {pca.explained_variance_ratio_}')

    # Map each trajectory to 3D
    start = 0
    traj_3d = []
    for L in seq_lengths:
        traj_3d.append(pca.transform(H_all[start : start + L]))
        start += L

    # Fixed points in original space, then project to 3D
    fixed_h = find_fixed_points(trajectories, threshold_percentile=args.fixed_point_percentile)
    if len(fixed_h) > 0:
        fixed_3d = pca.transform(fixed_h)
    else:
        fixed_3d = np.zeros((0, 3))

    os.makedirs(args.out_dir, exist_ok=True)

    # Plot 1: trajectory paths in 3D (subset)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    n_plot = min(args.max_trajs_plot, len(traj_3d))
    indices = np.random.choice(len(traj_3d), size=n_plot, replace=False)
    for idx in indices:
        path = traj_3d[idx]
        if len(path) < 2:
            continue
        ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.6, linewidth=1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('RNN hidden paths (top 3 PCs)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'trajectory_paths_3d.pdf'))
    plt.close()
    print(f'Saved trajectory_paths_3d.pdf ({n_plot} trajectories)')

    # Plot 2: fixed points in 3D (overlay on one trajectory path for context)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for idx in indices[: min(15, len(traj_3d))]:
        path = traj_3d[idx]
        if len(path) < 2:
            continue
        ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.3, linewidth=0.8, color='gray')
    if len(fixed_3d) > 0:
        ax.scatter(fixed_3d[:, 0], fixed_3d[:, 1], fixed_3d[:, 2],
                   c='red', s=20, alpha=0.7, label=f'Fixed-point candidates (n={len(fixed_3d)})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Fixed points (h_t ≈ h_{t-1}) in PCA 3D')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'fixed_points_3d.pdf'))
    plt.close()
    print(f'Saved fixed_points_3d.pdf (fixed-point candidates: {len(fixed_3d)})')

    # Plot 3: single trajectory with fixed points highlighted
    if len(traj_3d) > 0 and len(trajectories[0]) >= 2:
        idx0 = 0
        path = traj_3d[idx0]
        h_seq = trajectories[idx0]
        diffs = np.linalg.norm(h_seq[1:] - h_seq[:-1], axis=1)
        thresh = np.percentile(diffs, args.fixed_point_percentile)
        fixed_mask = np.zeros(len(path), dtype=bool)
        fixed_mask[0] = False
        for t in range(1, len(path)):
            fixed_mask[t] = diffs[t - 1] <= thresh

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', alpha=0.7, label='Path')
        if np.any(fixed_mask):
            ax.scatter(path[fixed_mask, 0], path[fixed_mask, 1], path[fixed_mask, 2],
                       c='red', s=50, alpha=0.9, label='Fixed-point steps')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('One trajectory with fixed-point steps highlighted')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'single_trajectory_fixed.pdf'))
        plt.close()
        print('Saved single_trajectory_fixed.pdf')

    print(f'Done. Outputs in {args.out_dir}')


if __name__ == '__main__':
    main()
