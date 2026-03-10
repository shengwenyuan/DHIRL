"""
Animate one trajectory example in 3D PCA space as a video.

- Loads model and data, gets RNN hidden sequences, fits PCA to 3D.
- Picks one trajectory and animates the point moving along the path in 3D.
- Saves as GIF (pillow, no extra deps) or MP4 (requires ffmpeg).
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from src_visualRNN.algorithms_b import PGIAVI_B


def collect_hidden_sequences(model, trajs, device):
    """Return hidden state sequences (list of (L, D) arrays) and full matrix for PCA."""
    batch_states, batch_actions, mask = model.encode_batch_trajs(trajs)
    max_len = batch_states.shape[1]
    seq_lengths = mask.sum(dim=1).cpu().numpy().astype(int)

    with torch.no_grad():
        _, h_sequence = model.target_intention_net.forward_with_hidden(
            batch_states.to(device), batch_actions.to(device), mask=mask.to(device), total_length=max_len
        )
    h_sequence = h_sequence.cpu().numpy()

    trajectories = []
    all_h = []
    for i in range(h_sequence.shape[0]):
        L = seq_lengths[i]
        h_i = h_sequence[i, :L, :]
        trajectories.append(h_i)
        all_h.append(h_i)
    H_all = np.concatenate(all_h, axis=0)
    return H_all, trajectories


def main():
    parser = argparse.ArgumentParser(description='Animate one trajectory in 3D PCA space')
    parser.add_argument('--num_latents', type=int, default=3)
    parser.add_argument('--num_trajs', type=int, default=238)
    parser.add_argument('--traj_index', type=int, default=0, help='Which trajectory to animate (index in train set)')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='outputs/labyrinth_testRNN/rnn_hidden_viz')
    parser.add_argument('--out_name', type=str, default='trajectory_3d_anim')
    parser.add_argument('--format', type=str, default='gif', choices=['mp4', 'gif'])
    parser.add_argument('--data_dir', type=str, default='data/labyrinth/data')
    parser.add_argument('--elev', type=float, default=20.0, help='3D view elevation (deg)')
    parser.add_argument('--azim_start', type=float, default=45.0, help='Initial azimuth (deg)')
    parser.add_argument('--azim_rotate', type=float, default=360.0, help='Total azimuth rotation over video (deg)')
    args = parser.parse_args()

    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rand_seed)

    num_states = 127
    num_actions = 4
    kf = KFold(n_splits=5, shuffle=True, random_state=10042)

    with open(os.path.join(args.data_dir, 'trans_probs.npy'), 'rb') as f:
        P = np.load(f)
    with open(os.path.join(args.data_dir, 'trajs.js')) as f:
        trajs = json.load(f)
    trajs = trajs[: args.num_trajs]
    train_idx, test_idx = next(kf.split(trajs))
    train_trajs = [trajs[i] for i in train_idx]
    test_trajs = [trajs[i] for i in test_idx]

    print('Training PGIAVI_B...')
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

    print('Collecting hidden sequences...')
    H_all, trajectories = collect_hidden_sequences(model, train_trajs, device)
    pca = PCA(n_components=3)
    pca.fit(H_all)

    idx = min(args.traj_index, len(trajectories) - 1)
    path_3d = pca.transform(trajectories[idx])
    T = len(path_3d)
    if T < 2:
        print('Trajectory too short for animation.')
        return

    os.makedirs(args.out_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Faint full path so it doesn't obstruct the animation
    ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
            'b-', alpha=0.12, linewidth=1.0)
    ax.scatter([path_3d[0, 0]], [path_3d[0, 1]], [path_3d[0, 2]],
               c=[cm.viridis(0.0)], s=80, marker='o', zorder=5)
    point_artist, = ax.plot([path_3d[0, 0]], [path_3d[0, 1]], [path_3d[0, 2]],
                            'o', color=cm.viridis(0.0), markersize=10, zorder=10)
    # Trail as line collection: segments get fading alpha and time-based color.
    # Initialize with one segment so add_collection3d gets valid bounds (empty list breaks auto_scale_xyz).
    dummy_seg = path_3d[0:2] if len(path_3d) >= 2 else np.array([path_3d[0], path_3d[0]])
    trail_collection = Line3DCollection([dummy_seg], linewidths=2.5, zorder=4,
                                        colors=[(0, 0, 0, 0)])
    ax.add_collection3d(trail_collection)
    time_cmap = cm.viridis

    margin = 0.1 * (path_3d.max() - path_3d.min()) or 0.5
    ax.set_xlim(path_3d[:, 0].min() - margin, path_3d[:, 0].max() + margin)
    ax.set_ylim(path_3d[:, 1].min() - margin, path_3d[:, 1].max() + margin)
    ax.set_zlim(path_3d[:, 2].min() - margin, path_3d[:, 2].max() + margin)

    def init():
        point_artist.set_data([], [])
        point_artist.set_3d_properties([])
        trail_collection.set_segments([])
        trail_collection.set_facecolors([])
        trail_collection.set_edgecolors([])
        return point_artist, trail_collection

    def update(frame):
        t = frame
        # Current point color by time
        t_norm = t / max(T - 1, 1)
        point_artist.set_data([path_3d[t, 0]], [path_3d[t, 1]])
        point_artist.set_3d_properties([path_3d[t, 2]])
        point_artist.set_color(time_cmap(t_norm))

        # Trail: segments from 0..t with fading alpha (old = transparent) and time color
        if t >= 1:
            segments = [path_3d[j : j + 2] for j in range(t)]
            alphas = np.clip(0.08 + 0.92 * (np.arange(t, dtype=float) + 1) / t, 0, 1)
            colors = np.array([time_cmap(j / max(t - 1, 1)) for j in range(t)], dtype=float)
            colors[:, 3] = alphas
            colors = np.clip(colors, 0, 1)
            trail_collection.set_segments(segments)
            trail_collection.set_edgecolors(colors)
        else:
            trail_collection.set_segments([])
            trail_collection.set_edgecolors([])

        ax.view_init(elev=args.elev, azim=args.azim_start + (args.azim_rotate * t / max(T - 1, 1)))
        return point_artist, trail_collection

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=1000 // args.fps, blit=False
    )

    out_path = os.path.join(args.out_dir, f'{args.out_name}.{args.format}')
    if args.format == 'mp4':
        try:
            writer = animation.FFMpegWriter(fps=args.fps, metadata=dict(artist='DHIRL'))
            anim.save(out_path, writer=writer)
        except FileNotFoundError:
            out_path = os.path.join(args.out_dir, f'{args.out_name}.gif')
            writer = animation.PillowWriter(fps=args.fps)
            anim.save(out_path, writer=writer)
            print('ffmpeg not found; saved as GIF instead.')
    else:
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(out_path, writer=writer)

    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
