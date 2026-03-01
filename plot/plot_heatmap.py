import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from env.gridworld import GridWorld
from env.collect_demo import value_iteration, policy_eval

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
OUTPUTS_TRAIN = os.path.join(ROOT, 'outputs', 'train')
OUT_FIG = os.path.join(ROOT, 'outputs', 'train', 'gridworld_heatmaps.pdf')
NUM_TRAJS = 1024
GRID_SIZE = 5
N_FOLDS = 5

MODELS = [
    ('pgiql', 2),
    ('hiavi', 2),
    ('iavi', 1),
    ('max_causal_entropy', 1),
    ('max_entropy', 1),
]
LL_CSV = {
    'pgiql': os.path.join(OUTPUTS_TRAIN, 'll_pgiql.csv'),
    'hiavi': os.path.join(OUTPUTS_TRAIN, 'll_hiavi.csv'),
    'iavi': os.path.join(OUTPUTS_TRAIN, 'll_iavi.csv'),
    'max_causal_entropy': os.path.join(OUTPUTS_TRAIN, 'll_max_causal_entropy.csv'),
    'max_entropy': os.path.join(OUTPUTS_TRAIN, 'll_max_entropy.csv'),
}
# Short labels for heatmap columns
MODEL_LABELS = {'max_causal_entropy': 'MaxCausalEnt', 'max_entropy': 'MaxEnt'}


def ground_truth_v(env):
    P = env.P

    r_goal = np.zeros(env.num_states)
    r_goal[env.state_to_int(env.goal_state)] = 1
    v_goal = value_iteration(
        r_goal, P, env.num_states, env.num_actions, env.gamma
    )

    r_abandon = np.zeros(env.num_states)
    r_abandon[env.state_to_int(env.initial_state)] = 1
    v_abandon = value_iteration(
        r_abandon, P, env.num_states, env.num_actions, env.gamma
    )
    return v_goal, v_abandon, P, r_goal, r_abandon


def q_to_policy(q):
    """ π(a|s) = softmax(Q(s,a))."""
    x = q - q.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)


def v_to_grid(V, grid_size):
    return V.reshape(grid_size, grid_size)


def evd(V, V_gt, s0=None):
    """Expected Value Difference: mean |V - V_gt|; if s0 given also V(s0)-V_gt(s0)."""
    mae = np.mean(np.abs(V - V_gt))
    at_s0 = float(V[s0] - V_gt[s0]) if s0 is not None else None
    return mae, at_s0


def mean_stderr(arr):
    a = np.asarray(arr)
    n = len(a)
    if n <= 1:
        return np.mean(a), 0.0
    return np.mean(a), np.std(a, ddof=1) / np.sqrt(n)


def best_fold_indices(ll_path, num_trajs, num_folds_to_show=1):
    if not ll_path or not os.path.isfile(ll_path):
        return list(range(num_folds_to_show))
    import pandas as pd
    df = pd.read_csv(ll_path)
    df = df[df['num_trajs'] == num_trajs].sort_values('test_ll', ascending=False)
    return [int(f) for f in df['fold'].values[:num_folds_to_show]]


def load_model_values(model_name, num_agents, num_experiments, num_trajs, grid_size,
                      P, r_goal, r_abandon, num_states, num_actions, discount):
    """Load Q from saved .npy, then V = policy_eval(π, r, P) with π=softmax(Q)."""
    base = os.path.join(OUTPUTS_TRAIN, model_name, str(num_trajs))
    if not os.path.isdir(base):
        return [], []

    ll_path = LL_CSV.get(model_name)
    fold_indices = best_fold_indices(ll_path, num_trajs, num_experiments)

    V_goals, V_abandons = [], []
    for fi in fold_indices:
        fold_dir = os.path.join(base, f'fold_{fi}')
        if num_agents == 2:
            q0 = np.load(os.path.join(fold_dir, 'q_0.npy'))
            q1 = np.load(os.path.join(fold_dir, 'q_1.npy'))
            pi_goal = q_to_policy(q0)
            pi_abandon = q_to_policy(q1)
        else:
            q = np.load(os.path.join(fold_dir, 'q.npy'))
            pi = q_to_policy(q)
            pi_goal = pi_abandon = pi
        v_goal = policy_eval(pi_goal, r_goal, P, num_states, discount)
        v_abandon = policy_eval(pi_abandon, r_abandon, P, num_states, discount)
        V_goals.append(v_goal)
        V_abandons.append(v_abandon)
    return V_goals, V_abandons


def run():
    sys.path.insert(0, ROOT)

    env = GridWorld()
    assert env.grid_size == GRID_SIZE
    s0 = env.state_to_int(env.initial_state)

    v_goal_gt, v_abandon_gt, P, r_goal, r_abandon = ground_truth_v(env)

    columns = [(v_goal_gt, v_abandon_gt)]
    column_labels = ['GT']

    model_evd_data = []
    for model_name, num_agents in MODELS:
        V_goals_all, V_abandons_all = load_model_values(
            model_name, num_agents, N_FOLDS, NUM_TRAJS, GRID_SIZE,
            P, r_goal, r_abandon,
            env.num_states, env.num_actions, env.gamma
        )
        if not V_goals_all:
            continue
        model_evd_data.append((model_name, V_goals_all, V_abandons_all))
        columns.append((V_goals_all[0], V_abandons_all[0]))
        column_labels.append(MODEL_LABELS.get(model_name, model_name))

    n_rows = 2
    n_cols = len(columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=150)
    if n_cols == 1:
        axes = axes[:, None]

    for col_idx, (v_goal, v_abandon) in enumerate(columns):
        ax = axes[0, col_idx]
        grid = v_to_grid(v_goal, GRID_SIZE)
        vmin, vmax = float(grid.min()), float(grid.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        # print(f'goal   {column_labels[col_idx]:12s} min={vmin:.4f} max={vmax:.4f}')
        im0 = ax.imshow(grid, aspect='equal', vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel('goal', fontsize=10)
        ax.set_title(column_labels[col_idx], fontsize=9)
        cbar0 = fig.colorbar(im0, ax=ax, shrink=0.6, aspect=20, pad=0.02)
        cbar0.set_ticks([vmin, vmax])
        cbar0.set_ticklabels([f'{vmin:.3f}', f'{vmax:.3f}'])
        if col_idx == 0:
            print(f'goal   col={column_labels[col_idx]:12s} EVD_MAE=0.000000±0.000000  EVD(s0)=0.000000±0.000000')

        ax = axes[1, col_idx]
        grid = v_to_grid(v_abandon, GRID_SIZE)
        vmin, vmax = float(grid.min()), float(grid.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        # print(f'abandon {column_labels[col_idx]:12s} min={vmin:.4f} max={vmax:.4f}')
        im1 = ax.imshow(grid, aspect='equal', vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel('abandon', fontsize=10)
            print(f'abandon col={column_labels[col_idx]:12s} EVD_MAE=0.000000±0.000000  EVD(s0)=0.000000±0.000000')
        cbar1 = fig.colorbar(im1, ax=ax, shrink=0.6, aspect=20, pad=0.02)
        cbar1.set_ticks([vmin, vmax])
        cbar1.set_ticklabels([f'{vmin:.3f}', f'{vmax:.3f}'])

    for model_name, V_goals_all, V_abandons_all in model_evd_data:
        mae_goal = [evd(V, v_goal_gt, s0)[0] for V in V_goals_all]
        s0_goal = [evd(V, v_goal_gt, s0)[1] for V in V_goals_all]
        m_mae_g, se_mae_g = mean_stderr(mae_goal)
        m_s0_g, se_s0_g = mean_stderr(s0_goal)
        print(f'goal   col={model_name:12s} EVD_MAE={m_mae_g:.6f}±{se_mae_g:.6f}  EVD(s0)={m_s0_g:.6f}±{se_s0_g:.6f}')

        mae_abandon = [evd(V, v_abandon_gt, s0)[0] for V in V_abandons_all]
        s0_abandon = [evd(V, v_abandon_gt, s0)[1] for V in V_abandons_all]
        m_mae_a, se_mae_a = mean_stderr(mae_abandon)
        m_s0_a, se_s0_a = mean_stderr(s0_abandon)
        print(f'abandon col={model_name:12s} EVD_MAE={m_mae_a:.6f}±{se_mae_a:.6f}  EVD(s0)={m_s0_a:.6f}±{se_s0_a:.6f}')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT_FIG}')


if __name__ == '__main__':
    run()
