import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from env.gridworld import GridWorld

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))

OUTPUTS_TRAIN = os.path.join(ROOT, 'outputs', 'train')
OUT_FIG = os.path.join(ROOT, 'outputs', 'train', 'gridworld_rewardmaps.pdf')
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

MODEL_LABELS = {
    'max_causal_entropy': 'MaxCausalEnt',
    'max_entropy': 'MaxEnt',
    'pgiql': 'PGIQL',
    'hiavi': 'HIAVI',
    'iavi': 'IAVI'
}


def best_fold_index(ll_path, num_trajs):
    if not ll_path or not os.path.isfile(ll_path):
        return 0
    import pandas as pd
    df = pd.read_csv(ll_path)
    df = df[df['num_trajs'] == num_trajs].sort_values('test_ll', ascending=False)
    if len(df) == 0:
        return 0
    return int(df['fold'].values[0])


def q_to_policy(q):
    # π(a|s) = softmax(Q(s,a))
    x = q - q.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)


def get_optimal_action(q):
    return np.argmax(q, axis=-1)


def arrow_color_for_action(direction):
    color_map = {
        0: 'white',   # stay
        1: '#2563eb', # down  -> blue
        2: '#dc2626', # up    -> red
        3: '#2563eb', # right -> blue
        4: '#dc2626', # left  -> red
    }
    return color_map.get(direction, 'gray')


def draw_arrow(ax, x, y, direction, color=None, alpha=1.0, scale=0.25):
    if color is None:
        color = arrow_color_for_action(direction)
    
    arrow_map = {
        0: (0, 0),      # stay - draw a circle instead
        1: (0, 0.35),   # down: (1, 0) -> move down in visual
        2: (0, -0.35),  # up: (-1, 0) -> move up in visual
        3: (0.35, 0),   # right: (0, 1) -> move right in visual
        4: (-0.35, 0),  # left: (0, -1) -> move left in visual
    }
    
    dx, dy = arrow_map[direction]
    
    if direction == 0:  # stay: circle
        circle = patches.Circle(
            (x, y), radius=0.15 * scale,
            facecolor=color, edgecolor='0.4', linewidth=1.5,
            alpha=alpha, zorder=10
        )
        ax.add_patch(circle)
    else:
        ax.arrow(x - dx/2, y - dy/2, dx, dy,
                 head_width=0.2*scale, head_length=0.15*scale,
                 fc=color, ec=color, alpha=alpha, linewidth=2,
                 length_includes_head=True, zorder=10)


def plot_reward_map_with_policy(ax, rewards, q_values, grid_size, title, cmap='viridis'):
    if len(rewards.shape) == 2:
        reward_grid = rewards.mean(axis=1).reshape(grid_size, grid_size)
    else:
        reward_grid = rewards.reshape(grid_size, grid_size)
    
    # Normalize rewards for this specific model
    vmin, vmax = reward_grid.min(), reward_grid.max()
    if vmin == vmax:
        vmax = vmin + 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(reward_grid, cmap=cmap, norm=norm, aspect='equal')
    optimal_actions = get_optimal_action(q_values)

    # Draw arrows: right/down=blue, left/up=red, stay=white
    for state in range(grid_size * grid_size):
        row = state // grid_size
        col = state % grid_size
        action = optimal_actions[state]
        draw_arrow(ax, col, row, action, alpha=0.95, scale=1.0)
    
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', linewidth=0.5, alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Reward', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    return im


def load_model_rewards_and_q(model_name, num_agents, num_trajs):
    base = os.path.join(OUTPUTS_TRAIN, model_name, str(num_trajs))
    if not os.path.isdir(base):
        return []
    
    ll_path = LL_CSV.get(model_name)
    fold_idx = best_fold_index(ll_path, num_trajs)
    fold_dir = os.path.join(base, f'fold_{fold_idx}')
    
    results = []
    
    if num_agents == 2:
        # Two-agent model: load both agents
        for agent_idx in range(2):
            r_path = os.path.join(fold_dir, f'r_{agent_idx}.npy')
            q_path = os.path.join(fold_dir, f'q_{agent_idx}.npy')
            
            if os.path.exists(r_path) and os.path.exists(q_path):
                r = np.load(r_path)
                q = np.load(q_path)
                agent_label = f'Agent {agent_idx}'
                results.append((r, q, agent_label))
    else:
        # Single-agent model
        r_path = os.path.join(fold_dir, 'r.npy')
        q_path = os.path.join(fold_dir, 'q.npy')
        
        if os.path.exists(r_path) and os.path.exists(q_path):
            r = np.load(r_path)
            q = np.load(q_path)
            results.append((r, q, ''))
    
    return results


def run():
    env = GridWorld()
    assert env.grid_size == GRID_SIZE
    
    model_data = {}
    for model_name, num_agents in MODELS:
        data = load_model_rewards_and_q(model_name, num_agents, NUM_TRAJS)
        if not data:
            print(f"Warning: No data found for {model_name}")
            continue
        
        label = MODEL_LABELS.get(model_name, model_name)
        
        if num_agents == 2:
            if len(data) >= 2:
                model_data[label] = {
                    'goal': (data[0][0], data[0][1]),  # r, q for agent 0
                    'abandon': (data[1][0], data[1][1])  # r, q for agent 1
                }
        else:
            if len(data) >= 1:
                model_data[label] = {
                    'goal': (data[0][0], data[0][1]),
                    'abandon': (data[0][0], data[0][1])
                }
    
    if not model_data:
        print("Error: No model data found!")
        return
    
    # Create figure: 2 rows (goal, abandon) x n_models columns
    n_cols = len(model_data)
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(5 * n_cols, 4.5 * n_rows), 
                            dpi=150)
    
    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, (model_label, intentions) in enumerate(model_data.items()):
        r_goal, q_goal = intentions['goal']
        title_goal = f"{model_label} (goal)"
        plot_reward_map_with_policy(axes[0, col_idx], r_goal, q_goal, 
                                    GRID_SIZE, title_goal)
        
        r_abandon, q_abandon = intentions['abandon']
        title_abandon = f"{model_label} (abandon)"
        plot_reward_map_with_policy(axes[1, col_idx], r_abandon, q_abandon, 
                                    GRID_SIZE, title_abandon)
        
        if col_idx == 0:
            axes[0, col_idx].set_ylabel('Goal', fontsize=12, fontweight='bold')
            axes[1, col_idx].set_ylabel('Abandon', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved reward maps to: {OUT_FIG}")
    print(f"Generated {n_cols} model(s) x 2 intentions = {n_cols * 2} plots")


if __name__ == '__main__':
    run()
