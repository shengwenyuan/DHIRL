import os
import json

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold

from src.algorithms import PGIAVI


if __name__ == '__main__':
    num_folds = 5
    num_repeats = 1
    num_states = 127
    num_actions = 4
    num_latents = 3

    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reproducibility settings for CUDA (when available)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    output_dir = f'outputs/labyrinth_train'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    with open('data/labyrinth/data/trans_probs.npy', 'rb') as f:
        P = np.load(f)
    P = np.transpose(P, (0, 2, 1))
    with open('data/labyrinth/data/trajs.js') as f:
        trajs = json.load(f)

    len_trajs = len(trajs)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10015)
    # for num_trajs in np.arange(37, len_trajs, 50):
    for num_trajs in [57, 107, 167, 237]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.inf
            best_ll = None
            for repeats in range(num_repeats):
                model = PGIAVI(num_latents=num_latents, num_states=num_states, num_actions=num_actions,
                                train_trajs=train_trajs, test_trajs=test_trajs, P=P, discount=0.9)
                ll, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    if num_trajs == len_trajs - 1:
                        param_dir = os.path.join(output_dir, f'pgiql/{num_trajs}/fold_{kf_idx}')
                        os.makedirs(param_dir, exist_ok=True)
                        for agent_idx, agent in enumerate(agents):
                            np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r)
                            np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q)
            output_df.loc[len(output_df)] = [num_trajs, kf_idx, best_ll['train'], best_ll['test']]
            output_df.to_csv(os.path.join(output_dir, 'll_pgiql.csv'), index=False)
