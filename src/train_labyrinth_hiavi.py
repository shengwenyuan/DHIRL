import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.algorithms import HIAVI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_filename', type=str, default='ll_hiavi.csv')
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--num_latents', type=int, default=3)
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()

    num_folds = 5
    num_repeats = args.num_repeats
    num_states = 127
    num_actions = 4
    num_latents = args.num_latents
    np.random.seed(args.rand_seed)

    output_dir = 'outputs/labyrinth_train'
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    with open('data/labyrinth/data/trans_probs.npy', 'rb') as f:
        P = np.load(f)
    P = np.transpose(P, (0, 2, 1))
    with open('data/labyrinth/data/trajs.js') as f:
        trajs = json.load(f)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)
    for num_trajs in [237]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.inf
            best_ll = None
            for repeat in range(num_repeats):
                model = HIAVI(num_latents=num_latents, num_states=num_states, num_actions=num_actions,
                              train_trajs=train_trajs, test_trajs=test_trajs, P=P, discount=0.9)
                ll, logp_init, logp_tr, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    param_dir = os.path.join(output_dir, f'hiavi/{num_trajs}/fold_{kf_idx}')
                    os.makedirs(param_dir, exist_ok=True)
                    np.save(os.path.join(param_dir, 'logp_init.npy'), logp_init)
                    np.save(os.path.join(param_dir, 'logp_tr.npy'), logp_tr)
                    for agent_idx, agent in enumerate(agents):
                        np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r)
                        np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q)

            output_df.loc[len(output_df)] = [num_trajs, kf_idx, best_ll['train'], best_ll['test']]
            output_df.to_csv(os.path.join(output_dir, args.ll_filename), index=False)
