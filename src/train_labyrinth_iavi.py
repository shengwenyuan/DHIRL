import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.algorithms import IAVI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_filename', type=str, default='ll_iavi.csv')
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()

    num_folds = 5
    num_states = 127
    num_actions = 4
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
            pi = np.zeros((num_states, num_actions))
            for train_idx in train_idxes:
                for s, a, ns in trajs[train_idx]:
                    pi[s, a] += 1
            pi[pi.sum(axis=1) == 0] = 1e-6
            pi /= pi.sum(axis=1).reshape(-1, 1)

            agent = IAVI(num_states=num_states, num_actions=num_actions,
                         P=P, expert_policy=pi, discount=0.9)
            agent.train()

            pi_hat = np.exp(agent.q) / np.sum(np.exp(agent.q), axis=-1, keepdims=True)
            ll = {'train': [], 'test': []}
            for ds in ['train', 'test']:
                input_idxes = eval(f'{ds}_idxes')
                for idx in input_idxes:
                    like = []
                    for s, a, ns in trajs[idx]:
                        like.append(pi_hat[s, a])
                    like = np.log(like)
                    ll[ds].append(np.mean(like))

            if num_trajs == 237:
                param_dir = os.path.join(output_dir, f'iavi/{num_trajs}/fold_{kf_idx}')
                os.makedirs(param_dir, exist_ok=True)
                np.save(os.path.join(param_dir, 'r.npy'), agent.r)
                np.save(os.path.join(param_dir, 'q.npy'), agent.q)

            output_df.loc[len(output_df)] = [num_trajs, kf_idx, np.mean(ll['train']), np.mean(ll['test'])]
            output_df.to_csv(os.path.join(output_dir, args.ll_filename), index=False)
