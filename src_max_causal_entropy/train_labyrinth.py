"""Labyrinth experiment: Max Causal Entropy IRL. Data/output aligned with src/train_labyrinth."""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src_max_causal_entropy.max_causal_entropy import MaxEnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_filename', type=str, default='ll_max_causal_entropy.csv')
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()

    num_states = 127
    num_actions = 4
    np.random.seed(args.rand_seed)

    output_dir = 'outputs/labyrinth_train'
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    with open('data/labyrinth/data/trans_probs.npy', 'rb') as f:
        P = np.load(f)
    # P = np.transpose(P, (0, 2, 1))  # (S, A, S')

    with open('data/labyrinth/data/trajs.js') as f:
        trajs = json.load(f)

    kf = KFold(n_splits=5, shuffle=True, random_state=10042)
    for num_trajs in [237, 167, 107]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[i] for i in train_idxes]
            test_trajs = [trajs[i] for i in test_idxes]

            expert_sa = np.zeros((num_states, num_actions))
            for traj in train_trajs:
                for s, a, ns in traj:
                    expert_sa[s, a] += 1

            agent = MaxEnt(
                num_states=num_states,
                num_actions=num_actions,
                P=P,
                expert_sa_count=expert_sa,
                discount=0.9,
            )
            agent.train(trajs=train_trajs)

            pi_hat = agent.get_policy()
            ll_train = np.mean([np.mean(np.log(np.array([pi_hat[s, a] for s, a, _ in t]) + 1e-8)) for t in train_trajs])
            ll_test = np.mean([np.mean(np.log(np.array([pi_hat[s, a] for s, a, ns in t]) + 1e-8)) for t in test_trajs])

            if num_trajs == 237:
                param_dir = os.path.join(output_dir, f'max_causal_entropy/{num_trajs}/fold_{kf_idx}')
                os.makedirs(param_dir, exist_ok=True)
                np.save(os.path.join(param_dir, 'r.npy'), agent.get_rewards())
                np.save(os.path.join(param_dir, 'q.npy'), agent.get_q_values())

            output_df.loc[len(output_df)] = [num_trajs, kf_idx, ll_train, ll_test]
            output_df.to_csv(os.path.join(output_dir, args.ll_filename), index=False)
