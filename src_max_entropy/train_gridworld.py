import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from env.gridworld import GridWorld
from src_max_entropy.max_entropy_irl import MaxEntropyIRL


if __name__ == '__main__':
    np.random.seed(42)
    output_dir = 'outputs/train'
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    envr = GridWorld()
    with open('data/gridworld/trajs_frustration.json') as f:
        trajs = json.load(f)
    P = np.transpose(envr.P, (0, 2, 1))  # (S, A, S')

    kf = KFold(n_splits=5, shuffle=True, random_state=10015)
    for num_trajs in [1024, 512, 256]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            # expert state counts from train (Maximum Entropy IRL matches state visitation)
            expert_s = np.zeros(envr.num_states)
            for idx in train_idxes:
                for s, a, ns in trajs[idx]:
                    expert_s[s] += 1

            agent = MaxEntropyIRL(
                num_states=envr.num_states,
                num_actions=envr.num_actions,
                P=P,
                expert_s_count=expert_s,
                discount=envr.gamma,
            )
            agent.train(trajs=[trajs[i] for i in train_idxes])

            pi_hat = agent.get_policy()
            ll = {'train': [], 'test': []}
            for ds, idxes in [('train', train_idxes), ('test', test_idxes)]:
                for idx in idxes:
                    likes = [pi_hat[s, a] for s, a, ns in trajs[idx]]
                    ll[ds].append(np.mean(np.log(np.array(likes) + 1e-8)))
            ll_train = np.mean(ll['train'])
            ll_test = np.mean(ll['test'])

            if num_trajs == 1024:
                param_dir = os.path.join(output_dir, f'max_entropy/{num_trajs}/fold_{kf_idx}')
                os.makedirs(param_dir, exist_ok=True)
                np.save(os.path.join(param_dir, 'r.npy'), agent.get_rewards())
                np.save(os.path.join(param_dir, 'q.npy'), agent.get_q_values())

            output_df.loc[len(output_df)] = [num_trajs, kf_idx, ll_train, ll_test]
            output_df.to_csv(os.path.join(output_dir, 'll_max_entropy.csv'), index=False)
