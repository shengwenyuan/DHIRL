import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import json
import numpy as np

from model.fa import R, Q, Qsh, SAVisitationClassifier
from env.gridworld import GridWorld
from sklearn.model_selection import KFold

class DeepInverseQLearner:
    def __init__(self, reward_net, q_net, qsh_net, policy_net, 
                 lr=1e-3, gamma=0.99, tau=0.001, n_actions=None, device='cpu'):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Networks
        self.reward_net = reward_net.to(device)
        self.q_net = q_net.to(device)
        self.qsh_net = qsh_net.to(device)
        self.policy_net = policy_net.to(device)

        # Target networks
        self.target_reward_net = self._clone_network(reward_net).to(device)
        self.target_q_net = self._clone_network(q_net).to(device)
        self.target_qsh_net = self._clone_network(qsh_net).to(device)
        # Store state dicts for cloning
        self.reward_net_state = reward_net.state_dict()
        self.q_net_state = q_net.state_dict()
        self.qsh_net_state = qsh_net.state_dict()

        # Optimizers
        self.opt_r = optim.Adam(self.reward_net.parameters(), lr=lr)
        self.opt_q = optim.Adam(self.q_net.parameters(), lr=lr)
        self.opt_qsh = optim.Adam(self.qsh_net.parameters(), lr=lr)
        self.opt_pi = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()

    def _clone_network(self, net):
        target = type(net).__new__(type(net))
        target.__dict__.update(net.__dict__)
        return target

    def _polyak_update(self, source, target):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def train_on_batch(self, states, actions, next_states, dones):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        batch_size = states.size(0)

        # 1. Update Shifted Q (QSh)
        with torch.no_grad():
            next_q_max = self.target_q_net(next_states).max(dim=1, keepdim=True).values  # [B, 1]
            y_qsh = self.gamma * next_q_max * (1.0 - dones)  # [B, 1]
            y_qsh = torch.clamp(y_qsh, -10, 10)
        pred_qsh = self.qsh_net(states)
        loss_qsh = self.criterion_mse(pred_qsh, y_qsh)
        self.opt_qsh.zero_grad()
        loss_qsh.backward()
        self.opt_qsh.step()

        # 2. Update policy network rho
        logits = self.policy_net(states)  # [B, A]
        loss_pi = self.criterion_ce(logits, actions)
        self.opt_pi.zero_grad()
        loss_pi.backward()
        self.opt_pi.step()

        # 3. Update reward network r
        with torch.no_grad():
            pi_probs = torch.softmax(logits, dim=1)  # [B, A]
            log_pi = torch.log(pi_probs + 1e-7)

            qsh_vals = self.target_qsh_net(states)  # [B, A]
            eta = log_pi - qsh_vals
            sum_eta = eta.sum(dim=1, keepdim=True)

            r_prime_all = self.target_reward_net(states)  # [B, A]
            sum_r_prime = r_prime_all.sum(dim=1, keepdim=True)

            r_ai = r_prime_all.gather(1, actions.unsqueeze(1))      # [B, 1]
            eta_ai = eta.gather(1, actions.unsqueeze(1))            # [B, 1]

            y_r = eta_ai + (1.0 / (self.n_actions - 1)) * (
                (sum_r_prime - r_ai) - (sum_eta - eta_ai)
            )  # [B, 1]
        pred_r = self.reward_net(states).gather(1, actions.unsqueeze(1))  # [B, 1]
        loss_r = self.criterion_mse(pred_r, y_r)
        self.opt_r.zero_grad()
        loss_r.backward()
        self.opt_r.step()

        # 4. Update Q
        with torch.no_grad():
            r_target = self.target_reward_net(states).gather(1, actions.unsqueeze(1)).detach()  # [B, 1]
            next_q_max = self.target_q_net(next_states).max(dim=1, keepdim=True).values   # [B, 1]
            y_q = r_target + self.gamma * next_q_max * (1.0 - dones)
            y_q = torch.clamp(y_q, -10, 10)
        pred_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        loss_q = self.criterion_mse(pred_q, y_q)
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()

        self._polyak_update(self.reward_net, self.target_reward_net)
        self._polyak_update(self.q_net, self.target_q_net)
        self._polyak_update(self.qsh_net, self.target_qsh_net)

        return {
            'loss_r': loss_r.item(),
            'loss_q': loss_q.item(),
            'loss_qsh': loss_qsh.item(),
            'loss_pi': loss_pi.item()
        }

    def train(self, states, actions, next_states, dones, batch_size=64, epochs=100):
        num_samples = states.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i+batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_next_states = next_states[idx]
                batch_dones = dones[idx]
                # if epoch == epochs - 1:
                #     print(f"Final Epoch Batch idx: {idx}")  # Debugging line
                losses = self.train_on_batch(batch_states, batch_actions, batch_next_states, batch_dones)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Losses: {losses}")
            


if __name__ == "__main__":
    torch.manual_seed(42)
    output_dir = f'outputs/train'
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    # Gridworld example
    envr = GridWorld()
    with open('data/trajs.js') as f:
        trajs = json.load(f)
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10015)

    state_dim = envr.num_states  # 25 for one-hot encoding
    action_dim = envr.num_actions  # 5
    latent_dim = 2

    reward_net = R(state_dim, action_dim, hidden_dim=128)
    q_net = Q(state_dim, action_dim, hidden_dim=256)
    qsh_net = Qsh(state_dim, action_dim, hidden_dim=256)
    policy_net = SAVisitationClassifier(state_dim, action_dim, hidden_dims=[128, 128])

    # run
    for num_trajs in np.arange(1024, 1025, 100):
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            # Build replay buffer D [one-hot states, action indices, dones]
            states_list, actions_list, next_states_list, dones_list = [], [], [], []
            for train_idx in train_idxes:
                traj = trajs[train_idx]
                for t_i, (s, a, ns) in enumerate(traj):
                    s_onehot = np.zeros(state_dim)
                    s_onehot[s] = 1
                    ns_onehot = np.zeros(state_dim)
                    ns_onehot[ns] = 1
                    done_flag = 1 if t_i == (len(traj) - 1) else 0

                    states_list.append(s_onehot)
                    actions_list.append(a)
                    next_states_list.append(ns_onehot)
                    dones_list.append(done_flag)
            D_states = np.array(states_list)  # (N, 25)
            D_actions = np.array(actions_list)  # (N,)
            D_next_states = np.array(next_states_list)  # (N, 25)
            D_dones = np.array(dones_list)  # (N,)

            learner = DeepInverseQLearner(
                reward_net, q_net, qsh_net, policy_net,
                lr=1e-3, gamma=envr.gamma, tau=0.005,
                n_actions=action_dim, device='cpu'
            )
            learner.train(D_states, D_actions, D_next_states, D_dones, batch_size=64, epochs=100)
            
            # Compute loglikelihood on train and test trajectories
            ll = {'train': [], 'test': []}
            for ds in ['train', 'test']:
                input_idxes = eval(f'{ds}_idxes')
                for idx in input_idxes:
                    like = []
                    for s, a, ns in trajs[idx]:
                        # Create one-hot encoded state
                        s_onehot = np.zeros(state_dim)
                        s_onehot[s] = 1
                        s_tensor = torch.as_tensor(s_onehot, dtype=torch.float32, device=learner.device).unsqueeze(0)  # [1, 25]
                        with torch.no_grad():
                            q_vals = learner.q_net(s_tensor)  # [1, 5]
                            pi_hat = torch.softmax(q_vals, dim=1)  # [1, 5]
                        like.append(pi_hat[0, a].item())
                    like = np.log(np.array(like))
                    ll[ds].append(np.mean(like))
            
            output_df.loc[len(output_df)] = [num_trajs, kf_idx, np.mean(ll['train']), np.mean(ll['test'])]
            output_df.to_csv(os.path.join(output_dir, 'll_diql.csv'), index=False)
    