import numpy as np
import time
import torch
import torch.nn.functional as F

from scipy.special import logsumexp
from model.intention import IntentionRNN, IntentionLSTM, IntentionTransformer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


RETROSPECTIVE_SCORE_MODES = ('legacy', 'action_normalized')
PARITY_ATOL = 1e-6



class IAVI:
    def __init__(self, num_states, num_actions, P, expert_policy, discount, threshold=1e-3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = P
        self.expert_policy = expert_policy
        self.discount = discount
        self.threshold = threshold
        self.epsilon = 1e-6

        self.r = np.random.randn(self.num_states, self.num_actions)
        self.q = np.random.randn(self.num_states, self.num_actions)

    def train(self):
        X = np.ones((self.num_actions, self.num_actions))
        X *= -1 / (self.num_actions - 1)
        for i in range(self.num_actions):
            X[i, i] = 1

        e = 0
        while True:
            e += 1
            delta = 0
            for s in range(self.num_states):
                tp = self.P[s, :, :]
                # eta = np.log(self.expert_policy[s, :] + self.epsilon) - self.discount * np.matmul(
                #     tp.T, logsumexp(self.q, axis=1).reshape(-1, 1)).reshape(-1)
                eta = np.log(self.expert_policy[s, :] + self.epsilon) - self.discount * np.matmul(
                    tp.T, np.max(self.q, axis=1).reshape(-1, 1)).reshape(-1)

                Y = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    eta_a = eta[a]
                    action_b = [b for b in range(self.num_actions) if b != a]
                    eta_b = eta[action_b]
                    Y[a] = eta_a - 1 / (self.num_actions - 1) * np.sum(eta_b)

                r = np.linalg.lstsq(X, Y, rcond=None)[0]

                delta = max(delta, np.max(np.abs(self.r[s, :] - r)))

                self.r[s, :] = r
                # self.q[s, :] = r + self.discount * np.matmul(tp.T, logsumexp(self.q, axis=1).reshape(-1, 1)).reshape(-1)
                self.q[s, :] = r + self.discount * np.matmul(tp.T, np.max(self.q, axis=1).reshape(-1, 1)).reshape(-1)
            if delta < self.threshold:
                break


class HIAVI:
    def __init__(self, num_latents, num_states, num_actions, P, train_trajs, test_trajs, discount, kl_weight=0.0):
        self.num_latents = num_latents
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = P
        self.discount = discount
        self.train_trajs = train_trajs
        self.test_trajs = test_trajs
        self.kl_weight = kl_weight

    def _get_mc_probs(self, pis, trajs, logp_init, logp_tr):
        num_latents = logp_init.shape[0]
        # KL(pi_l(·|s) || uniform) = log(K) - H(pi_l(·|s)) for each latent and state
        kl_per_state = []
        for l_idx in range(num_latents):
            pi_l = pis[l_idx]  # (num_states, num_actions)
            kl_l = np.sum(pi_l * np.log(pi_l + 1e-10), axis=1) + np.log(self.num_actions)  # (num_states,)
            kl_per_state.append(kl_l)
        logp_gammas = []
        logp_xis = []
        lls = []
        for traj in trajs:
            logp_obs = []
            for s, a, ns in traj:
                logp_obs.append([np.log(pis[l_idx][s, a]) - self.kl_weight * kl_per_state[l_idx][s]
                                  for l_idx in range(num_latents)])
            logp_obs = np.array(logp_obs)

            logp_alpha_prev = logp_init + logp_obs[0]
            logp_alpha = [logp_alpha_prev]
            for lpo in logp_obs[1:]:
                logp_alpha_prev = logsumexp(logp_alpha_prev + logp_tr.T, axis=-1)
                logp_alpha_prev += lpo
                logp_alpha.append(logp_alpha_prev)

            logp_beta_next = np.log(np.ones((num_latents,)))
            logp_beta = [logp_beta_next]
            for lpo_idx, lpo in enumerate(reversed(logp_obs[1:])):
                logp_beta_next += lpo
                logp_beta_next = logsumexp(logp_beta_next + logp_tr, axis=-1)
                logp_beta.append(logp_beta_next)

            logp_alpha = np.array(logp_alpha)
            logp_beta = np.array(logp_beta[::-1])

            logp_gamma = logp_alpha + logp_beta
            logp_gamma -= logsumexp(logp_gamma, axis=-1, keepdims=True)

            logp_xi = []
            for lpa_idx, lpa in enumerate(logp_alpha[:-1]):
                lpx = lpa[:, np.newaxis] + logp_tr + logp_beta[lpa_idx + 1] + logp_obs[lpa_idx + 1]
                lpx -= logsumexp(lpx)
                logp_xi.append(lpx)
            logp_xi = np.array(logp_xi)

            ll = logsumexp(logp_gamma + logp_obs, axis=-1)

            lls.append(ll)
            logp_gammas.append(logp_gamma)
            logp_xis.append(logp_xi)

        return logp_gammas, logp_xis, lls

    def fit(self):
        p_init = np.abs(np.random.randn(self.num_latents))
        p_init /= np.sum(p_init)
        p_tr = 0.95 * np.identity(self.num_latents)
        p_tr += np.abs(np.random.normal(0, 0.05, (self.num_latents, self.num_latents)))
        p_tr /= np.sum(p_tr, axis=-1, keepdims=True)
        logp_init = np.log(p_init)
        logp_tr = np.log(p_tr)

        pis = []
        for l_idx in range(self.num_latents):
            pi = np.abs(np.random.randn(self.num_states, self.num_actions))
            pi /= np.sum(pi, axis=-1, keepdims=True)
            pis.append(pi)
        logp_gammas, *_ = self._get_mc_probs(pis, self.train_trajs, logp_init, logp_tr)

        while True:
            z_hat = np.argmax(np.vstack(logp_gammas), axis=-1)
            pis = []
            agents = []
            for latent_idx in range(self.num_latents):
                inputs = []
                for session_idx, session_trajs in enumerate(self.train_trajs):
                    logp_gamma = logp_gammas[session_idx]
                    for traj_idx, traj in enumerate(session_trajs):
                        if np.random.uniform() > np.exp(logp_gamma[traj_idx, latent_idx]):
                            continue
                        inputs.append(traj)

                expert_pi = np.zeros((self.num_states, self.num_actions))
                for s, a, ns in inputs:
                    expert_pi[s, a] += 1
                expert_pi[expert_pi.sum(axis=1) == 0] = 1e-6
                expert_pi /= expert_pi.sum(axis=1).reshape(-1, 1)
                agent = IAVI(num_states=self.num_states, num_actions=self.num_actions,
                             P=self.P, expert_policy=expert_pi, discount=self.discount)
                agent.train()

                agents.append(agent)
                q = agent.q
                pis.append(np.exp(q) / np.sum(np.exp(q), axis=-1, keepdims=True))

            logp_gammas, logp_xis, _ = self._get_mc_probs(pis, self.train_trajs, logp_init, logp_tr)

            logp_init = logsumexp([logp_gamma[0] for logp_gamma in logp_gammas], b=1 / len(logp_gammas), axis=0)
            logp_tr = logsumexp(np.concatenate(logp_xis), axis=0)
            logp_tr -= logsumexp(np.concatenate([logp_gamma[:-1] for logp_gamma in logp_gammas]), axis=0,
                                 keepdims=True).T
            logp_tr -= logsumexp(logp_tr, axis=-1, keepdims=True)

            if (z_hat == np.argmax(np.vstack(logp_gammas), axis=-1)).all():
                break

        # Evaluation
        ll = {}
        for ds in ['train', 'test']:
            inputs = eval(f'self.{ds}_trajs')
            *_, lls = self._get_mc_probs(pis, inputs, logp_init, logp_tr)
            lls = np.mean(np.hstack(lls))
            ll[ds] = np.mean(lls)

        return ll, logp_init, logp_tr, agents

    def predict(self, pis, trajs, logp_init, logp_tr):
        logp_gammas, *_ = self._get_mc_probs(pis, trajs, logp_init, logp_tr)
        return logp_gammas


class PGIAVI:
    def __init__(self, num_latents, num_states, num_actions, P, train_trajs, test_trajs, discount,
                 gate_mode='retrospective', loss_threshold=1e-3, max_iterations=100,
                 retrospective_score_mode='legacy'):
        if retrospective_score_mode not in RETROSPECTIVE_SCORE_MODES:
            raise ValueError(
                f'Unknown retrospective_score_mode={retrospective_score_mode!r}; '
                f'expected one of {RETROSPECTIVE_SCORE_MODES}.'
            )
        if retrospective_score_mode == 'action_normalized' and gate_mode != 'retrospective':
            raise ValueError(
                'action_normalized retrospective scoring requires gate_mode=retrospective.'
            )
        self.num_latents = num_latents  # K
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_phis = num_states + num_actions     # φ
        self.P = P                      # env trans
        self.discount = discount
        self.train_trajs = train_trajs
        self.test_trajs = test_trajs
        self.gate_mode = gate_mode
        self.loss_threshold = loss_threshold
        self.max_iterations = max_iterations
        self.retrospective_score_mode = retrospective_score_mode

        # self.intention_net = IntentionTransformer(num_states=self.num_states, 
        #                                num_actions=self.num_actions,
        #                                num_latents=self.num_latents, 
        #                                d_model=128, 
        #                                nhead=4,
        #                                num_layers=1,
        #                                dropout=0.2)
        # self.target_intention_net = IntentionTransformer(num_states=self.num_states, 
        #                                num_actions=self.num_actions,
        #                                num_latents=self.num_latents, 
        #                                d_model=128, 
        #                                nhead=4,
        #                                num_layers=1,
        #                                dropout=0.2)
        self.intention_net = IntentionRNN(num_states=self.num_states,
                                       num_actions=self.num_actions,
                                       num_latents=self.num_latents,
                                       hidden_dim=64, 
                                       rnn_hidden_dim=64, 
                                       num_layers=1,
                                       dropout=0.3,
                                       gate_mode=gate_mode)
        self.target_intention_net = IntentionRNN(num_states=self.num_states,
                                       num_actions=self.num_actions,
                                       num_latents=self.num_latents,
                                       hidden_dim=64, 
                                       rnn_hidden_dim=64, 
                                       num_layers=1,
                                       dropout=0.3,
                                       gate_mode=gate_mode)
        self.target_intention_net.load_state_dict(self.intention_net.state_dict())
        self.target_intention_net.eval()
        self.optimizer = torch.optim.Adam(self.intention_net.parameters(), lr=1e-3)

    def intention_mapping(self, states, actions, log_pi):
        f_logits = self.target_intention_net(states.unsqueeze(0), actions.unsqueeze(0)).squeeze(0)              # (T, K)
        log_f = torch.log_softmax(f_logits, dim=-1)
        log_joint = log_f + log_pi.T                # (T, K), log(f_k * π_k) = log P(z_t=k, a_t | s_t, phi_t)
        log_p_gamma = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)  # (T, K)

        return log_p_gamma, log_f, log_joint
    
    def get_log_pi(self, traj, agents):
        log_pi = torch.zeros((self.num_latents, len(traj)))

        for latent_idx, agent in enumerate(agents):
            q = torch.as_tensor(agent.q, dtype=torch.float32)
            if self.gate_mode == 'retrospective':
                pi = torch.softmax(q, dim=-1)
                log_policy = torch.log(pi + 1e-8)
            else:
                log_policy = torch.log_softmax(q, dim=-1)
            for t, (s, a, ns) in enumerate(traj):
                log_pi[latent_idx, t] = log_policy[s, a]

        return log_pi

    def get_all_action_log_pi(self, states, agents):
        agent_log_policies = []
        for agent in agents:
            q = torch.as_tensor(agent.q, dtype=torch.float32)
            log_policy = torch.log(torch.softmax(q, dim=-1) + 1e-8)
            log_policy = log_policy - np.log1p(self.num_actions * 1e-8)
            agent_log_policies.append(log_policy)
        agent_log_policies = torch.stack(agent_log_policies, dim=0)
        return agent_log_policies[:, states, :].permute(1, 2, 0)

    def retrospective_action_scores(
        self, states, actions, agents, reference_log_f, reference_log_gamma
    ):
        if self.retrospective_score_mode != 'action_normalized':
            raise ValueError('Retrospective action normalization was not requested.')
        self.target_intention_net.eval()
        with torch.no_grad():
            candidate_logits = self.target_intention_net.candidate_action_logits(
                states.unsqueeze(0), actions.unsqueeze(0)
            ).squeeze(0)
            candidate_log_f = torch.log_softmax(candidate_logits, dim=-1)
            gather_index = actions[:, None, None].expand(
                -1, 1, self.num_latents
            )
            observed_candidate_log_f = candidate_log_f.gather(
                1, gather_index
            ).squeeze(1)
            candidate_log_pi = self.get_all_action_log_pi(states, agents)
            candidate_log_joint = (
                candidate_log_f.to(torch.float64)
                + candidate_log_pi.to(torch.float64)
            )
            candidate_log_energy = torch.logsumexp(candidate_log_joint, dim=-1)
            log_normalizer = torch.logsumexp(candidate_log_energy, dim=-1)
            candidate_log_probability = (
                candidate_log_energy - log_normalizer.unsqueeze(-1)
            )
            observed_log_probability = candidate_log_probability.gather(
                1, actions.unsqueeze(-1)
            ).squeeze(-1)

            observed_candidate_log_pi = candidate_log_pi.gather(
                1, gather_index
            ).squeeze(1)
            observed_candidate_log_gamma = torch.log_softmax(
                observed_candidate_log_f + observed_candidate_log_pi, dim=-1
            )
            finite_tensors = {
                'candidate logits': candidate_logits,
                'candidate gate': candidate_log_f,
                'candidate policy': candidate_log_pi,
                'candidate energy': candidate_log_energy,
                'action log normalizer': log_normalizer,
                'normalized action log probability': candidate_log_probability,
                'reference gate': reference_log_f,
                'reference responsibility': reference_log_gamma,
            }
            for name, values in finite_tensors.items():
                if not torch.isfinite(values).all():
                    raise RuntimeError(
                        f'Non-finite {name} in retrospective action scorer.'
                    )

            gate_max_error = torch.max(
                torch.abs(
                    torch.exp(observed_candidate_log_f)
                    - torch.exp(reference_log_f)
                )
            ).item()
            responsibility_max_error = torch.max(
                torch.abs(
                    torch.exp(observed_candidate_log_gamma)
                    - torch.exp(reference_log_gamma)
                )
            ).item()
            normalization_max_error = torch.max(
                torch.abs(
                    torch.exp(candidate_log_probability).sum(dim=-1) - 1.0
                )
            ).item()
            if not np.isfinite([
                gate_max_error,
                responsibility_max_error,
                normalization_max_error,
            ]).all():
                raise RuntimeError(
                    'Non-finite validation metric in retrospective action scorer.'
                )
            if gate_max_error >= PARITY_ATOL:
                raise RuntimeError(
                    f'Candidate gate replay error {gate_max_error:.3e} exceeds '
                    f'{PARITY_ATOL:.1e}.'
                )
            if responsibility_max_error >= PARITY_ATOL:
                raise RuntimeError(
                    'Candidate responsibility replay error '
                    f'{responsibility_max_error:.3e} exceeds {PARITY_ATOL:.1e}.'
                )
            if normalization_max_error >= PARITY_ATOL:
                raise RuntimeError(
                    'Retrospective action normalization error '
                    f'{normalization_max_error:.3e} exceeds {PARITY_ATOL:.1e}.'
                )
            if torch.max(observed_log_probability).item() > PARITY_ATOL:
                raise RuntimeError('A normalized action log probability exceeded zero.')

        return {
            'step_log_score': observed_log_probability.cpu().numpy(),
            'log_normalizer': log_normalizer.cpu().numpy(),
            'candidate_log_energy': candidate_log_energy.cpu().numpy(),
            'observed_gate': torch.exp(observed_candidate_log_f).cpu().numpy(),
            'observed_responsibility': (
                torch.exp(observed_candidate_log_gamma).cpu().numpy()
            ),
            'gate_max_error': gate_max_error,
            'responsibility_max_error': responsibility_max_error,
            'normalization_max_error': normalization_max_error,
        }

    def encode_session_traj(self, traj):
        states = torch.tensor([s for s, a, ns in traj], dtype=torch.long)
        actions = torch.tensor([a for s, a, ns in traj], dtype=torch.long)
        return states, actions
    
    def train_batched(self, batch_states, batch_actions, batch_target_gamma, batch_mask, total_length, num_epochs=1):
        """
        :param agents: List of IAVI agents
        :param num_epochs: Number of passes through the data
        """
        total_loss = 0
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            pred_logits = self.intention_net(batch_states, batch_actions, mask=batch_mask, total_length=total_length)  # (B, T, K)
            pred_logf = torch.log_softmax(pred_logits, dim=-1)  # (B, T, K)
            
            # Compute loss: negative log-likelihood
            loss = -(batch_target_gamma * pred_logf * batch_mask.unsqueeze(-1)).sum(dim=-1).mean()
            # ce_loss = -(batch_target_gamma * pred_logf).sum(-1).mean()
            # entropy = -(pred_logf * torch.exp(pred_logf)).sum(-1).mean()
            # loss = ce_loss - 0.02 * entropy
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / num_epochs
    
    def fit(self):
        uniform_policy = np.full((self.num_states, self.num_actions), 1.0 / self.num_actions)
        agents = []
        for _ in range(self.num_latents):
            agent = IAVI(
                num_states=self.num_states,
                num_actions=self.num_actions,
                P=self.P,
                expert_policy=uniform_policy,
                discount=self.discount
            )
            agent.train()
            agents.append(agent)

        logger_cnt = 0
        total_q_time = 0
        total_other_time = 0
        iteration_start_time = time.time()

        while True:
            logger_cnt += 1
            
            # * * * E-step: compute posterior * * *
            log_p_gammas = []
            batch_states = []
            batch_actions = []
            batch_mask = []
            batch_target_gamma = []
            for traj_idx, traj in enumerate(self.train_trajs):
                states, actions = self.encode_session_traj(traj)
                log_pi = self.get_log_pi(traj, agents)
                with torch.no_grad():
                    log_p_gamma, *_ = self.intention_mapping(states, actions, log_pi)
                log_p_gammas.append(log_p_gamma)

                batch_states.append(states)
                batch_actions.append(actions)
                batch_mask.append(torch.ones((states.shape[0],), dtype=torch.bool))
                batch_target_gamma.append(torch.exp(log_p_gamma))
            
            # Pad sequences to same length for RNN input
            max_len = max(s.shape[0] for s in batch_states)
            batch_states_padded = torch.zeros(len(batch_states), max_len, dtype=torch.long)
            batch_actions_padded = torch.zeros(len(batch_actions), max_len, dtype=torch.long)
            batch_mask_padded = torch.zeros(len(batch_mask), max_len, dtype=torch.bool)
            batch_target_gamma_padded = torch.zeros(len(batch_target_gamma), max_len, self.num_latents)
            
            for i, (states, actions, gamma) in enumerate(zip(batch_states, batch_actions, batch_target_gamma)):
                seq_len = states.shape[0]
                batch_states_padded[i, :seq_len] = states
                batch_actions_padded[i, :seq_len] = actions
                batch_mask_padded[i, :seq_len] = 1
                batch_target_gamma_padded[i, :seq_len] = gamma
            
            batch_states = batch_states_padded  # (B, T)
            batch_actions = batch_actions_padded  # (B, T)
            batch_mask = batch_mask_padded  # (B, T)
            batch_target_gamma = batch_target_gamma_padded  # (B, T, K)

            # * * * Update Q-value & policies * * *
            q_start_time = time.time()
            for latent_idx in range(self.num_latents):
                expert_pi = torch.zeros((self.num_states, self.num_actions))
                for traj_idx, traj in enumerate(self.train_trajs):
                    weights = batch_target_gamma[traj_idx][:, latent_idx]
                    for t, (s, a, ns) in enumerate(traj):
                        expert_pi[s, a] += weights[t]
                mask = expert_pi.sum(dim=1) == 0
                expert_pi[mask] = 1e-6
                expert_pi /= expert_pi.sum(dim=1, keepdim=True)

                agent = IAVI(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    P=self.P,
                    expert_policy=expert_pi.numpy(),
                    discount=self.discount
                )
                agent.train()
                agents[latent_idx] = agent
            q_time = time.time() - q_start_time
            total_q_time += q_time

            # * * * Update intention network * * *
            other_start_time = time.time()
            total_loss = self.train_batched(batch_states, batch_actions, batch_target_gamma, batch_mask, max_len, num_epochs=1)
            other_time = time.time() - other_start_time
            total_other_time += other_time

            self.target_intention_net.load_state_dict(self.intention_net.state_dict())

            if logger_cnt % 4 == 0:
                iteration_time = time.time() - iteration_start_time
                print(f'Iteration {logger_cnt}, Loss: {total_loss:.4f}, Q-update: {total_q_time:.2f}s, NN: {total_other_time:.2f}s, Total: {iteration_time:.2f}s')
                total_q_time = 0
                total_other_time = 0

            if abs(total_loss) < self.loss_threshold or logger_cnt >= self.max_iterations:
                final_iteration_time = time.time() - iteration_start_time
                stop_reason = (
                    'loss_threshold'
                    if abs(total_loss) < self.loss_threshold
                    else 'max_iterations'
                )
                print(f'Iteration {logger_cnt}, Stopped ({stop_reason}) with Loss: {total_loss:.4f}, Total time: {final_iteration_time:.2f}s')
                break

        f = {}
        ll = {}
        ll['score_type'] = (
            'retrospective_compatibility_score'
            if self.gate_mode == 'retrospective'
            else 'predictive_action_log_likelihood'
        )
        ll['iterations'] = logger_cnt
        ll['stop_reason'] = stop_reason
        ll['final_loss'] = float(total_loss)
        for ds in ['train', 'test']:
            trajs = eval(f'self.{ds}_trajs')
            fs = []
            responsibilities = []
            step_log_scores = []
            normalized_step_log_scores = []
            retrospective_log_normalizers = []
            candidate_log_energies = []
            candidate_observed_gates = []
            candidate_observed_responsibilities = []
            scorer_gate_errors = []
            scorer_responsibility_errors = []
            scorer_normalization_errors = []
            lls = []
            for traj_idx, traj in enumerate(trajs):
                states, actions = self.encode_session_traj(traj)
                log_pi = self.get_log_pi(traj, agents)
                with torch.no_grad():
                    log_p_gamma, log_f, log_p_joint = self.intention_mapping(states, actions, log_pi)
                    if (
                        ds == 'test'
                        and self.retrospective_score_mode == 'action_normalized'
                    ):
                        normalized = self.retrospective_action_scores(
                            states, actions, agents, log_f, log_p_gamma
                        )
                    log_p_gamma = log_p_gamma.numpy()
                    log_f = log_f.numpy()
                    log_p_joint = log_p_joint.numpy()
                fs.append(np.exp(log_f))
                responsibilities.append(np.exp(log_p_gamma))
                step_score = logsumexp(log_p_joint, axis=-1)
                step_log_scores.append(step_score)
                lls.append(np.mean(step_score))
                if (
                    ds == 'test'
                    and self.retrospective_score_mode == 'action_normalized'
                ):
                    normalized_step_log_scores.append(normalized['step_log_score'])
                    retrospective_log_normalizers.append(normalized['log_normalizer'])
                    candidate_log_energies.append(normalized['candidate_log_energy'])
                    candidate_observed_gates.append(normalized['observed_gate'])
                    candidate_observed_responsibilities.append(
                        normalized['observed_responsibility']
                    )
                    scorer_gate_errors.append(normalized['gate_max_error'])
                    scorer_responsibility_errors.append(
                        normalized['responsibility_max_error']
                    )
                    scorer_normalization_errors.append(
                        normalized['normalization_max_error']
                    )
            ll[ds] = float(np.mean(lls))
            ll[f'{ds}_traj_mean'] = ll[ds]
            ll[f'{ds}_step_mean'] = float(np.mean(np.concatenate(step_log_scores)))
            ll[f'{ds}_steps'] = int(sum(len(scores) for scores in step_log_scores))
            f[ds] = fs
            f[f'gate_{ds}'] = fs
            f[f'responsibility_{ds}'] = responsibilities
            f[f'step_log_score_{ds}'] = step_log_scores
            if (
                ds == 'test'
                and self.retrospective_score_mode == 'action_normalized'
            ):
                normalized_flat = np.concatenate(normalized_step_log_scores)
                normalizer_flat = np.concatenate(retrospective_log_normalizers)
                ll['retrospective_normalized_score_type'] = (
                    'retrospective_action_conditional_log_probability'
                )
                ll['test_retrospective_normalized_traj_mean'] = float(
                    np.mean([np.mean(scores) for scores in normalized_step_log_scores])
                )
                ll['test_retrospective_normalized_step_mean'] = float(
                    np.mean(normalized_flat)
                )
                ll['test_retrospective_normalized_steps'] = int(
                    normalized_flat.size
                )
                ll['test_retrospective_log_normalizer_mean'] = float(
                    np.mean(normalizer_flat)
                )
                ll['test_retrospective_fraction_z_gt_one'] = float(
                    np.mean(normalizer_flat > 0.0)
                )
                ll['legacy_policy_log_row_normalizer'] = float(
                    np.log1p(self.num_actions * 1e-8)
                )
                ll['candidate_gate_max_error'] = float(max(scorer_gate_errors))
                ll['candidate_responsibility_max_error'] = float(
                    max(scorer_responsibility_errors)
                )
                ll['candidate_normalization_max_error'] = float(
                    max(scorer_normalization_errors)
                )
                f['step_log_score_retrospective_normalized_test'] = (
                    normalized_step_log_scores
                )
                f['retrospective_log_normalizer_test'] = (
                    retrospective_log_normalizers
                )
                f['candidate_log_energy_test'] = candidate_log_energies
                f['candidate_observed_gate_test'] = candidate_observed_gates
                f['candidate_observed_responsibility_test'] = (
                    candidate_observed_responsibilities
                )

        return ll, f, agents

    def predict(self, trajs, agents):
        fs = []
        lls = []
        for traj_idx, traj in enumerate(trajs):
            states, actions = self.encode_session_traj(traj)
            log_pi = self.get_log_pi(traj, agents)
            with torch.no_grad():
                _, log_f, log_p_joint = self.intention_mapping(states, actions, log_pi)
                log_f = log_f.numpy()
                log_p_joint = log_p_joint.numpy()
            fs.append(np.exp(log_f))
            lls.append(logsumexp(log_p_joint, axis=-1).sum()) # whole trajectory LL
        lls = np.mean(np.hstack(lls))
        
        return lls, fs
