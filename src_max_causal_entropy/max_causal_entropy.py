import numpy as np
from scipy.special import logsumexp


class MaxEnt:
    def __init__(self, num_states, num_actions, P, expert_sa_count, discount,
                 lr=0.1, threshold=1e-4, max_iter=500):
        # P: (S, A, S') = P(s'|s,a)
        self.nS, self.nA = num_states, num_actions
        self.P = np.asarray(P, dtype=np.float64)
        self.expert_sa = np.asarray(expert_sa_count, dtype=np.float64)
        self.expert_sa += 1e-10
        self.expert_sa /= self.expert_sa.sum()
        self.gamma = discount
        self.lr = lr
        self.threshold = threshold
        self.max_iter = max_iter

        self.r = np.zeros((self.nS, self.nA))
        self._mu0 = None  # initial state dist from expert
        self.pi = None
        self.q = None

    def _expert_mu0(self, trajs):
        """Empirical initial state distribution from trajs."""
        mu0 = np.zeros(self.nS)
        for traj in trajs:
            if len(traj):
                s, _, _ = traj[0]
                mu0[s] += 1
        mu0 += 1e-10
        mu0 /= mu0.sum()
        return mu0

    def set_initial_dist(self, mu0):
        self._mu0 = np.asarray(mu0, dtype=np.float64)
        self._mu0 += 1e-10
        self._mu0 /= self._mu0.sum()

    def _backward_soft(self):
        """Soft value iteration: V(s) = log sum_a exp(r(s,a) + γ E_s'[V(s')])."""
        V = np.zeros(self.nS)
        for _ in range(500):
            Q = self.r + self.gamma * (self.P @ V)
            V_new = logsumexp(Q, axis=1)
            if np.max(np.abs(V_new - V)) < self.threshold:
                break
            V = V_new
        Q = self.r + self.gamma * (self.P @ V)
        log_pi = Q - logsumexp(Q, axis=1, keepdims=True)
        pi = np.exp(log_pi)
        return V, Q, pi

    def _forward_occupancy(self, pi):
        """Discounted state occupancy μ(s) under π."""
        if self._mu0 is None:
            self._mu0 = np.ones(self.nS) / self.nS
        mu = self._mu0.copy()
        T = np.einsum('sa,sap->sp', pi, self.P)
        for _ in range(500):
            mu_new = (1 - self.gamma) * self._mu0 + self.gamma * (T.T @ mu)
            if np.max(np.abs(mu_new - mu)) < self.threshold:
                break
            mu = mu_new
        return mu

    def train(self, trajs=None):
        if trajs is not None:
            self.set_initial_dist(self._expert_mu0(trajs))
        elif self._mu0 is None:
            self._mu0 = np.ones(self.nS) / self.nS

        diff = np.inf
        for it in range(self.max_iter):
            V, Q, pi = self._backward_soft()
            mu = self._forward_occupancy(pi)
            D_sa = mu[:, None] * pi
            D_sa += 1e-10
            D_sa /= D_sa.sum()
            grad = self.expert_sa - D_sa
            self.r += self.lr * grad
            diff = np.max(np.abs(grad))
            if diff < self.threshold:
                break
        V, Q, pi = self._backward_soft()
        self.q = Q
        self.pi = pi
        return diff

    def get_policy(self):
        return getattr(self, 'pi', self._backward_soft()[2])

    def get_rewards(self):
        return self.r.copy()

    def get_q_values(self):
        return getattr(self, 'q', self._backward_soft()[1])
