import os
import json
import numpy as np

from gridworld import GridWorld, TreasureCollectionWorld


root = os.path.dirname(os.path.abspath(__file__))

def value_iteration(reward, P, num_states, num_actions, discount, threshold=1e-2):
    """
    Calculate the optimal state value function of given enviroment.

    :param reward: reward vector. nparray. (states, )
    :param P: transition probability p(st | s, a). nparray. (states, states, actions).
    :param discount: discount rate gamma. float. Default: 0.99
    :param num_states: number of states. int.
    :param num_actions: number of actions. int.
    :param threshold: stop when difference smaller than threshold. float.
    :return: optimal state value function. nparray. (states)
    """

    v = np.zeros(num_states)

    while True:
        delta = 0

        for s in range(num_states):
            max_v = float("-inf")
            for a in range(num_actions):
                tp = P[s, :, a]
                max_v = max(max_v, np.dot(tp, (reward + discount * v)))

            diff = abs(v[s] - max_v)
            delta = max(delta, diff)

            v[s] = max_v

        if delta < threshold:
            break

    return v


def vi_policy(num_states, num_actions, P, reward, discount, stochastic=True, threshold=1e-2):
    """
    Find the optimal policy.

    num_states: Number of states. int.
    num_actions: Number of actions. int.
    P: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    v = value_iteration(reward, P, num_states, num_actions, discount, threshold)

    policy = np.zeros((num_states, num_actions))
    if stochastic:
        for s in range(num_states):
            for a in range(num_actions):
                p = P[s, :, a]
                policy[s, a] = p.dot(reward + discount*v)
        policy -= policy.max(axis=1).reshape((num_states, 1))  # For numerical stability.
        policy = np.exp(policy)/np.exp(policy).sum(axis=1).reshape((num_states, 1))

    else:
        def _policy(s):
            return max(range(num_actions),
                       key=lambda a: sum(P[s, k, a] *
                                         (reward[k] + discount * v[k])
                                         for k in range(num_states)))
        for s in range(num_states):
            policy[s, _policy(s)] = 1
    return policy


def policy_eval(policy, reward, P, num_states, discount, threshold=1e-2):
    """
    Policy evaluation.

    :param policy: policy to evaluation. nparray. (states, actions).
    :param reward: ground truth reward of the enviroment. nparray. (states, ).
    :param P: transition probability p(st | s, a). nparray. (states, states, actions).
    :param num_states: number of states in the enviroment. int.
    :param discount: discount rate gamma. float.
    :param threshold: stop when difference smaller than threshold. float.
    :return: state value estimation for given policy. nparray. (states, ).
    """
    v = np.zeros(num_states)
    while True:
        delta = 0
        for s in range(num_states):
            pi = policy[s]
            tp = P[s, :, :]
            target = np.dot(pi, np.matmul(tp.T, (reward + discount * v).reshape(-1, 1)))

            delta = max(delta, np.abs(target - v[s]))

            v[s] = target

        if delta < threshold:
            break

    return v


def collect_barriercase():
    num_trajs = 1024
    p_barrier = 0.3
    p_t = 0.5

    envr = GridWorld()

    r_goal = np.zeros(envr.num_states)
    r_goal[envr.state_to_int(envr.goal_state)] = 1
    v_goal = value_iteration(reward=r_goal, P=envr.P, num_actions=envr.num_actions,
                             num_states=envr.num_states, discount=envr.gamma)
    pi_goal = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                        P=envr.P, reward=r_goal, discount=envr.gamma, stochastic=False)

    r_return = np.zeros(envr.num_states)
    r_return[envr.state_to_int(envr.initial_state)] = 1
    v_return = value_iteration(reward=r_return, P=envr.P, num_actions=envr.num_actions,
                               num_states=envr.num_states, discount=envr.gamma)
    pi_return = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                          P=envr.P, reward=r_return, discount=envr.gamma, stochastic=False)

    pis = [pi_goal, pi_return]

    trajs = []
    latents = []
    for repeat in range(num_trajs):
        traj = []
        latent = []
        s = envr.state_to_int(envr.initial_state)
        pi_idx = 0
        t = 0
        while True:
            if envr.int_to_state(s) in envr.barriers:
                if np.random.uniform() < p_barrier:
                    pi_idx = 1 - pi_idx
            elif t == 8:
                if np.random.uniform() < p_t:
                    pi_idx = 1

            pi = pis[pi_idx]
            a = np.random.choice(envr.num_actions, p=pi[s])
            ns, done = envr.step(s, a)

            traj.append([s, a, ns])
            latent.append(pi_idx)
            s = ns
            t += 1
            if done:
                break
        trajs.append(traj)
        latents.append(latent)

    data_dir = os.path.join(root, '../data/gridworld')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'trajs.json'), 'w') as f:
        json.dump(trajs, f)
    with open(os.path.join(data_dir, 'latents.json'), 'w') as f:
        json.dump(latents, f)

def collect_treasurehunt():
    """
    Collect trajectories for treasure hunting scenario.
    Agent switches from search to target policy after finding treasure.
    """
    num_trajs = 1024

    envr = TreasureCollectionWorld()

    # Policy 1: Random exploration with punishment on initial state
    # Stochastic policy that explores uniformly with penalty for staying at start
    r_search = np.ones(envr.num_states) * 0.1
    # for s in range(envr.num_states):
    #     x, y = envr.int_to_state(s)
    #     terrain = envr.terrain_cost[x, y]
    #     r_search[s] -= (terrain - 1) * 0.1
    r_search[envr.state_to_int(envr.initial_state)] = -0.5
    r_search[envr.state_to_int(envr.goal_state)] = -0.5
    pi_search = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                          P=envr.P, reward=r_search, discount=envr.gamma, stochastic=True)

    # Policy 2: Target policy - go to goal while minimizing energy cost (terrain-aware)
    # Reward goal state highly, penalize high-cost terrain to encourage energy-efficient paths
    r_target = np.zeros(envr.num_states)
    r_target[envr.state_to_int(envr.goal_state)] = 1.0
    for s in range(envr.num_states):
        x, y = envr.int_to_state(s)
        terrain = envr.terrain_cost[x, y]
        r_target[s] -= (terrain - 1) * 0.1
    pi_target = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                          P=envr.P, reward=r_target, discount=envr.gamma, stochastic=False)

    
    trajs = []
    latents = []
    observations = []
    for repeat in range(num_trajs):
        pis = [pi_search, pi_target]
        traj = []
        latent = []
        obs_seq = []
        
        # Reset environment for new trajectory
        s = envr.reset()
        pi_idx = 0  # Start with search policy
        t = 0
        
        while True:
            # Record observation before taking action (does not include treasure status)
            obs = envr._get_observation(s)
            obs['state'] = int(s)
            obs_seq.append(obs)
            
            # Switch policy with probability based on energy remaining
            if pi_idx == 0 and envr.has_treasure:
                switch_prob = 1 - envr.energy / envr.initial_energy
                # if np.random.uniform() < switch_prob:
                if 1:
                    pi_idx = 1  # Switch to target policy
            
            pi = pis[pi_idx]
            a = np.random.choice(envr.num_actions, p=pi[s])
            ns = envr.step(s, a)
            
            traj.append([int(s), int(a), int(ns)])
            latent.append(int(pi_idx))
            
            s = ns
            t += 1
            
            # Check termination conditions
            if s == envr.state_to_int(envr.goal_state) and envr.has_treasure:
                break
            elif envr.energy == 0:
                break
            elif t >= 30:  # Max steps
                break
                
        trajs.append(traj)
        latents.append(latent)
        observations.append(obs_seq)
    
    data_dir = os.path.join(root, '../data/gridworld')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'trajs_treasure.json'), 'w') as f:
        json.dump(trajs, f)
    with open(os.path.join(data_dir, 'latents_treasure.json'), 'w') as f:
        json.dump(latents, f)
    with open(os.path.join(data_dir, 'observations_treasure.json'), 'w') as f:
        json.dump(observations, f)

    successful = sum(1 for traj in trajs 
                     if traj[-1][2] == envr.state_to_int(envr.goal_state))
    found_treasure = sum(1 for obs_list in observations 
                         if any(obs['state'] in [envr.state_to_int((x, y)) 
                                for x in range(envr.grid_size) 
                                for y in range(envr.grid_size) 
                                if envr.treasure_locations[x, y] == 1] 
                                for obs in obs_list))
    print(f"Total trajectories: {num_trajs}")
    print(f"Found treasure locations: {found_treasure} ({found_treasure/num_trajs*100:.1f}%)")
    print(f"Reached goal: {successful} ({successful/num_trajs*100:.1f}%)")
    print(f"Average trajectory length: {np.mean([len(t) for t in trajs]):.1f}")

if __name__ == '__main__':
    np.random.seed(10015)
    
    # collect_barriercase()
    collect_treasurehunt()