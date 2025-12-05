"""
Labyrinth Data Preprocessing Script
Processes mouse trajectory data for HMM analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import sys

# Add module path
module_path = 'code'
if module_path not in sys.path:
    sys.path.append(module_path)

# Load utilities from Rosenberg-2021-Repository
from MM_Plot_Utils import plot, hist
from MM_Maze_Utils import *
from MM_Traj_Utils import *

# Load transition probabilities
trans_probs = np.load('../data/trans_probs.npy')

def compute_prev_state_map(trans_probs):
    """Compute valid previous states for each state"""
    n_state = trans_probs.shape[0]
    prev_state_map = {x: [] for x in range(n_state)}
    
    for x in range(n_state):
        for prev_x in range(n_state):
            if np.sum(trans_probs[prev_x, :, x]) > 0:
                prev_state_map[x].append(prev_x)
    
    return prev_state_map

prev_state_map = compute_prev_state_map(trans_probs)

# Initialize data structures
RewNames = ['B1', 'B2', 'B3', 'B4', 'C1', 'C3', 'C6', 'C7', 'C8', 'C9']
ma = NewMaze(6)

no_list = []
act_list = []
emissions_list = []
nose_pos_list = []
names = []
times = []
xy_list = []
k_list = []
b_list = []

traj = []
act = []
emission = []
time = []
nose_pos = []
x = []
y = []
bs = []

action_arr = MakeStepType(ma)
tc = []

# Process trajectory data
for i, nickname in enumerate(RewNames):
    tf = LoadTraj(nickname + '-tf')
    ba = np.array([[b[0]/30, b[1]/30] for b in tf.fr])  # start and end of each bout in seconds
    tc += [np.array(ba[1:, 0] - ba[:-1, 1])]
    
    for j in range(len(tf.no)):
        n = tf.no[j]
        b = tf.fr[j]
        k = tf.ke[j]

        for t in range(n.shape[0] - 1):
            if len(traj) == 500:
                # Check if sufficient visits to reward and start states
                if np.count_nonzero(np.array(traj) == 116) < 7 or np.count_nonzero(np.array(traj) == 0) < 7:
                    traj.pop(0)
                    act.pop(0)
                    emission.pop(0)
                    x.pop(0)
                    y.pop(0)
                    time.pop(0)
                    bs.pop(0)
                else:
                    no_list.append(traj)
                    act_list.append(act)
                    emissions_list.append(emission)
                    k_list.append(k)
                    xy_list.append([x, y])
                    names.append(nickname)
                    times.append(time)
                    b_list.append(bs)
                    traj = []
                    act = []
                    emission = []
                    x = []
                    y = []
                    time = []
                    bs = []
                    
            # Check for invalid transitions
            if len(traj) > 0:
                if traj[-1] not in prev_state_map[n[t, 0]]:
                    traj = []
                    act = []
                    emission = []
                    time = []
                    x = []
                    y = []
                    bs = []
                    print(n[t, 0], traj[-1])
            
            traj.append(n[t, 0])
            act.append(action_arr[n[t, 0], n[t+1, 0]])
            emission.append([n[t, 0], action_arr[n[t, 0], n[t+1, 0]]])
            time.append(n[t, 1])
            x.append(k[n[t, 1], 0])
            y.append(k[n[t, 1], 1])
            bs.append(b)

# Compute trial times
def compute_trial_time(trial):
    """Compute total time accounting for wrap-arounds"""
    total_time = 0
    first_time = trial[0]
    for i in range(1, len(trial)):
        if trial[i] < trial[i - 1]:  # Wrap-around detected
            total_time += trial[i - 1] - first_time
            first_time = trial[i]
    total_time += trial[-1] - first_time
    return total_time

trial_times = np.array([compute_trial_time(trial) for trial in np.array(times)])

# Convert to numpy arrays
emissions = np.array(emissions_list)
times = np.array(times)
b_list = np.array(b_list)

def compute_state_intervals(times, emissions, blist):
    """
    Compute time intervals spent in each state
    Returns: dict mapping state to list of time intervals
    """
    state_intervals = {}
    n_trials, trial_length = times.shape
    
    for trial in range(n_trials):
        for t in range(trial_length - 1):
            if times[trial, t+1] < times[trial, t]:  # Wrap-around
                if blist[trial, t+1, 0] > blist[trial, t, 1]:
                    delta_t = blist[trial, t+1, 0] - blist[trial, t, 1]
                else:
                    continue
            else:
                delta_t = times[trial, t+1] - times[trial, t]
                
            if delta_t >= 2000:
                print(trial)
                continue
                
            state = emissions[trial, t]
            if state not in state_intervals:
                state_intervals[state] = []
            state_intervals[state].append(delta_t)
    
    return state_intervals

state_intervals = compute_state_intervals(times[:], emissions[:, :, 0], b_list)

# Create DataFrame for visualization
rows = []
for state, intervals in state_intervals.items():
    for interval in intervals:
        rows.append({'state': state, 'interval': interval/30})
df = pd.DataFrame(rows)

# Plot state interval distributions
states = sorted(df['state'].unique())
plt.figure(figsize=(35, 6))
ax = sns.boxplot(data=df, x='state', y='interval', order=states)
ax.set_xlabel('State')
ax.set_ylabel('Time Interval (s)')
ax.set_title('Time Interval Distribution and Number of Visitations per State')

# Overlay visitation counts
counts = [df[df['state'] == state].shape[0] for state in states]
ax2 = ax.twinx()
ax2.plot(range(len(states)), counts, color='red', marker='o', linestyle='-', linewidth=2, label='Number of Visitations')
ax2.set_ylabel('Number of Visitations', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')
plt.show()

def convert_emissions(times, emissions, blist, trans_probs):
    """
    Convert emissions sequence with temporal expansion
    
    Expands sequences based on time intervals:
    - < 60 frames: 1 timepoint
    - 60-500 frames: 2 timepoints
    - > 500 frames: 3 timepoints
    
    Returns: (n_trials, 500, 2) array with state and action
    """
    n_trials, trial_length = times.shape
    new_emissions = []

    for trial in range(n_trials):
        expanded = []
        
        for t in range(trial_length - 1):
            # Compute time interval
            if times[trial, t+1] < times[trial, t]:
                if blist[trial, t+1, 0] > blist[trial, t, 1]:
                    delta_t = blist[trial, t+1, 0] - blist[trial, t, 1]
                else:
                    delta_t = 1
            else:
                delta_t = times[trial, t+1] - times[trial, t]

            # Determine expansion factor
            if delta_t < 60:
                num_timepoints = 1
            elif 60 <= delta_t <= 500:
                num_timepoints = 2
            else:
                num_timepoints = 3

            curr_state = emissions[trial, t, 0]
            provided_action = emissions[trial, t, 1]
            
            # Expand timepoints
            if num_timepoints == 1:
                expanded.append((curr_state, provided_action))
            else:
                # Sample self-transition actions
                p = trans_probs[curr_state, :, curr_state]
                p_sum = np.sum(p)
                if p_sum == 0:
                    raise ValueError("No valid self-transitions")
                p_norm = p / p_sum
                sampled_actions = np.random.choice(np.arange(len(p_norm)), 
                                                  size=num_timepoints-1, p=p_norm)
                for act in sampled_actions:
                    expanded.append((curr_state, act))
                expanded.append((curr_state, provided_action))
        
        expanded.append((emissions[trial, -1, 0], emissions[trial, -1, 1]))
        
        # Ensure exactly 500 timepoints
        if len(expanded) >= 500:
            truncated = expanded[:500]
        else:
            raise ValueError(f"Trial {trial} has insufficient timepoints")
        new_emissions.append(truncated)
    
    return np.array(new_emissions)

converted_emissions = convert_emissions(times, emissions, b_list, trans_probs)

# Plot state occurrence distribution
states_flat = converted_emissions[:, :, 0].flatten()
unique_states, counts = np.unique(states_flat, return_counts=True)

df_counts = pd.DataFrame({
    'state': unique_states,
    'count': counts
})

plt.figure(figsize=(35, 6))
sns.barplot(data=df_counts, x='state', y='count', palette='viridis')
plt.xlabel('State')
plt.ylabel('Occurrence Count')
plt.title('Number of Occurrences of Each State in Converted Emissions')
plt.show()

# Save processed data
np.save('converted_emissions500new.npy', converted_emissions)  # Time-expanded emissions
np.save('emissions500new.npy', np.array(emissions_list))  # Raw emissions
xy_list = np.array(xy_list, dtype=object)
np.save('xy_list500new.npy', xy_list, allow_pickle=True)  # Position coordinates

print("Preprocessing complete!")
print(f"Converted emissions shape: {converted_emissions.shape}")