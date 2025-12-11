import re

def average_convergence_stats(log_file, n=5):
    converged_lines = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        if 'Converged' in lines[i]:
            prev_line_idx = i - 1
            while prev_line_idx >= 0 and 'Converged' in lines[prev_line_idx]:
                prev_line_idx -= 1
            if prev_line_idx >= 0:
                converged_lines.append((lines[prev_line_idx], lines[i]))
                if len(converged_lines) == n:
                    break

    if not converged_lines:
        print("No convergence lines found.")
        return None, None, None, None

    iterations = []
    total_times = []
    q_updates = []
    nn_times = []

    for prev_line, conv_line in converged_lines:
        iter_match = re.search(r'Iteration (\d+)', conv_line)
        total_match = re.search(r'Total time: ([\d.]+)s', conv_line)
        q_match = re.search(r'Q-update: ([\d.]+)s', prev_line)
        nn_match = re.search(r'NN: ([\d.]+)s', prev_line)

        if iter_match and total_match and q_match and nn_match:
            iterations.append(int(iter_match.group(1)))
            total_times.append(float(total_match.group(1)))
            q_updates.append(float(q_match.group(1)))
            nn_times.append(float(nn_match.group(1)))

    avg_iter = sum(iterations) / len(iterations)
    avg_time = sum(total_times) / len(total_times)
    avg_q = sum(q_updates) / len(q_updates)
    avg_nn = sum(nn_times) / len(nn_times)

    return avg_iter, avg_time, avg_q, avg_nn

# Usage:
avg_iter, avg_time, avg_q, avg_nn = average_convergence_stats('ll_pgiql_4hid-lstm.log')
if avg_iter is not None:
    print(f"Average iterations: {int(avg_iter)}, Average time: {int(avg_time)}s, "
          f"Average Q-update: {avg_q/10:.2f}s, Average NN: {avg_nn/10:.2f}s")