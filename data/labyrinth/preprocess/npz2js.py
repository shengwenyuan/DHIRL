import json
import numpy as np
import os


script_dir = os.path.dirname(__file__)
npy_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'emissions500new.npy'))
out_js = os.path.abspath(os.path.join(script_dir, '..', 'data', 'trajs.js'))

print('Loading:', npy_path)
arr = np.load(npy_path)
N, T, D = arr.shape
if D == 2:
	zeros = np.zeros((N, T, 1), dtype=arr.dtype)
	arr3 = np.concatenate([arr, zeros], axis=2)
else:
	arr3 = arr
py_data = arr3.tolist()

os.makedirs(os.path.dirname(out_js), exist_ok=True)
print('Writing:', out_js)
with open(out_js, 'w') as f:
	json.dump(py_data, f)
