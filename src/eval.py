import numpy as np
import os
import argparse
from utils import set_seed, get_npy_files, calculate_metrics

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_dir', type=str, default='results_face')
argparser.add_argument('--target', type=str, default='fatigue')

args = argparser.parse_args()

set_seed(42)

npy_files = get_npy_files(args.log_dir, args.target)
print(f"Found {len(npy_files)} npy files in {args.log_dir}")

targets = []
outputs = []
for npy_file in npy_files:
    data = np.load(npy_file, allow_pickle=True).item()
    target = data['target']
    output = data['output']
    targets.append(target)
    outputs.append(output)

targets = np.concatenate(targets)
outputs = np.concatenate(outputs)

if len(targets.shape) == 2:
    targets = np.argmax(targets, axis=1)

metrics_results = calculate_metrics(outputs, targets)
print(metrics_results)