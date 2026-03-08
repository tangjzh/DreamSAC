#!/usr/bin/env python3
"""Plot training loss curves from metrics.jsonl"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_metrics(jsonl_file, key):
    """Load metrics from jsonl file"""
    steps = []
    values = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if key in data:
                steps.append(data['step'])
                values.append(data[key])
    
    return np.array(steps), np.array(values)

def smooth(values, window=10):
    """Apply moving average smoothing"""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    return np.array(smoothed)

def plot_loss(logdir, keys=['train/loss/image'], smooth_window=0, output=None):
    """Plot loss curves"""
    logdir = Path(logdir)
    metrics_file = logdir / 'metrics.jsonl'
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    plt.figure(figsize=(12, 6))
    
    for key in keys:
        steps, values = load_metrics(metrics_file, key)
        
        if len(steps) == 0:
            print(f"Warning: No data found for {key}")
            continue
        
        print(f"{key}:")
        print(f"  Data points: {len(steps)}")
        print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"  First: {values[0]:.4f}, Last: {values[-1]:.4f}")
        
        # Plot raw data with transparency
        plt.plot(steps, values, alpha=0.7, label=f'{key} (raw)', linewidth=2)
        
        # Plot smoothed data if requested
        if smooth_window > 0:
            smoothed = smooth(values, smooth_window)
            plt.plot(steps, smoothed, label=f'{key} (smoothed)', linewidth=2)
        else:
            plt.plot(steps, values, label=key, linewidth=2)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if output is None:
        output = logdir / 'loss_curves.png'
    else:
        output = Path(output)
    
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output}")
    
    # Also show the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss curves')
    parser.add_argument('logdir', type=str, help='Directory containing metrics.jsonl')
    parser.add_argument('--keys', type=str, nargs='+', 
                        default=['train/loss/image'],
                        help='Metric keys to plot (default: train/loss/image)')
    parser.add_argument('--smooth', type=int, default=10,
                        help='Smoothing window size (0 for no smoothing, default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: logdir/loss_curves.png)')
    
    args = parser.parse_args()
    plot_loss(args.logdir, args.keys, args.smooth, args.output)

