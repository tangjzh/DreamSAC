"""
Plot results for specific parameter transfer learning experiments.
Compares:
1. Pre-training on standard parameters
2. Fine-tuning on modified parameters (from pretrained model)
3. Baseline: DreamerV3 from scratch on modified parameters
4. Zero-shot: Pretrained model directly on modified parameters
"""

import argparse
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_scores(scores_file):
    """Load scores from JSONL file."""
    scores = []
    steps = []
    
    with open(scores_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'step' in data and 'score' in data:
                steps.append(data['step'])
                scores.append(data['score'])
    
    return np.array(steps), np.array(scores)


def smooth(data, window=10):
    """Smooth data using moving average."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_comparison(experiment_dir, save_dir=None):
    """Plot comparison of different training approaches."""
    experiment_dir = pathlib.Path(experiment_dir)
    
    if save_dir is None:
        save_dir = experiment_dir
    else:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data from different stages
    data = {}
    
    # Fine-tuning (from pretrained)
    finetune_scores_file = experiment_dir / 'finetune_modified' / 'scores.jsonl'
    if finetune_scores_file.exists():
        steps, scores = load_scores(finetune_scores_file)
        data['finetune'] = {'steps': steps, 'scores': scores, 'label': 'Fine-tuning (Ours)', 'color': 'blue'}
    
    # Baseline (from scratch)
    baseline_scores_file = experiment_dir / 'baseline_dreamerv3' / 'scores.jsonl'
    if baseline_scores_file.exists():
        steps, scores = load_scores(baseline_scores_file)
        data['baseline'] = {'steps': steps, 'scores': scores, 'label': 'DreamerV3 Baseline', 'color': 'red'}
    
    # Zero-shot (if available)
    zeroshot_scores_file = experiment_dir / 'zeroshot_modified' / 'scores.jsonl'
    if zeroshot_scores_file.exists():
        steps, scores = load_scores(zeroshot_scores_file)
        data['zeroshot'] = {'steps': steps, 'scores': scores, 'label': 'Zero-shot', 'color': 'green'}
    
    if not data:
        print(f"No data found in {experiment_dir}")
        return
    
    # Extract experiment info from directory name
    experiment_name = experiment_dir.name
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========================================================================
    # Left plot: Learning curves
    # ========================================================================
    for key, info in data.items():
        steps = info['steps']
        scores = info['scores']
        label = info['label']
        color = info['color']
        
        # Raw data (lighter)
        ax1.plot(steps, scores, color=color, alpha=0.2, linewidth=0.5)
        
        # Smoothed data
        if len(scores) > 10:
            smoothed_scores = smooth(scores, window=10)
            smoothed_steps = steps[9:]  # Align with smoothed data
            ax1.plot(smoothed_steps, smoothed_scores, color=color, 
                    label=label, linewidth=2)
        else:
            ax1.plot(steps, scores, color=color, label=label, linewidth=2)
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Episode Return', fontsize=12)
    ax1.set_title(f'Learning Curves: {experiment_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Right plot: Final performance comparison (bar chart)
    # ========================================================================
    final_scores = {}
    final_stds = {}
    
    for key, info in data.items():
        scores = info['scores']
        # Take last 20% of data for final performance
        final_window = max(1, len(scores) // 5)
        final_data = scores[-final_window:]
        final_scores[info['label']] = np.mean(final_data)
        final_stds[info['label']] = np.std(final_data)
    
    labels = list(final_scores.keys())
    means = [final_scores[l] for l in labels]
    stds = [final_stds[l] for l in labels]
    colors = [info['color'] for info in data.values()]
    
    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Average Return (Final 20%)', fontsize=12)
    ax2.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = save_dir / 'transfer_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_file}")
    
    # plt.show()
    plt.close()
    
    # ========================================================================
    # Print summary statistics
    # ========================================================================
    print("\n" + "="*60)
    print(f"Experiment: {experiment_name}")
    print("="*60)
    
    for key, info in data.items():
        scores = info['scores']
        label = info['label']
        
        final_window = max(1, len(scores) // 5)
        final_data = scores[-final_window:]
        
        print(f"\n{label}:")
        print(f"  Final performance: {np.mean(final_data):.2f} ± {np.std(final_data):.2f}")
        print(f"  Max score: {np.max(scores):.2f}")
        print(f"  Total episodes: {len(scores)}")
    
    # Compute improvement
    if 'finetune' in data and 'baseline' in data:
        finetune_final = np.mean(data['finetune']['scores'][-max(1, len(data['finetune']['scores'])//5):])
        baseline_final = np.mean(data['baseline']['scores'][-max(1, len(data['baseline']['scores'])//5):])
        
        improvement = ((finetune_final - baseline_final) / abs(baseline_final)) * 100
        print(f"\n{'='*60}")
        print(f"Transfer Learning Improvement: {improvement:+.2f}%")
        print(f"{'='*60}\n")


def plot_multiple_experiments(base_logdir, pattern='specific_walker_walk_*'):
    """Plot results for multiple experiments matching a pattern."""
    base_logdir = pathlib.Path(base_logdir)
    
    experiment_dirs = sorted(base_logdir.glob(pattern))
    
    if not experiment_dirs:
        print(f"No experiments found matching pattern: {pattern}")
        return
    
    print(f"Found {len(experiment_dirs)} experiments:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")
    
    print("\nPlotting results...\n")
    
    for exp_dir in experiment_dirs:
        print(f"Processing: {exp_dir.name}")
        try:
            plot_comparison(exp_dir)
        except Exception as e:
            print(f"  Error: {e}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot specific transfer learning results')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment directory name')
    parser.add_argument('--logdir', type=str, default='logdir',
                       help='Base log directory')
    parser.add_argument('--pattern', type=str, default='specific_walker_walk_*',
                       help='Pattern to match experiment directories')
    parser.add_argument('--all', action='store_true',
                       help='Plot all experiments matching pattern')
    
    args = parser.parse_args()
    
    if args.experiment:
        # Plot specific experiment
        experiment_dir = pathlib.Path(args.logdir) / args.experiment
        if not experiment_dir.exists():
            print(f"Error: Experiment directory not found: {experiment_dir}")
        else:
            plot_comparison(experiment_dir)
    elif args.all:
        # Plot all matching experiments
        plot_multiple_experiments(args.logdir, args.pattern)
    else:
        # Default: plot all specific transfer experiments
        plot_multiple_experiments(args.logdir, 'specific_walker_walk_*')


