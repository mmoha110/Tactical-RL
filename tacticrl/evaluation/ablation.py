# evaluation/ablation.py
"""
Ablation: PPO + curriculum vs PPO without curriculum.

Run two separate training runs (with/without curriculum) and compare:
  - Final win rate
  - Sample efficiency (wins per 1M steps)
  - Formation diversity (entropy)
  - KL divergence vs StatsBomb
"""
import matplotlib.pyplot as plt
import numpy as np


def run_ablation(n_timesteps: int = 3_000_000):
    """
    Placeholder to run both conditions and save comparison.

    NOTE: This takes significant compute — run on GPU/cloud, not laptop.
    Each condition: ~3-5 hours on a modern GPU for 3M steps.

    In practice, run two separate training jobs:
        python -m tacticrl.training.train --config configs/ppo_config.yaml
        python -m tacticrl.training.train --config configs/ppo_no_curriculum.yaml

    Then call evaluate_model() on each checkpoint and pass results here.
    """
    configs = {
        'with_curriculum': 'configs/ppo_config.yaml',
        'no_curriculum': 'configs/ppo_no_curriculum.yaml',
    }

    results = {}
    for condition, config_path in configs.items():
        print(f"\n=== Running ablation: {condition} ===")
        # Placeholder — replace with actual evaluate_model() calls after training
        # results[condition] = evaluate_model(f'checkpoints/{condition}/final_model', ...)
        pass

    return results


def plot_ablation_comparison(results: dict, output_path: str = 'outputs/ablation.png'):
    """
    Bar chart comparing the two conditions across three key metrics.

    Args:
        results: dict with keys 'with_curriculum' and 'no_curriculum',
                 each mapping to an evaluate_model() result dict.
        output_path: where to save the figure.
    """
    conditions = list(results.keys())
    metrics = ['win_rate', 'formation_entropy', 'kl_divergence']
    colors = ['#2196F3', '#FF5722']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for i, metric in enumerate(metrics):
        vals = [results[c].get(metric, 0) for c in conditions]
        bars = axes[i].bar(conditions, vals, color=colors, width=0.5)
        axes[i].set_title(metric.replace('_', ' ').title())
        max_val = max(vals) if max(vals) > 0 else 1
        axes[i].set_ylim(0, max_val * 1.3)
        for bar, val in zip(bars, vals):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10
            )

    plt.suptitle('Ablation Study: PPO+Curriculum vs PPO Only', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation plot to {output_path}")
