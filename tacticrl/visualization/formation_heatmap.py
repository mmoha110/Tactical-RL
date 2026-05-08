# visualization/formation_heatmap.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from tacticrl.data.formation_dist import SUPPORTED_FORMATIONS


def plot_formation_heatmap(
    formation_episode_log: list,
    output_path: str = 'outputs/formation_heatmap.png'
):
    """
    Plot formation usage heatmap by match state (score differential).

    Args:
        formation_episode_log: list of dicts with keys 'step', 'formation', 'score_diff'
        output_path: where to save the figure
    """
    score_bins = [-2, -1, 0, 1, 2]
    heatmap = np.zeros((len(SUPPORTED_FORMATIONS), len(score_bins)))

    for event in formation_episode_log:
        f = event.get('formation')
        s = int(np.clip(event.get('score_diff', 0), -2, 2))
        if f in SUPPORTED_FORMATIONS:
            fi = SUPPORTED_FORMATIONS.index(f)
            si = score_bins.index(s)
            heatmap[fi, si] += 1

    # Normalize each column
    col_sums = heatmap.sum(axis=0)
    col_sums[col_sums == 0] = 1
    heatmap_norm = heatmap / col_sums

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_norm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(score_bins)))
    ax.set_xticklabels([f'Score Diff {s:+d}' for s in score_bins])
    ax.set_yticks(range(len(SUPPORTED_FORMATIONS)))
    ax.set_yticklabels(SUPPORTED_FORMATIONS)
    ax.set_xlabel('Match State (Score Differential)')
    ax.set_ylabel('Formation')
    ax.set_title('Formation Usage Heatmap by Match State')
    plt.colorbar(im, ax=ax, label='Usage Proportion')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def plot_reward_curve(
    timesteps,
    rewards,
    output_path: str = 'outputs/reward_curve.png'
):
    """
    Plot smoothed reward curve from tensorboard data or callback logs.

    Args:
        timesteps: array-like of training step indices
        rewards:   array-like of mean episode rewards at each step
        output_path: where to save the figure
    """
    import numpy as np

    timesteps = np.asarray(timesteps)
    rewards = np.asarray(rewards, dtype=float)
    smooth_size = max(1, len(rewards) // 20)
    rewards_smooth = uniform_filter1d(rewards, size=smooth_size)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, rewards, alpha=0.2, color='steelblue', label='Raw')
    ax.plot(timesteps, rewards_smooth, color='steelblue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Training Reward Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reward curve to {output_path}")
