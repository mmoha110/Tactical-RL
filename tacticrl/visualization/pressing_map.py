# visualization/pressing_map.py
"""
Pressing intensity map: shows when and where the agent applied high/medium/low
pressing during a match or across episodes.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict


PRESSING_COLORS = {
    'low': '#4CAF50',     # green
    'medium': '#FF9800',  # orange
    'high': '#F44336',    # red
}


def plot_pressing_map(
    pressing_events: List[Dict],
    total_steps: int = 3000,
    output_path: str = 'outputs/pressing_map.png'
):
    """
    Visualize pressing intensity over time during an episode.

    Args:
        pressing_events: list of dicts with keys 'step' and 'pressing'
                         (as collected by TacticalEnv._episode_pressing_events)
        total_steps: total length of the episode in steps
        output_path: where to save the figure
    """
    if not pressing_events:
        print("No pressing events to plot.")
        return

    # Build step -> pressing array
    pressing_levels = ['low', 'medium', 'high']
    level_to_val = {lv: i for i, lv in enumerate(pressing_levels)}

    # Fill in pressing level for each step
    pressing_arr = np.ones(total_steps, dtype=int)  # default 'medium' = 1
    events_sorted = sorted(pressing_events, key=lambda x: x['step'])
    for k, ev in enumerate(events_sorted):
        start = ev['step']
        end = events_sorted[k + 1]['step'] if k + 1 < len(events_sorted) else total_steps
        level = ev.get('pressing', 'medium')
        val = level_to_val.get(level, 1)
        pressing_arr[start:end] = val

    fig, ax = plt.subplots(figsize=(14, 3))
    steps = np.arange(total_steps)

    for level, val in level_to_val.items():
        mask = pressing_arr == val
        if mask.any():
            ax.fill_between(
                steps,
                0,
                1,
                where=mask,
                color=PRESSING_COLORS[level],
                alpha=0.7,
                label=level.capitalize()
            )

    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Episode Step')
    ax.set_yticks([])
    ax.set_title('Pressing Intensity Over Episode')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pressing map to {output_path}")


def plot_pressing_frequency_by_state(
    episode_logs: List[Dict],
    output_path: str = 'outputs/pressing_frequency.png'
):
    """
    Bar chart: pressing level frequency broken down by score state.

    Args:
        episode_logs: list of episode info dicts (from TacticalEnv step info)
        output_path: where to save the figure
    """
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    for ep in episode_logs:
        pressing = ep.get('pressing', 'medium')
        score_diff = int(np.clip(ep.get('score_diff', 0), -2, 2))
        counts[score_diff][pressing] += 1

    score_diffs = sorted(counts.keys())
    pressing_levels = ['low', 'medium', 'high']
    x = np.arange(len(score_diffs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, level in enumerate(pressing_levels):
        vals = [counts[sd].get(level, 0) for sd in score_diffs]
        ax.bar(x + i * width, vals, width, label=level.capitalize(),
               color=PRESSING_COLORS[level], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Score {sd:+d}' for sd in score_diffs])
    ax.set_xlabel('Score Differential')
    ax.set_ylabel('Event Count')
    ax.set_title('Pressing Level Frequency by Match State')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pressing frequency chart to {output_path}")
