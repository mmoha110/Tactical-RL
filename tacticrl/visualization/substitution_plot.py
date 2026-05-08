# visualization/substitution_plot.py
"""
Formation switch timeline: visualizes the sequence of tactical formation
changes the agent made across an episode, annotated with the score at each switch.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Dict

from tacticrl.data.formation_dist import SUPPORTED_FORMATIONS

FORMATION_COLORS = {
    '4-4-2':   '#1976D2',
    '4-3-3':   '#388E3C',
    '4-2-3-1': '#F57C00',
    '3-5-2':   '#7B1FA2',
    '4-5-1':   '#D32F2F',
    '5-3-2':   '#00796B',
    '3-4-3':   '#C62828',
}


def plot_substitution_timeline(
    formation_events: List[Dict],
    total_steps: int = 3000,
    output_path: str = 'outputs/substitution_timeline.png'
):
    """
    Gantt-style timeline showing which formation was active at each step.

    Args:
        formation_events: list of dicts with keys 'step' and 'formation'
                          (as collected by TacticalEnv._episode_formations)
        total_steps: total length of the episode in steps
        output_path: where to save the figure
    """
    if not formation_events:
        print("No formation events to plot.")
        return

    # Build intervals
    events_sorted = sorted(formation_events, key=lambda x: x['step'])
    intervals = []
    for k, ev in enumerate(events_sorted):
        start = ev['step']
        end = events_sorted[k + 1]['step'] if k + 1 < len(events_sorted) else total_steps
        intervals.append({'formation': ev['formation'], 'start': start, 'end': end})

    # If no initial event at step 0, prepend default
    if not intervals or intervals[0]['start'] > 0:
        first_start = intervals[0]['start'] if intervals else total_steps
        intervals.insert(0, {'formation': '4-4-2', 'start': 0, 'end': first_start})

    fig, ax = plt.subplots(figsize=(14, 3))
    for interval in intervals:
        formation = interval['formation']
        color = FORMATION_COLORS.get(formation, '#9E9E9E')
        width = interval['end'] - interval['start']
        ax.barh(
            0,
            width,
            left=interval['start'],
            height=0.6,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )
        # Label if wide enough
        if width > total_steps * 0.05:
            ax.text(
                interval['start'] + width / 2,
                0,
                formation,
                ha='center', va='center',
                fontsize=8, color='white', fontweight='bold'
            )

    # Legend
    legend_patches = [
        mpatches.Patch(color=FORMATION_COLORS.get(f, '#9E9E9E'), label=f)
        for f in SUPPORTED_FORMATIONS
        if any(iv['formation'] == f for iv in intervals)
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.9)

    ax.set_xlim(0, total_steps)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Episode Step')
    ax.set_yticks([])
    ax.set_title('Tactical Formation Timeline')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved substitution timeline to {output_path}")


def plot_formation_switch_frequency(
    all_episode_formations: List[List[Dict]],
    output_path: str = 'outputs/switch_frequency.png'
):
    """
    Histogram of how many formation switches happen per episode.

    Args:
        all_episode_formations: list of episode formation event lists
        output_path: where to save the figure
    """
    switch_counts = [len(ep) for ep in all_episode_formations]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(switch_counts, bins=range(0, max(switch_counts) + 2), color='#1976D2',
            alpha=0.85, edgecolor='white')
    ax.set_xlabel('Number of Formation Switches per Episode')
    ax.set_ylabel('Episode Count')
    ax.set_title('Formation Switch Frequency Distribution')
    ax.axvline(np.mean(switch_counts), color='red', linestyle='--',
               label=f'Mean: {np.mean(switch_counts):.1f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved switch frequency chart to {output_path}")
