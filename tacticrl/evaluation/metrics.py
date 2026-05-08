# evaluation/metrics.py
"""
Standalone metric computation utilities for post-hoc analysis.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from tacticrl.data.formation_dist import SUPPORTED_FORMATIONS, compute_kl_divergence


def compute_formation_entropy(formation_counts: Dict[str, int]) -> float:
    """
    Shannon entropy of formation usage.
    Higher = more diverse tactical choices.

    Args:
        formation_counts: dict mapping formation string to count

    Returns:
        Entropy in nats. 0 if no data.
    """
    if not formation_counts:
        return 0.0

    counts = np.array(list(formation_counts.values()), dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_kl_vs_reference(
    formation_counts: Dict[str, int],
    ref_dist: Dict[Tuple[int, int], np.ndarray],
    score_diff: int = 0,
    time_bin: int = 1
) -> float:
    """
    Compute KL(agent || StatsBomb_reference) for a given match state.

    Args:
        formation_counts: dict mapping formation string to episode count
        ref_dist: loaded reference distribution (from load_distribution)
        score_diff: score differential bin [-2, 2], default 0 (drawing)
        time_bin: time bin [0, 1, 2], default 1 (31-60 min)

    Returns:
        KL divergence scalar.
    """
    counts = np.array(
        [formation_counts.get(f, 0) for f in SUPPORTED_FORMATIONS],
        dtype=float
    )
    agent_dist = counts / max(counts.sum(), 1.0)

    ref = ref_dist.get(
        (score_diff, time_bin),
        np.ones(len(SUPPORTED_FORMATIONS)) / len(SUPPORTED_FORMATIONS)
    )
    return compute_kl_divergence(agent_dist, ref)


def compute_win_rate_curve(
    episode_rewards: List[float],
    window: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling win rate over episodes.

    Returns:
        (episode_indices, win_rate_array)
    """
    rewards = np.array(episode_rewards)
    wins = (rewards > 0).astype(float)

    indices = []
    rates = []
    for i in range(window, len(wins) + 1):
        indices.append(i)
        rates.append(float(np.mean(wins[i - window:i])))

    return np.array(indices), np.array(rates)


def compute_formation_transition_matrix(
    episode_formation_logs: List[List[Dict]]
) -> np.ndarray:
    """
    Build a formation transition probability matrix from episode logs.

    Args:
        episode_formation_logs: list of lists, each inner list is
            [{'step': int, 'formation': str}, ...] for one episode.

    Returns:
        (N x N) transition matrix where entry [i, j] is P(switch to j | currently i).
        N = len(SUPPORTED_FORMATIONS).
    """
    n = len(SUPPORTED_FORMATIONS)
    idx = {f: i for i, f in enumerate(SUPPORTED_FORMATIONS)}
    counts = np.zeros((n, n), dtype=float)

    for ep_log in episode_formation_logs:
        for k in range(len(ep_log) - 1):
            src = ep_log[k].get('formation')
            dst = ep_log[k + 1].get('formation')
            if src in idx and dst in idx:
                counts[idx[src], idx[dst]] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums
