# data/formation_dist.py
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Formations we support in our action space
SUPPORTED_FORMATIONS = [
    '4-4-2', '4-3-3', '4-2-3-1', '3-5-2', '4-5-1', '5-3-2', '3-4-3'
]


def build_formation_distribution(
    formation_df: pd.DataFrame,
    match_state_bins: bool = True
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Build P(formation | match_state) reference distributions from StatsBomb data.

    match_state is discretized as (score_diff_bin, time_bin):
      - score_diff: -2 (losing badly), -1 (losing), 0 (drawing), +1 (winning), +2 (winning big)
      - time_bin: 0 (0-30 min), 1 (31-60 min), 2 (61-90 min)

    Returns: dict mapping (score_diff_bin, time_bin) -> np.array of shape (len(SUPPORTED_FORMATIONS),)

    CRITICAL: Apply Laplace smoothing (alpha=1) to all bins.
    Without it, any unseen (formation, state) pair gets P=0, which makes KL divergence = inf.
    """
    formation_df = formation_df.copy()

    # Normalize formation strings — StatsBomb uses both '442' and '4-4-2'
    def normalize_formation(f):
        if pd.isna(f):
            return None
        f = str(f).strip()
        # Convert '442' -> '4-4-2', '433' -> '4-3-3', etc.
        if '-' not in f and len(f) in [3, 4, 5]:
            parts = list(f)
            return '-'.join(parts)
        return f

    formation_df['formation_norm'] = formation_df['formation'].apply(normalize_formation)
    # Keep only formations in our action space
    formation_df = formation_df[formation_df['formation_norm'].isin(SUPPORTED_FORMATIONS)]

    # Compute score differential
    formation_df['score_diff'] = (
        formation_df.get('scoreline_home', pd.Series(0, index=formation_df.index))
        - formation_df.get('scoreline_away', pd.Series(0, index=formation_df.index))
    )
    formation_df['score_diff_bin'] = formation_df['score_diff'].clip(-2, 2).astype(int)

    # Time bins — clip minute to 90 first so extra-time values don't produce NaN
    formation_df['time_bin'] = pd.cut(
        formation_df['minute'].clip(0, 90),
        bins=[0, 30, 60, 90],
        labels=[0, 1, 2],
        include_lowest=True
    ).fillna(2).astype(int)

    n_formations = len(SUPPORTED_FORMATIONS)
    alpha = 1  # Laplace smoothing

    dist: Dict[Tuple[int, int], np.ndarray] = {}
    for score_diff in range(-2, 3):
        for time_bin in range(3):
            mask = (
                (formation_df['score_diff_bin'] == score_diff) &
                (formation_df['time_bin'] == time_bin)
            )
            subset = formation_df[mask]

            counts = np.zeros(n_formations) + alpha  # Laplace prior
            for i, formation in enumerate(SUPPORTED_FORMATIONS):
                counts[i] += (subset['formation_norm'] == formation).sum()

            dist[(score_diff, time_bin)] = counts / counts.sum()

    return dist


def save_distribution(dist: Dict, path: str = 'tacticrl/data/outputs/formation_dist.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'dist': dist, 'formations': SUPPORTED_FORMATIONS}, f)
    print(f"Saved distribution to {path}")


def load_distribution(path: str = 'tacticrl/data/outputs/formation_dist.pkl') -> Tuple[Dict, List[str]]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['dist'], data['formations']


def compute_kl_divergence(agent_dist: np.ndarray, reference_dist: np.ndarray) -> float:
    """
    KL(agent || reference).

    Both distributions MUST sum to 1 and have no zeros.
    The reference_dist from build_formation_distribution() is always safe (Laplace smoothed).
    The agent_dist comes from counting agent actions — add a tiny epsilon before normalizing.
    """
    epsilon = 1e-10
    p = np.clip(agent_dist, epsilon, 1.0)
    q = np.clip(reference_dist, epsilon, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))
