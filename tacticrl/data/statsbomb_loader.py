# data/statsbomb_loader.py
import ast
import os
import pickle

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')  # statsbombpy is verbose

from statsbombpy import sb

# Available free competitions (no auth required)
FREE_COMPETITIONS = {
    'World Cup 2018': {'competition_id': 43, 'season_id': 3},
    'Champions League 2018/19': {'competition_id': 16, 'season_id': 4},
    'La Liga 2020/21': {'competition_id': 11, 'season_id': 90},
    'FAWSL 2020/21': {'competition_id': 37, 'season_id': 90},
    'Euro 2020': {'competition_id': 55, 'season_id': 43},
}


def load_all_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Load all matches for a competition/season."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    return matches


def load_events_for_match(match_id: int) -> pd.DataFrame:
    """
    Load events for a single match.

    CRITICAL: statsbombpy returns nested dicts in some columns.
    Use split=False and flatten_attrs=False to get raw dicts, then manually
    extract what you need. flatten_attrs=True is unreliable across versions —
    in statsbombpy 1.1.0 it does NOT produce type_id / team_id columns.
    """
    events = sb.events(match_id=match_id, split=False, flatten_attrs=False)
    return events


def extract_formation_changes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract tactical formation changes from events.

    Filters on the 'tactics' field having a formation value rather than on a
    hardcoded type_id. This captures both 'Starting XI' and 'Tactical Shift'
    events across different StatsBomb schema versions without ID assumptions.
    """
    if 'tactics' not in events.columns:
        return pd.DataFrame()

    def has_formation(val):
        if isinstance(val, dict):
            return bool(val.get('formation'))
        return False

    tactical = events[events['tactics'].apply(has_formation)].copy()

    if tactical.empty:
        return pd.DataFrame()

    def safe_extract_formation(tactics_val):
        if isinstance(tactics_val, dict):
            return tactics_val.get('formation', None)
        if isinstance(tactics_val, str):
            try:
                return ast.literal_eval(tactics_val).get('formation', None)
            except Exception:
                return None
        return None

    tactical['formation'] = tactical['tactics'].apply(safe_extract_formation)
    tactical['minute'] = tactical['minute'].fillna(0).astype(int)

    # Extract team_id and team_name — 'team' column is a dict in statsbombpy 1.1.0
    if 'team' in tactical.columns:
        tactical['team_id'] = tactical['team'].apply(
            lambda x: x.get('id') if isinstance(x, dict) else x
        )
        tactical['team_name'] = tactical['team'].apply(
            lambda x: x.get('name') if isinstance(x, dict) else None
        )
    elif 'team_id' not in tactical.columns:
        tactical['team_id'] = None
        tactical['team_name'] = None

    # Ensure match_id column exists (statsbombpy adds it; guard just in case)
    if 'match_id' not in tactical.columns:
        tactical['match_id'] = None

    return tactical[['match_id', 'minute', 'formation', 'team_id', 'team_name']].dropna(
        subset=['formation']
    )


def load_all_formation_changes(max_matches_per_competition: int = 50) -> pd.DataFrame:
    """
    Load formation changes across all free competitions.
    This is the main data pipeline — takes ~5-10 min first run.
    Cache the output with pickle.
    """
    cache_path = 'tacticrl/data/outputs/all_formation_changes.pkl'

    if os.path.exists(cache_path):
        print("Loading from cache...")
        return pd.read_pickle(cache_path)

    all_changes = []

    for comp_name, ids in FREE_COMPETITIONS.items():
        print(f"Loading {comp_name}...")
        try:
            matches = load_all_matches(**ids)
            match_ids = matches['match_id'].tolist()[:max_matches_per_competition]

            for mid in match_ids:
                try:
                    events = load_events_for_match(mid)
                    changes = extract_formation_changes(events)
                    if not changes.empty:
                        changes['competition'] = comp_name
                        all_changes.append(changes)
                except Exception as e:
                    print(f"  Skipping match {mid}: {e}")
                    continue
        except Exception as e:
            print(f"  Skipping competition {comp_name}: {e}")
            continue

    if not all_changes:
        raise RuntimeError("No formation data loaded. Check StatsBomb connectivity.")

    result = pd.concat(all_changes, ignore_index=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result.to_pickle(cache_path)
    print(f"Loaded {len(result)} formation change events. Cached.")
    return result
