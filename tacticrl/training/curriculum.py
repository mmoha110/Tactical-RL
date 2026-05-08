# training/curriculum.py
"""
Self-play curriculum manager.

Manages opponent snapshots for self-play training.
The opponent is periodically updated to a snapshot of the current agent,
creating a curriculum that adapts as the agent improves.

NOTE: Checkpoint filenames must use the .zip extension — SB3 silently fails otherwise.
      When calling model.save(), do NOT add .zip yourself; SB3 adds it automatically.
      When calling PPO.load(), do NOT include .zip in the path.
"""
import os
import shutil
from typing import List, Optional


class CurriculumManager:
    """
    Manages self-play opponent snapshots.

    Usage:
        curriculum = CurriculumManager(snapshot_dir='checkpoints/self_play')
        # During training callback:
        curriculum.snapshot(model, step=n_calls)
        opponent_path = curriculum.sample_opponent()
    """

    def __init__(
        self,
        snapshot_dir: str = 'checkpoints/self_play',
        snapshot_freq: int = 200_000,
        max_snapshots: int = 5
    ):
        self.snapshot_dir = snapshot_dir
        self.snapshot_freq = snapshot_freq
        self.max_snapshots = max_snapshots
        os.makedirs(snapshot_dir, exist_ok=True)
        self._snapshots: List[str] = []

    def snapshot(self, model, step: int) -> str:
        """
        Save a snapshot of the current model as a potential future opponent.

        Returns the path of the saved snapshot (without .zip extension).
        """
        # SB3 adds .zip automatically — do NOT include it here
        path = os.path.join(self.snapshot_dir, f'opponent_step_{step}')
        model.save(path)
        self._snapshots.append(path)

        # Evict oldest snapshots beyond max_snapshots
        while len(self._snapshots) > self.max_snapshots:
            oldest = self._snapshots.pop(0)
            zip_path = oldest + '.zip'
            if os.path.exists(zip_path):
                os.remove(zip_path)

        return path

    def sample_opponent(self, strategy: str = 'latest') -> Optional[str]:
        """
        Sample an opponent snapshot path.

        strategy:
          'latest'  — use the most recent snapshot (default)
          'random'  — pick uniformly at random from all snapshots
          'oldest'  — use the oldest snapshot (most stable opponent)

        Returns path without .zip, or None if no snapshots exist.
        """
        if not self._snapshots:
            return None

        if strategy == 'random':
            import random
            return random.choice(self._snapshots)
        elif strategy == 'oldest':
            return self._snapshots[0]
        else:  # 'latest'
            return self._snapshots[-1]

    def list_snapshots(self) -> List[str]:
        """Return all currently tracked snapshot paths."""
        return list(self._snapshots)
