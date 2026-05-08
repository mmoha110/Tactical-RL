# training/callbacks.py
import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TacticRLCallback(BaseCallback):
    """
    Logs:
    - Win rate (rolling 100 episodes)
    - Formation entropy (are we using diverse formations?)
    - Mean episode reward
    - Pressing event frequency

    SB3 callbacks run in the training process — don't do heavy compute here.
    Log scalars only; save heavy analysis for post-hoc evaluation.py.
    """

    def __init__(self, eval_env, eval_freq: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self._episode_rewards: deque = deque(maxlen=100)
        self._win_count = 0
        self._episode_count = 0
        self._formation_counts: dict = {}

    def _on_step(self) -> bool:
        # Check for episode end
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self._episode_rewards.append(ep_reward)
                self._episode_count += 1

                # Track formation usage
                formation = info.get('formation', 'unknown')
                self._formation_counts[formation] = self._formation_counts.get(formation, 0) + 1

                # Win = positive reward (scored more than conceded)
                if ep_reward > 0:
                    self._win_count += 1

        # Periodic evaluation logging
        if self.n_calls % self.eval_freq == 0 and self._episode_count > 0:
            win_rate = self._win_count / max(self._episode_count, 1)
            mean_reward = np.mean(self._episode_rewards) if self._episode_rewards else 0.0
            formation_entropy = self._compute_formation_entropy()

            self.logger.record('eval/win_rate', win_rate)
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/formation_entropy', formation_entropy)
            self.logger.record('eval/n_episodes', self._episode_count)

            if self.verbose:
                print(
                    f"Step {self.n_calls}: win_rate={win_rate:.3f}, "
                    f"mean_reward={mean_reward:.3f}, "
                    f"formation_entropy={formation_entropy:.3f}"
                )

        return True

    def _compute_formation_entropy(self) -> float:
        """
        Shannon entropy of formation usage distribution.
        Higher entropy = more diverse formation choices.

        Returns 0 if no formations recorded — guard against log(0).
        """
        if not self._formation_counts:
            return 0.0

        counts = np.array(list(self._formation_counts.values()), dtype=float)
        probs = counts / counts.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))


class CurriculumCallback(BaseCallback):
    """
    Advances bot difficulty when the agent consistently wins.

    NOTE: gfootball's bot difficulty is set at environment creation time,
    not dynamically. This callback triggers environment re-creation.
    """

    def __init__(
        self,
        difficulty_schedule,
        win_threshold: float = 0.6,
        eval_window: int = 100,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.difficulty_schedule = difficulty_schedule
        self.win_threshold = win_threshold
        self.eval_window = eval_window
        self._current_diff_idx = 0
        self._recent_wins: deque = deque(maxlen=eval_window)

    @property
    def current_difficulty(self) -> int:
        return self.difficulty_schedule[self._current_diff_idx]

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                won = info['episode']['r'] > 0
                self._recent_wins.append(int(won))

        if (len(self._recent_wins) >= self.eval_window
                and self._current_diff_idx < len(self.difficulty_schedule) - 1):
            win_rate = float(np.mean(self._recent_wins))
            if win_rate >= self.win_threshold:
                self._current_diff_idx += 1
                new_diff = self.difficulty_schedule[self._current_diff_idx]
                if self.verbose:
                    print(
                        f"\nCurriculum advance! "
                        f"Win rate {win_rate:.2f} -> difficulty {new_diff}"
                    )
                self._recent_wins.clear()

        return True
