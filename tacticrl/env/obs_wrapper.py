# env/obs_wrapper.py
import gym
import numpy as np
from gym import spaces


class FlatObsWrapper(gym.ObservationWrapper):
    """
    Wraps gfootball's simple115v2 observation into a flat Box space
    compatible with SB3's PPO.

    Also appends tactical context: current formation index, score diff, time remaining.

    NOTE: SB3 PPO requires Box observation space. gfootball's dict obs WILL crash SB3.
    NOTE: simple115v2 already returns a flat array — but we augment it with tactical state.
    """

    FORMATION_TO_IDX = {
        '4-4-2': 0, '4-3-3': 1, '4-2-3-1': 2,
        '3-5-2': 3, '4-5-1': 4, '5-3-2': 5, '3-4-3': 6
    }
    N_FORMATIONS = 7

    def __init__(self, env):
        super().__init__(env)
        # simple115v2 gives 115-dim obs
        base_dim = 115
        # We add: formation_one_hot (7) + score_diff (1) + time_remaining_norm (1)
        extra_dim = self.N_FORMATIONS + 2
        total_dim = base_dim + extra_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,), dtype=np.float32
        )
        self._current_formation_idx = 0

    def observation(self, obs):
        # obs is already flat (115,) from simple115v2
        if isinstance(obs, dict):
            # Should not happen with simple115v2, but guard anyway
            obs = obs.get('obs', np.zeros(115, dtype=np.float32))

        obs = np.asarray(obs, dtype=np.float32)

        # Formation one-hot
        formation_oh = np.zeros(self.N_FORMATIONS, dtype=np.float32)
        formation_oh[self._current_formation_idx] = 1.0

        # Score diff (normalized to [-1, 1] assuming max 5 goal diff)
        # NOTE: gfootball's internal obs access — may differ by version.
        # Safe fallback: use 0 if unavailable.
        try:
            raw_obs = self.env.unwrapped._env.observation()
            score_diff = (raw_obs[0]['score'][0] - raw_obs[0]['score'][1]) / 5.0
            score_diff = float(np.clip(score_diff, -1.0, 1.0))
            # Normalized time remaining [0, 1], 0 = just started, 1 = full time
            time_remaining = raw_obs[0].get('steps_left', 3000) / 3000.0
        except Exception:
            score_diff = 0.0
            time_remaining = 1.0

        augmented = np.concatenate([
            obs,
            formation_oh,
            np.array([score_diff, time_remaining], dtype=np.float32)
        ])
        return augmented

    def set_formation(self, formation_name: str):
        """Called by TacticalEnv when the agent picks a new formation."""
        self._current_formation_idx = self.FORMATION_TO_IDX.get(formation_name, 0)
