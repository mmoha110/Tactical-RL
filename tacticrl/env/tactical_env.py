# env/tactical_env.py
import gym
import numpy as np
import gfootball.env as football_env
from gym import spaces

from tacticrl.env.obs_wrapper import FlatObsWrapper
from tacticrl.env.reward_shaper import RewardShaper

FORMATIONS = ['4-4-2', '4-3-3', '4-2-3-1', '3-5-2', '4-5-1', '5-3-2', '3-4-3']
PRESSING_LEVELS = ['low', 'medium', 'high']  # 3 levels


class TacticalEnv(gym.Env):
    """
    Two-level action space:
      - Low-level: gfootball's 19 discrete actions (move, pass, shoot, etc.)
      - High-level (tactical): formation choice + pressing level, applied every K steps

    Implemented as a FLAT discrete action space:
      actions 0-18:  gfootball base actions
      actions 19-25: formation switches (7 formations)
      actions 26-28: pressing intensity changes (3 levels)

    Total: 29 discrete actions.

    Tactical actions are only "applied" every TACTICAL_INTERVAL steps.
    In between, the env remembers the current tactic.

    NOTE: ALWAYS set render=False for training. render=True requires a display server.
    """

    TACTICAL_INTERVAL = 50  # Apply tactical decisions every 50 env steps
    N_BASE_ACTIONS = 19
    N_FORMATIONS = 7
    N_PRESSING = 3

    def __init__(
        self,
        env_name: str = 'academy_3_vs_1_with_keeper',
        render: bool = False,
        reward_weights: dict = None,
        formation_reference_dist: dict = None
    ):
        super().__init__()

        self._base_env = football_env.create_environment(
            env_name=env_name,
            representation='simple115v2',
            number_of_left_players_agent_controls=1,
            render=render,
            write_video=False,
            dump_frequency=0,  # Disable video dumps — they slow training massively
            logdir='/tmp/tacticrl_logs'
        )

        # Wrap obs
        self._wrapped_env = FlatObsWrapper(self._base_env)

        self.action_space = spaces.Discrete(
            self.N_BASE_ACTIONS + self.N_FORMATIONS + self.N_PRESSING
        )
        self.observation_space = self._wrapped_env.observation_space

        self._reward_shaper = RewardShaper(weights=reward_weights)
        self._formation_dist = formation_reference_dist  # For KL logging

        # State tracking
        self._current_formation = '4-4-2'
        self._current_pressing = 'medium'
        self._step_count = 0
        self._episode_formations = []
        self._episode_pressing_events = []

    def reset(self):
        obs = self._wrapped_env.reset()
        self._step_count = 0
        self._current_formation = '4-4-2'
        self._current_pressing = 'medium'
        self._episode_formations = []
        self._episode_pressing_events = []
        return obs

    def step(self, action: int):
        tactical_action = None

        if action >= self.N_BASE_ACTIONS + self.N_FORMATIONS:
            # Pressing action
            pressing_idx = action - self.N_BASE_ACTIONS - self.N_FORMATIONS
            self._current_pressing = PRESSING_LEVELS[pressing_idx]
            self._episode_pressing_events.append({
                'step': self._step_count,
                'pressing': self._current_pressing
            })
            base_action = 0  # No-op for base env
            tactical_action = 'pressing'

        elif action >= self.N_BASE_ACTIONS:
            # Formation action
            formation_idx = action - self.N_BASE_ACTIONS
            new_formation = FORMATIONS[formation_idx]
            if new_formation != self._current_formation:
                self._current_formation = new_formation
                self._wrapped_env.set_formation(new_formation)
                self._episode_formations.append({
                    'step': self._step_count,
                    'formation': new_formation
                })
            base_action = 0  # No-op for base env
            tactical_action = 'formation'

        else:
            base_action = action

        obs, base_reward, done, info = self._wrapped_env.step(base_action)
        self._step_count += 1

        # Shape reward
        shaped_reward = self._reward_shaper.shape(
            base_reward=base_reward,
            tactical_action=tactical_action,
            pressing_level=self._current_pressing,
            info=info
        )

        info['formation'] = self._current_formation
        info['pressing'] = self._current_pressing
        info['episode_formations'] = self._episode_formations

        return obs, shaped_reward, done, info

    def close(self):
        self._base_env.close()
