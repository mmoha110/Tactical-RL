# training/train.py
import argparse
import os

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tacticrl.data.formation_dist import build_formation_distribution, save_distribution, load_distribution
from tacticrl.data.statsbomb_loader import load_all_formation_changes
from tacticrl.env.tactical_env import TacticalEnv
from tacticrl.training.callbacks import CurriculumCallback, TacticRLCallback


def make_env(config: dict, render: bool = False):
    """
    Factory function for creating the environment.

    ALWAYS wrap with Monitor before VecEnv — otherwise episode stats are lost.
    ALWAYS wrap with DummyVecEnv even for single env — SB3 requires VecEnv.
    """
    try:
        ref_dist, _ = load_distribution()
    except FileNotFoundError:
        ref_dist = None

    def _make():
        env = TacticalEnv(
            env_name=config['env']['name'],
            render=render,
            reward_weights=config.get('reward_weights'),
            formation_reference_dist=ref_dist
        )
        env = Monitor(env)
        return env

    return _make


def train(config_path: str = 'configs/ppo_config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['tensorboard_log'], exist_ok=True)

    # Build/load formation reference distribution
    dist_path = 'tacticrl/data/outputs/formation_dist.pkl'
    if not os.path.exists(dist_path):
        print("Building StatsBomb formation distribution...")
        df = load_all_formation_changes()
        dist = build_formation_distribution(df)
        save_distribution(dist, dist_path)

    # Create vectorized training env
    env = DummyVecEnv([make_env(config)])

    # VecNormalize: normalize observations and rewards for stable training.
    # Save the normalization stats alongside the model — required for evaluation.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ppo_cfg = config['ppo']
    model = PPO(
        policy=ppo_cfg['policy'],
        env=env,
        learning_rate=ppo_cfg['learning_rate'],
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ppo_cfg['ent_coef'],
        vf_coef=ppo_cfg['vf_coef'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        tensorboard_log=config['training']['tensorboard_log'],
        verbose=1
    )

    # Create eval env (separate from training env)
    eval_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    callbacks = [
        TacticRLCallback(
            eval_env=eval_env,
            eval_freq=config['training']['eval_freq']
        ),
        CurriculumCallback(
            difficulty_schedule=config['curriculum']['bot_difficulty_schedule'],
            win_threshold=0.6
        )
    ]

    print(f"Starting training: {config['training']['total_timesteps']:,} timesteps")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callbacks,
        reset_num_timesteps=True
    )

    # Save model AND VecNormalize stats together
    checkpoint_dir = config['training']['checkpoint_dir']
    model.save(os.path.join(checkpoint_dir, 'final_model'))  # SB3 adds .zip automatically
    env.save(os.path.join(checkpoint_dir, 'vec_normalize_final.pkl'))
    print("Training complete. Model saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ppo_config.yaml')
    args = parser.parse_args()
    train(args.config)
