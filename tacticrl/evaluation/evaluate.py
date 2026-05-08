# evaluation/evaluate.py
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tacticrl.data.formation_dist import SUPPORTED_FORMATIONS, compute_kl_divergence, load_distribution
from tacticrl.env.tactical_env import TacticalEnv


def evaluate_model(
    model_path: str,
    vec_normalize_path: str,
    n_episodes: int = 100,
    env_name: str = 'academy_3_vs_1_with_keeper'
) -> dict:
    """
    Evaluates the trained model.

    MUST load VecNormalize with the SAME stats used during training.
    Set training=False on VecNormalize for eval — don't update running stats.
    model_path should NOT include .zip — SB3 adds it automatically.
    """
    ref_dist, _ = load_distribution()

    def _make_eval_env():
        env = TacticalEnv(
            env_name=env_name,
            render=False,
            formation_reference_dist=ref_dist
        )
        return Monitor(env)

    eval_env = DummyVecEnv([_make_eval_env])
    eval_env = VecNormalize.load(vec_normalize_path, eval_env)
    eval_env.training = False  # Critical: do not update normalization stats during eval
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'episode_rewards': [],
        'formation_counts': defaultdict(int),
        'pressing_events': [],
    }

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])

            ep_info = info[0] if info else {}
            formation = ep_info.get('formation', 'unknown')
            results['formation_counts'][formation] += 1

        results['episode_rewards'].append(ep_reward)
        if ep_reward > 0:
            results['wins'] += 1
        elif ep_reward < 0:
            results['losses'] += 1
        else:
            results['draws'] += 1

    n = n_episodes
    results['win_rate'] = results['wins'] / n
    results['loss_rate'] = results['losses'] / n
    results['draw_rate'] = results['draws'] / n
    results['mean_reward'] = float(np.mean(results['episode_rewards']))
    results['std_reward'] = float(np.std(results['episode_rewards']))

    # Formation entropy
    counts = np.array(
        [results['formation_counts'].get(f, 0) for f in SUPPORTED_FORMATIONS],
        dtype=float
    )
    if counts.sum() > 0:
        probs = counts / counts.sum()
        probs_safe = np.clip(probs, 1e-10, 1.0)
        results['formation_entropy'] = float(-np.sum(probs_safe * np.log(probs_safe)))
    else:
        results['formation_entropy'] = 0.0

    # KL divergence vs StatsBomb (use neutral state: score_diff=0, time_bin=1)
    ref = ref_dist.get(
        (0, 1),
        np.ones(len(SUPPORTED_FORMATIONS)) / len(SUPPORTED_FORMATIONS)
    )
    agent_dist = counts / max(counts.sum(), 1)
    results['kl_divergence'] = compute_kl_divergence(agent_dist, ref)

    eval_env.close()
    return results


def compare_baselines(
    model_path: str,
    vec_normalize_path: str,
    n_episodes: int = 50
) -> dict:
    """Compare PPO agent vs fixed formation baselines."""
    agent_results = evaluate_model(model_path, vec_normalize_path, n_episodes)

    baselines = {}
    for formation in ['4-4-2', '4-3-3']:
        baselines[formation] = _evaluate_fixed_formation(formation, n_episodes)

    return {'agent': agent_results, 'baselines': baselines}


def _evaluate_fixed_formation(formation: str, n_episodes: int) -> dict:
    """Run a random-action agent with a fixed formation as baseline."""
    import gfootball.env as football_env

    env = football_env.create_environment(
        env_name='academy_3_vs_1_with_keeper',
        representation='simple115v2',
        number_of_left_players_agent_controls=1,
        render=False
    )

    wins, losses, draws = 0, 0, 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()
    return {
        'win_rate': wins / n_episodes,
        'loss_rate': losses / n_episodes,
        'draw_rate': draws / n_episodes,
        'formation': formation,
    }
