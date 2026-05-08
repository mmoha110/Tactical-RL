# verify_install.py — run this before anything else
import gfootball.env as football_env
import stable_baselines3
import statsbombpy
import gym
import torch

print(f"gfootball: OK")
print(f"SB3: {stable_baselines3.__version__}")
print(f"gym: {gym.__version__}")
print(f"torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Quick env smoke test
env = football_env.create_environment(
    env_name='academy_3_vs_1_with_keeper',
    representation='simple115v2',
    number_of_left_players_agent_controls=1,
    render=False  # ALWAYS False in cloud/headless
)
obs = env.reset()
print(f"Obs shape: {obs.shape}")  # Should be (115,)
env.close()
print("Installation verified.")
