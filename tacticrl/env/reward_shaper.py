# env/reward_shaper.py

DEFAULT_WEIGHTS = {
    'goal_scored': 1.0,
    'goal_conceded': -1.0,
    'possession_bonus': 0.001,   # Per step with possession
    'pressing_success': 0.05,    # Winning ball in opponent half
    'formation_switch_penalty': -0.01,  # Small penalty for excessive switching
}


class RewardShaper:
    """
    Shapes the base gfootball reward (+1 goal, -1 concede).

    Keep shaping bonuses SMALL relative to goal reward:
    - If possession_bonus > 0.01, the agent will learn to just hoard the ball.
    - If pressing_success > 0.2, agent will recklessly press and concede on counter.

    Start with DEFAULT_WEIGHTS and tune from there.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._prev_ball_owned_team = None

    def shape(self, base_reward: float, tactical_action, pressing_level: str, info: dict) -> float:
        reward = base_reward  # Raw goal reward from gfootball

        # Possession bonus (only if we have the ball)
        ball_owned = info.get('ball_owned_team', -1)
        if ball_owned == 0:  # 0 = left team (our agent)
            reward += self.weights['possession_bonus']

        # Pressing success: we gained possession in opponent half
        ball_x = info.get('ball', [0, 0, 0])[0] if 'ball' in info else 0
        if (self._prev_ball_owned_team != 0 and ball_owned == 0
                and ball_x > 0 and pressing_level == 'high'):
            reward += self.weights['pressing_success']

        self._prev_ball_owned_team = ball_owned

        return reward
