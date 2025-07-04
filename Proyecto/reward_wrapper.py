import gymnasium as gym


class TraumaWrapper(gym.RewardWrapper):
    """A reward wrapper that applies trauma to the reward.

    Trauma is applied if the current step is the trauma step and the trauma has not occurred yet.
    Trauma is the product of the max recorded reward and the trauma multiplier.
    The max recorded reward is the maximum reward recorded during the episode.
    The trauma multiplier is a negative number that is used to reduce the reward.
    """

    def __init__(self, env, trauma_step=300_001, trauma_reward=-1_000_000):
        """Initialize the TraumaWrapper."""
        super().__init__(env)
        self.trauma_step = trauma_step
        self.trauma_reward = trauma_reward
        self.current_step = 0
        self.trauma_occurred = False

    def reset(self, **kwargs):
        """Reset the environment.

        This function is called when the environment is reset.
        It resets the current step, the trauma occurred flag, and the max recorded reward.
        """
        self.trauma_occurred = False
        return self.env.reset(**kwargs)

    def step(self, action):
        """Apply trauma to the reward.

        Trauma is the product of the max recorded reward and the trauma multiplier.

        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        # Apply trauma if on the target step
        if self.current_step == self.trauma_step and not self.trauma_occurred:
            reward = self.trauma_reward
            self.trauma_occurred = True

        return obs, reward, terminated, truncated, info
