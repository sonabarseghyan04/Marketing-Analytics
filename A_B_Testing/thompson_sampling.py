import numpy as np
import pandas as pd
from loguru import logger
from Bandit import Bandit


class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit algorithm using Beta prior (simplified for reward > 0).

    Attributes:
        p (np.ndarray): True means of arms.
        k (int): Number of arms.
        alpha (np.ndarray): Alpha parameters of Beta distributions.
        beta (np.ndarray): Beta parameters of Beta distributions.
        rewards_history (list): Rewards collected per trial.
        arm_history (list): Chosen arm indices per trial.
    """

    def __init__(self, p):
        """
        Initialize Thompson Sampling algorithm.

        Args:
            p (Sequence[float]): True mean rewards for arms.
        """
        self.p = p
        self.k = len(p)
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)
        self.rewards_history = []
        self.arm_history = []

    def __repr__(self):
        return "ThompsonSampling()"

    def pull(self):
        """
        Sample a candidate mean for each arm from posterior Beta and select max.

        Returns:
            tuple: (arm_index (int), reward (float))
        """
        theta = np.random.beta(self.alpha, self.beta)
        arm = np.argmax(theta)
        reward = np.random.normal(self.p[arm], 1)
        return arm, reward

    def update(self, arm, reward):
        """
        Update Beta posterior parameters for the chosen arm.

        Args:
            arm (int): Chosen arm index.
            reward (float): Observed reward.
        """
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, n_trials=20000):
        """
        Run Thompson Sampling experiment for n_trials.

        Args:
            n_trials (int): Number of trials.

        Returns:
            pandas.DataFrame: DataFrame of trial results ['Arm', 'Reward', 'Regret', 'Algorithm'].
        """
        logger.info("Running Thompson Sampling Experiment")

        for _ in range(n_trials):
            arm, reward = self.pull()
            self.update(arm, reward)
            self.arm_history.append(arm)
            self.rewards_history.append(reward)

        regret = np.max(self.p) - np.array([self.p[a] for a in self.arm_history])
        df = pd.DataFrame({
            "Arm": self.arm_history,
            "Reward": self.rewards_history,
            "Regret": regret,
            "Algorithm": "Thompson Sampling"
        })
        return df

    def report(self):
        """Log average reward and average regret for the experiment."""
        avg_reward = np.mean(self.rewards_history)
        avg_regret = np.mean(np.max(self.p) - np.array(self.rewards_history))
        logger.info(f"[Thompson Sampling] Avg Reward: {avg_reward:.4f}")
        logger.info(f"[Thompson Sampling] Avg Regret: {avg_regret:.4f}")
