import numpy as np
import pandas as pd
from loguru import logger
from Bandit import Bandit


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit algorithm.

    Attributes:
        p (np.ndarray): True means of arms.
        k (int): Number of arms.
        epsilon (float): Current exploration probability.
        counts (np.ndarray): Number of times each arm has been pulled.
        values (np.ndarray): Estimated mean reward per arm.
        rewards_history (list): Recorded rewards for each trial.
        arm_history (list): Indices of chosen arms per trial.
    """

    def __init__(self, p, epsilon=1.0):
        """
        Initialize epsilon-greedy algorithm.

        Args:
            p (Sequence[float]): True mean rewards for arms.
            epsilon (float, optional): Initial exploration probability. Default is 1.0.
        """
        self.p = p
        self.k = len(p)
        self.epsilon = epsilon
        self.counts = np.zeros(self.k)
        self.values = np.zeros(self.k)
        self.rewards_history = []
        self.arm_history = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon:.3f})"

    def pull(self):
        """
        Select an arm based on epsilon-greedy policy and sample reward.

        Returns:
            tuple: (arm_index (int), reward (float))
        """
        if np.random.random() < self.epsilon:
            arm = np.random.randint(self.k)
        else:
            arm = np.argmax(self.values)
        reward = np.random.normal(self.p[arm], 1)
        return arm, reward

    def update(self, arm, reward):
        """
        Update estimated mean and counts for the selected arm.

        Args:
            arm (int): Index of chosen arm.
            reward (float): Reward received from the arm.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def experiment(self, n_trials=20000):
        """
        Run epsilon-greedy experiment for n_trials with decaying epsilon.

        Args:
            n_trials (int): Number of trials to simulate.

        Returns:
            pandas.DataFrame: DataFrame of trial results with columns ['Arm', 'Reward', 'Regret', 'Algorithm'].
        """
        logger.info("Running Epsilon-Greedy Experiment")

        for t in range(1, n_trials + 1):
            self.epsilon = 1 / t
            arm, reward = self.pull()
            self.update(arm, reward)
            self.rewards_history.append(reward)
            self.arm_history.append(arm)

        regret = np.max(self.p) - np.array([self.p[a] for a in self.arm_history])
        df = pd.DataFrame({
            "Arm": self.arm_history,
            "Reward": self.rewards_history,
            "Regret": regret,
            "Algorithm": "Epsilon-Greedy"
        })
        return df

    def report(self):
        """Log average reward and average regret for the experiment."""
        avg_reward = np.mean(self.rewards_history)
        avg_regret = np.mean(np.max(self.p) - np.array(self.rewards_history))
        logger.info(f"[Epsilon-Greedy] Avg Reward: {avg_reward:.4f}")
        logger.info(f"[Epsilon-Greedy] Avg Regret: {avg_regret:.4f}")
