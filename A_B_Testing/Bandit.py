"""
Run this file at first, in order to see what it prints.
Instead of print(), use the respective log level via loguru.
"""

from abc import ABC, abstractmethod
from loguru import logger


class Bandit(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.

    Subclasses must implement the abstract methods:
        - __init__(p)
        - __repr__()
        - pull()
        - update(arm, reward)
        - experiment(n_trials=20000)
        - report()

    Notes:
        This class contains only abstract method signatures and should not be modified.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit with true arm means.

        Args:
            p (Sequence[float]): True mean rewards for each arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of the bandit instance.
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Select an arm and return the observed reward.

        Returns:
            tuple: (arm_index (int), reward (float))
        """
        pass

    @abstractmethod
    def update(self, arm, reward):
        """
        Update the estimates for the chosen arm based on observed reward.

        Args:
            arm (int): Index of the chosen arm.
            reward (float): Reward obtained from the arm.
        """
        pass

    @abstractmethod
    def experiment(self, n_trials=20000):
        """
        Run the experiment for a specified number of trials.

        Args:
            n_trials (int): Number of trials to simulate.

        Returns:
            pandas.DataFrame: DataFrame recording per-trial arm, reward, regret, and algorithm.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Summarize the experiment:
            - store results in a CSV
            - print average reward and regret using loguru
        """
        pass


class Visualization:
    """
    Visualization utilities for comparing bandit algorithms.

    Methods:
        plot1(): Plots learning curves of rewards over time.
        plot2(): Plots cumulative reward and cumulative regret.
    """

    def plot1(self):
        """Visualize the performance of each bandit algorithm over time."""
        pass

    def plot2(self):
        """Compare cumulative rewards and regrets of Epsilon-Greedy vs Thompson Sampling."""
        pass


class EpsilonGreedy(Bandit):
    """Placeholder class for Epsilon-Greedy algorithm."""
    pass


class ThompsonSampling(Bandit):
    """Placeholder class for Thompson Sampling algorithm."""
    pass


def comparison():
    """Utility function to visually compare the performances of the two algorithms."""
    pass


if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
