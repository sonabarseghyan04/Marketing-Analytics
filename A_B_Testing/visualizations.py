import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


class Visualization:
    """
    Visualization utilities for algorithm performance.

    Attributes:
        df_eg (pandas.DataFrame): Epsilon-Greedy experiment results.
        df_ts (pandas.DataFrame): Thompson Sampling experiment results.
    """

    def __init__(self, df_eg, df_ts):
        """
        Initialize with experiment DataFrames.

        Args:
            df_eg (pandas.DataFrame): Epsilon-Greedy results.
            df_ts (pandas.DataFrame): Thompson Sampling results.
        """
        self.df_eg = df_eg
        self.df_ts = df_ts

    def plot1(self):
        """Plot rolling mean of rewards over trials for each algorithm."""
        logger.info("Plot 1: Reward Performance Over Time")

        plt.figure()
        plt.plot(self.df_eg["Reward"].rolling(200).mean(), label="Epsilon-Greedy")
        plt.plot(self.df_ts["Reward"].rolling(200).mean(), label="Thompson Sampling")
        plt.title("Learning Curve (Rolling Mean of Rewards)")
        plt.xlabel("Trials")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()

    def plot2(self):
        """Plot cumulative rewards and cumulative regret for both algorithms."""
        logger.info("Plot 2: Cumulative Reward & Regret Comparison")

        plt.figure()
        plt.plot(self.df_eg["Reward"].cumsum(), label="Epsilon-Greedy")
        plt.plot(self.df_ts["Reward"].cumsum(), label="Thompson Sampling")
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.df_eg["Regret"].cumsum(), label="Epsilon-Greedy")
        plt.plot(self.df_ts["Regret"].cumsum(), label="Thompson Sampling")
        plt.title("Cumulative Regret")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.show()
