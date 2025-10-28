# A/B Testing with Epsilon-Greedy and Thompson Sampling

This project implements two multi-armed bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling** — to simulate an A/B testing scenario with four advertisement options. The goal is to compare how both algorithms learn over time and optimize cumulative rewards.

## Folder Structure

```text
A_B_Testing/
├── Bandit.py              # Abstract base class for bandits
├── epsilon_greedy.py      # Epsilon-Greedy algorithm implementation
├── thompson_sampling.py   # Thompson Sampling algorithm implementation
├── visualizations.py      # Plotting and visualization functions
├── experiment.ipynb       # Jupyter notebook to run experiments and visualize results
├── requirements.txt       # Dependencies (e.g., loguru)
└── results.csv            # Output file storing experiment results (generated after running notebook)
