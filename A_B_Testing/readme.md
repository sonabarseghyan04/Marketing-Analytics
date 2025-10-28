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
```

# How to run
- Open experiment.ipynb in Jupyter Notebook or VSCode with the Jupyter extension.
- Run all cells to:
- Execute experiments for both algorithms
- Generate plots of rewards and regret
- Save results to results.csv

# Overview of algorithms
- Epsilon-Greedy: Selects the best-known option most of the time, but occasionally explores randomly to discover better options.
- Thompson Sampling: Uses Bayesian updating to probabilistically select options, balancing exploration and exploitation efficiently.

# Results
After running the notebook, results.csv will contain the cumulative rewards and regret for each algorithm, and visualizations will show performance comparison over time.
