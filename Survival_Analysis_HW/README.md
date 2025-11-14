# Survival Analysis and Customer Lifetime Value (CLV) Project

## Project Overview
This project analyzes customer churn and lifetime value using a telco dataset of 1,000 subscribers. The main goals are to identify factors that influence churn, build parametric survival models, calculate Customer Lifetime Value (CLV), and explore customer segments to guide retention strategies.

I use the **Accelerated Failure Time (AFT) models** with different distributions (Weibull, Log-Normal, Log-Logistic, Generalized Gamma) to model churn. The final model is selected based on model fit, significance of features, and interpretability.

## Key Features
- Data preprocessing, one-hot encoding for categorical variables, and standardization for numerical features.
- Model fitting using Lifelines library for Python.
- Calculation of Customer Lifetime Value (CLV) for each subscriber.
- Exploration of CLV across customer segments such as service category, region, education, and service adoption (voice, internet, call forwarding).
- Identification of high-value and at-risk customer segments for retention planning.

## Files
- `telco.csv`: Dataset used for analysis.
- `MA_HW3_Sona_Barseghyan.ipynb`: Jupyter notebook containing all code, analysis, and visualizations.
- `requirements.txt`: List of Python packages required to run the project.

## How to Run
1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
Open MA_HW3_Sona_Barseghyan.ipynb in Jupyter Notebook or VS Code and run all cells.


