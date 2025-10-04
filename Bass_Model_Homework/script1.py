import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from helper_functions import bass_cumulative

years = np.arange(2015, 2026)
sales = np.array([3.52, 3.80, 5.15, 7.68, 9.00,
                  10.75, 12.61, 14.80, 17.41, 19.63, 22.10])
df = pd.DataFrame({"year": years, "sales": sales})
df["cum_sales"] = df["sales"].cumsum()

t = np.arange(len(years))
y = df["cum_sales"].values

initial_guess = [0.01, 0.4, max(y)*1.5]
params, _ = curve_fit(bass_cumulative, t, y, p0=initial_guess, bounds=(0, [1, 1, 1000]))
p, q, m = params

print("\n## Bass Model Parameters")
print(f"p: {p:.6f}, q: {q:.6f}, m: {m:.2f}")
