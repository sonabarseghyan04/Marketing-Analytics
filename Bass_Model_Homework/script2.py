import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import bass_cumulative, bass_incremental


p, q, m = 0.005149, 0.175320, 1000.00

years = np.arange(2015, 2026)
t_future = np.arange(0, len(years) + 15)

new_pred = bass_incremental(t_future, p, q, m)

plt.figure(figsize=(8,5))
plt.bar(years[0] + t_future, new_pred, color="skyblue", edgecolor="black")
plt.xlabel("Year")
plt.ylabel("New Adopters (millions)")
plt.title("Bass Model Prediction - New Adopters per Year")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.savefig("img/bass_forecast.png", dpi=300, bbox_inches="tight")
plt.show()

forecast_df = pd.DataFrame({
    "year": years[0] + t_future,
    "new_adopters": new_pred,
    "cumulative_adopters": bass_cumulative(t_future, p, q, m)
})
forecast_df.to_csv("data/dataset2.csv", index=False)
