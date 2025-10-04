import numpy as np

def bass_cumulative(t, p, q, m):
    return m * (1 - np.exp(-(p + q) * t)) / (1 + (q/p) * np.exp(-(p + q) * t))

def bass_incremental(t, p, q, m):
    f = np.exp(-(p+q)*t)
    return m * (((p+q)**2) / p) * f / (1 + (q/p) * f)**2
