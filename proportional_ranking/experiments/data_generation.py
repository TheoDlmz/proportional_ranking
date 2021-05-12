import numpy as np


def generate_profile(n, m, proba=0.5):
    return np.random.rand(n, m) > proba
