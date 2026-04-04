import numpy as np

def empirical_coverage(params):
    observations, lower_band, upper_band = params
    inside = np.sum((observations >= lower_band) & (observations <= upper_band))
    return inside / len(observations)