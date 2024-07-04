
import numpy as np


def weighted_mean(means, std_devs):
    """
    Compute the weighted mean given means and standard deviations.
    
    :param means: List or numpy array of mean values (m_i)
    :param std_devs: List or numpy array of standard deviations (s_i)
    :return: Weighted mean

    """
    weights = 1 / (std_devs**2)
    weighted_mean = np.sum(weights*means) / np.sum(weights)
    return weighted_mean

def weighted_variance(std_devs):
    """
    Compute the weighted variance given standard deviations.
    
    :param std_devs: List or numpy array of standard deviations (s_i)
    :return: Weighted variance
    """
    weights = 1 / (std_devs**2)
    weighted_variance = 1 / np.sqrt(np.sum(weights))
    return weighted_variance

# Example usage
means = np.array([4.9, 4.9, 5.8])
std_devs = np.array([0.8, 0.8, 0.7])

# Compute weighted mean and variance
best_estimate_mean = weighted_mean(means, std_devs)
best_estimate_variance = weighted_variance(std_devs)

print(f"Best Estimate Mean: {best_estimate_mean}")
print(f"Best Estimate Variance: {best_estimate_variance}")
