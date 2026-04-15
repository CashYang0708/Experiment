import numpy as np
from scipy import stats

def mean_absolute_error_fitness(predictions, returns):
                return -np.mean(np.abs(predictions - returns))
            
def mse_fitness(predictions, returns):
    return -np.mean((predictions - returns) ** 2)
            
def rmse_fitness(predictions, returns):
        return -np.sqrt(np.mean((predictions - returns) ** 2))
            
def pearson_fitness(predictions, returns):
    if np.std(predictions) == 0:
        return -np.inf
    correlation = np.corrcoef(predictions, returns)[0, 1]
    if np.isnan(correlation):
        return -np.inf
    return correlation
            
def spearman_fitness(predictions, returns):
    if np.std(predictions) == 0:
        return -np.inf
    correlation, _ = stats.spearmanr(predictions, returns)
    if np.isnan(correlation):
        return -np.inf
    return correlation