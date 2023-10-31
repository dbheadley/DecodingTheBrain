import numpy as np

# create zscore function with dimension parameter
def zscore(x, dim):
    return (x - np.mean(x, axis=dim, keepdims=True))/np.std(x, axis=dim, keepdims=True)