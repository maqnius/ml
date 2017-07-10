import numpy as np

def kernel_lin(x, y):
    return x.dot(y)

def kernel_rbf(x ,y, gamma=1.0):
    d = x - y
    return np.exp(-np.dot(d, d) * gamma)