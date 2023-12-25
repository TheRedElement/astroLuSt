
import numpy as np

def raise_div0():
    return 1/0

def raise_concaterr(n:int):
    return np.concatenate(n, 0)
