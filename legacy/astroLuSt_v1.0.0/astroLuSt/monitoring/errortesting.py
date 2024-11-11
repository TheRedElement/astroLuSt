
"""
    - functions for testing `astroLuSt.monitoring.errorlogging`
"""

import numpy as np

def raise_div0():
    return 1/0

def raise_concaterr(n:int):
    return np.concatenate(n, 0)

def raise_nested_exception():
    
    try:
        raise ValueError('custom error')
    except Exception as e:
        # raise ValueError('custom error in except')
        try: 
            raise ValueError('custom error in except')
        except:
            raise ValueError('custom error in except in except')