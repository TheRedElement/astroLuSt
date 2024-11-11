
#%%imports
import numpy as np
from typing import Tuple

#%%definitions

def sort_into_batches(
    X:np.ndarray, *y:np.ndarray
    ) -> Tuple[np.ndarray,...]:
    """
        - function that takes a nested array `X` and creates a new array
            - this array contains as entries arrays that contain all arrays of `X` which have the same length
            - if labels (`y`) exist, those will be sorted accordingly

        Parameters
        ----------
            - `X`
                - `np.ndarray`
                - nested `np.ndarray`
                    - can contain arrays of different lenghts
                - this array will be split up into arrays of similar length
            - `*y`
                - `np.ndarray`, optional
                - have to have same first axis shape as `X`
                - labels corresponding to `X`
                - will be split such that the same entries in `X` are still paired up with the same labels
                - the default is `None`
                    - will be ignored

        Raises
        ------

        Returns
        -------
            - `X_batches`
                - `np.ndarray`
                - has `dtype` `object`
                - contains subarrays
                    - each subarray contains samples of the same length
                - i.e. the output shape will be (`nlenghts`, `nperlength`, `length`, `nfeatures`)
                    - `nlenghts` is the number of unique array-lengths present in `X`
                    - `nperlength` is the number of arrays that have the same length (`length`)
                    - `length` is the common length of a batch of arrays
                    - `nfeatures` is the number of features present in the arrays of a batch
            - `*y_batches`
                - `np.ndarray`
                - has `dtype` `object`
                - contains subarrays
                - split of all arrays contained within `y` such that they are still comparabel with `X`
                - will return as many individual `y_batches`, as elements in `y`

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------
            - to only transform `X` alone call
                
                ```python
                >>> X_batches, = sort_into_batches(X)
                ```
                
            - to transform a multitude of arrays (n in the example) w.r.t. `X` use
                
                ```python
                >>> X_batches, y0_, y1_, ..., yn_ = sort_into_batches(X, y0, y1, ..., yn)
                ```
                
    """

    lengths = [len(x) for x in X]
    
    X_batches = []
    y_batches = []
    for idx, l in enumerate(np.unique(lengths)):
        X_batches.append(X[(lengths==l)])
        y_batches_l = []    #y_batches for length `l`
        for yidx, yi in enumerate(y):
            y_batches_l.append(yi[(lengths==l)])
        y_batches.append(y_batches_l)   #add to list of `y_batches`

    X_batches = np.array(X_batches, dtype=object)
    y_batches = np.array(y_batches, dtype=object).T

    return X_batches, *y_batches