
#%%imports
import numpy as np


#%%definitions

def wesenheit_magnitude(
    M:np.ndarray, CI:np.ndarray,
    R:np.ndarray=None,
    A_M:np.ndarray=None, E_CI:np.ndarray=None 
    ) -> np.ndarray:
    """
        - function to calculate the wesenheit magnitude for a given set of input parameters

        Parameters
        ----------
            - M
                - np.ndarray
                - absolute magnitude in some passband
            - CI
                - np.ndarray
                - color index
            - R
                - np.ndarray, optional
                - reddening factor
                - the default is None
            - A_M
                - np.ndarray, optional
                - interstellar extinction in the same passband as passed to 'M'
                - the default is None
            - E_CI
                - np.ndarray, optional
                - color excess in same color as passed to CI
                - the default is None

        Raises
        ------
            - ValueError
                - if 'R' and at least one of 'A_M' and 'E_CI' are None 

        Returns
        -------
            - w
                - np.ndarray
                - wesenheit magnitude

        Dependencies
        ------------
            - numpy

        Comments
        --------
    """

    if R is None and A_M is not None and E_CI is not None:
        R = A_M/E_CI
    elif R is not None:
        R = R
    else:
        raise ValueError('Either "R" or both "A_M" and E_CI" have to be not None')

    


    w = M - R*CI

    return w
