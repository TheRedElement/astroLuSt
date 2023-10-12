
#%%imports
import numpy as np

from typing import Union, Tuple, Callable

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
            - `M`
                - np.ndarray
                - absolute magnitude in some passband
            - `CI`
                - np.ndarray
                - color index
            - `R`
                - np.ndarray, optional
                - reddening factor
                - the default is `None`
            - `A_M`
                - np.ndarray, optional
                - interstellar extinction in the same passband as passed to `M`
                - the default is `None`
            - `E_CI`
                - np.ndarray, optional
                - color excess in same color as passed to `CI`
                - the default is `None`

        Raises
        ------
            - `ValueError`
                - if `R` and at least one of `A_M` and `E_CI` are `None` 

        Returns
        -------
            - `w`
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

def mags2fluxes(
    m:Union[np.ndarray,float],
    m_ref:Union[np.ndarray,float],
    f_ref:Union[np.ndarray,float]=1,
    ) -> Union[np.ndarray,float]:
    """
        - function to convert magnitudes to flux
        
        Parameters
        ----------
            - `m`
                - float, np.ndarray
                - magnitudes to be converted
            - `m_ref`
                - float, np.ndarray
                - reference magnitude for the conversion
                    - this value is dependent on the passband in use
            - `f_ref`
                - float, np.ndarray, optional
                - reference flux for the conversion
                    - corresponding to `m_ref`
                    - this value is dependent on the passband in use
                - the default is 1
                    - will return the fraction `f/f_ref`

        Raises
        ------

        Returns
        -------
            - `f`
                - float, np.array
                - flux corresponding to `m`

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------

    """
    f = 10**(-0.4*(m - m_ref)) * f_ref
    return f

def fluxes2mags(
    f:Union[np.ndarray,float],
    f_ref:Union[np.ndarray,float],
    m_ref:Union[np.ndarray,float]=0,
    ) -> Union[np.ndarray,float]:
    """
        - function to convert photon flux to magnitudes

        Parameters
        ----------
            - `f`
                - float, np.array
                - fluxes to be converted
            - `f_ref`
                - float, np.ndarray
                - reference flux for the conversion
                    - this value is dependent on the passband in use
            - `m_ref`
                - float, np.ndarray, optional
                - reference magnitude for the conversion
                    - corresponding to `f_ref`
                    - this value is dependent on the passband in use
                - the default is 0
                    - will return the difference `m - m_ref`

        Raises
        ------

        Returns
        -------
            - `m`
                - float, np.array
                - magnitudes corresponding to `f`

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------
    """
    m = -2.5*np.log10(f/f_ref) + m_ref

    return m

def add_mags(m:np.ndarray, w:np.ndarray=None):
    """
        - function to calculate the total magnitude of a set of magnitudes

        Parameters
        ----------
            - `m`
                - np.ndarray
                - contains magnitudes to add up
            - `w`
                - np.ndarray, optional
                - weight for each passed magnitude
                    - for example some distance measure
                - the default is `None`
                    - will be set to 1 for all elements in `m`

        Raises
        ------

        Returns
        -------
            - `m_tot`
                - float
                - combined (weighted) magnitude

        Comments
        --------

    """

    if w is None: w = np.ones_like(m)

    if m.shape != w.shape:  
        raise ValueError(
            f'`m` and `w` have to have same shape but are of shapes {m.shape}, {w.shape}'
        )

    m_exp = 10**(-0.4*m)
    m_tot = -2.5*np.log10(w.T@m_exp)

    return m_tot

def contamination(m:np.ndarray, m_cont:np.ndarray):
    
    ffrac = mags2fluxes(mag=m_cont, m_ref=m_cont, f_ref=0)

    return

m = np.array([1,1,1,1,1])
w = None #np.array([1,1])

m_tot = add_mags(m, w=w)
print(m_tot)
