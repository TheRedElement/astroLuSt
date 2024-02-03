
#%%imports
import numpy as np

from typing import Union, Tuple, Callable

from astroLuSt.monitoring import errorlogging as alme
from astroLuSt.monitoring import formatting as almf

#%%classes
class DistanceModule:
    """
        - class implementing the distance module
        - if you want to update the parameters simply update the attributes and run the corresponding method again
        
        Attributes
        ----------
            - `m`
                - `float, np.ndarray`, optional
                - apparent magnitude(s)
                - will infer `M` if both `M` and `m` are not `None`
                - the default is `None`
                    - will be infered from `M`
            - `M`
                - `float, np.ndarray`, optional
                - absolute magnitude(s)
                - will infer `M` if both `M` and `m` are not `None`
                - the default is `None`
                    - will be infered from `m`
            - `d`
                - `float, np.ndarray`, optional
                - distance(s) in parsec [pc]
                - will use `d` if both `d` and `plx` are not `None`
                - the default is `None`
                    - will try to infer distances via `plx`
            - `plx`
                - `float, np.ndarray`, optional
                - parallax(es) in arcseconds [arcsec]
                - will use `d` if both `d` and `plx` are not `None`
                - the default is `None`
                    - will try to infer parallaces via `d`
            - `dm`
                - `float, np.ndarray`, optional
                - uncertainty of `dm`
                - the default is 0
            - `dM`
                - `float, np.ndarray`, optional
                - uncertainty of `dM`
                - the default is 0
            - `dd`
                - `float, np.ndarray`, optional
                - uncertainty of `dd`
                - the default is 0
            - `dplx`
                - `float, np.ndarray`, optional
                - uncertainty of `dplx`
                - the default is 0

        Methods
        -------
            - `infer_d_plx()`
            - `infer_m_M()`
            - `__absmag()`
            - `__appmag()`

        Dependencies
        ------------
            - numpy
            - typing
        
        Comments
        --------

    """
    
    def __init__(self,
        m:Union[float,np.ndarray]=None, M:Union[float,np.ndarray]=None,
        d:Union[float,np.ndarray]=None, plx:Union[float,np.ndarray]=None,
        dm:Union[float,np.ndarray]=0, dM:Union[float,np.ndarray]=0,
        dd:Union[float,np.ndarray]=0, dplx:Union[float,np.ndarray]=0,
        ) -> None:


        self.m      = m
        self.M      = M
        self.d      = d
        self.plx    = plx

        #uncertainties
        self.dm      = dm
        self.dM      = dM
        self.dd      = dd
        self.dplx    = dplx

        #calculate parameters
        self.infer_d_plx()
        self.infer_m_M()


        pass

    def __repr__(self) -> str:
        return (
            f'DistanceModule(\n'
            f'    m={repr(self.m)}, M={repr(self.M)},\n'
            f'    d={repr(self.d)}, plx={repr(self.plx)},\n'
            f'    dm={repr(self.dm)}, dM={repr(self.dM)},\n'
            f'    dd={repr(self.dd)}, dplx={repr(self.dplx)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def infer_d_plx(self,
        ) -> None:
        """
            - method to infer distance `d` and parallax `plx` from the respective other quantity

            Parameters
            ----------

            Raises
            ------
                - `ValueError`
                    - if both `d` and `plx` are `None`
            
            Returns
            -------

            Comments
            --------
        """

        #raise error
        if self.d is None and self.plx is None:
            raise ValueError(f'One of `d` and `plx` has to be not `None` but they have the values {self.d} and {self.plx}!')
        #infer d from plx
        elif self.d is None and self.plx is not None:
            self.d = 1/self.plx
            self.dd = self.dplx * np.abs(-1/self.plx**2)
        #infer plx from d
        elif self.d is not None and self.plx is None:
            self.plx = 1/self.d
            self.dplx = self.dd * np.abs(-1/self.d**2)
        #use d if both provided
        elif self.d is not None and self.plx is not None:
            self.plx = 1/self.d
            self.dplx = self.dd * np.abs(-1/self.d**2)


        return

    def infer_m_M(self,
        ) -> None:
        """
            - method to infer apparent and absolute magnitude (`m` and `M`) from the respective other quantity

            Parameters
            ----------

            Raises
            ------
                - `ValueError`
                    - if both `m` and `M` are `None`
            
            Returns
            -------

            Comments
            --------        
        """


        if self.m is None and self.M is None:
            raise ValueError(f'At least on of `self.m` and `self.M` has to be not `None` but they have the values {self.m} and {self.M}!')
        elif self.m is None:
            self.__appmag()
        elif self.M is None:
            self.__absmag()
        else:
            self.__absmag()
        return

    def __absmag(self,
        ):
        """
            - method to convert apparent magnitudes to absolute magnitudes via the distance modulus

            Parameters
            ----------

            Raises
            ------
            
            Returns
            -------

            Dependencies
            ------------
                - numpy
                - typing
            
            Comments
            --------
        """

        self.M  = self.m - 5*np.log10(self.d/10)
        self.dM = self.dm \
            + self.dd * np.abs(-5/(np.log(10)*self.d))

        return 

    def __appmag(self,
        ):
        """
            - function to convert absolute magnitudes to apparent magnitudes via the distance modulus

            Parameters
            ----------

            Raises
            ------
            
            Returns
            -------

            Dependencies
            ------------
                - numpy
                - typing
            
            Comments
            --------
        """

        self.m = 5*np.log10(self.d/10) + self.M
        self.dm = self.dM \
            + self.dd * np.abs(5/(np.log(10)*self.d))

        return

#%%definitions

def mags2fluxes(
    m:Union[np.ndarray,float],
    m_ref:Union[np.ndarray,float],
    f_ref:Union[np.ndarray,float]=1,
    dm:Union[np.ndarray,float]=0,
    dm_ref:Union[np.ndarray,float]=0,
    df_ref:Union[np.ndarray,float]=0,
    ) -> Tuple[Union[np.ndarray,float],Union[np.ndarray,float]]:
    """
        - function to convert magnitudes to flux
        
        Parameters
        ----------
            - `m`
                - `float`, `np.ndarray`
                - magnitudes to be converted
            - `m_ref`
                - `float`, `np.ndarray`
                - reference magnitude for the conversion
                    - this value is dependent on the passband in use
            - `f_ref`
                - `float`, `np.ndarray`, optional
                - reference flux for the conversion
                    - corresponding to `m_ref`
                    - this value is dependent on the passband in use
                - the default is 1
                    - will return the fraction `f/f_ref`
            - `dm`
                - `float`, `np.ndarray`, optional
                - uncertainty of `dm`
                - the default is 0
            - `dm_ref`
                - `float`, `np.ndarray`, optional
                - uncertainty of `dm_ref`
                - the default is 0
            - `dm_ref`
                - `float`, `np.ndarray`, optional
                - uncertainty of `dm_ref`
                - the default is 0


        Raises
        ------

        Returns
        -------
            - `f`
                - `float`, `np.array`
                - flux corresponding to `m`
            - `df`
                - `float`, `np.array`
                - uncertainty of `f`

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------

    """
    f = 10**(-0.4*(m - m_ref)) * f_ref
    
    #uncertainty
    df =  dm     * np.abs(f*(-0.4*np.log(10))) \
        + dm_ref * np.abs(f*( 0.4*np.log(10))) \
        + df_ref * np.abs(10**(-0.4*(m - m_ref)))

    return f, df

def fluxes2mags(
    f:Union[np.ndarray,float],
    f_ref:Union[np.ndarray,float],
    m_ref:Union[np.ndarray,float]=0,
    df:Union[np.ndarray,float]=0,
    df_ref:Union[np.ndarray,float]=0,
    dm_ref:Union[np.ndarray,float]=0,
    ) -> Tuple[Union[np.ndarray,float],Union[np.ndarray,float]]:
    """
        - function to convert photon flux to magnitudes

        Parameters
        ----------
            - `f`
                - `float`, `np.array`
                - fluxes to be converted
            - `f_ref`
                - `float`, `np.ndarray`
                - reference flux for the conversion
                    - this value is dependent on the passband in use
            - `m_ref`
                - `float`, `np.ndarray`, optional
                - reference magnitude for the conversion
                    - corresponding to `f_ref`
                    - this value is dependent on the passband in use
                - the default is 0
                    - will return the difference `m - m_ref`
            - `df`
                - `float`, `np.ndarray`, optional
                - uncertainty of `df`
                - the default is 0
            - `df_ref`
                - `float`, `np.ndarray`, optional
                - uncertainty of `df_ref`
                - the default is 0
            - `dm_ref`
                - `float`, `np.ndarray`, optional
                - uncertainty of `dm_ref`
                - the default is 0

        Raises
        ------

        Returns
        -------
            - `m`
                - `float`, `np.array`
                - magnitudes corresponding to `f`
            - `dm`
                - `float`, `np.array`
                - uncertainty of `m`

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------
    """
    m = -2.5*np.log10(f/f_ref) + m_ref

    #uncertainty
    dm =  df     *np.abs(-2.5*1/(np.log(10)*f)) \
        + df_ref *np.abs( 2.5*1/(np.log(10)*f_ref)) \
        + dm_ref *np.abs(1)

    return m, dm

def wesenheit_magnitude(
    M:Union[float,np.ndarray], CI:Union[float,np.ndarray],
    R:Union[float,np.ndarray]=None,
    A_M:Union[float,np.ndarray]=None, E_CI:Union[float,np.ndarray]=None,
    dM:Union[float,np.ndarray]=0, dCI:Union[float,np.ndarray]=0,
    dR:Union[float,np.ndarray]=0,
    dA_M:Union[float,np.ndarray]=0, dE_CI:Union[float,np.ndarray]=0,
    ) -> Tuple[Union[float,np.ndarray],Union[float,np.ndarray]]:
    """
        - function to calculate the wesenheit magnitude for a given set of input parameters

        Parameters
        ----------
            - `M`
                - `np.ndarray`, `float`
                - absolute magnitude in some passband
            - `CI`
                - `np.ndarray`, `float`
                - color index
            - `R`
                - `np.ndarray`, `float`, optional
                - reddening factor
                - the default is `None`
            - `A_M`
                - `np.ndarray`, `float`, optional
                - interstellar extinction in the same passband as passed to `M`
                - the default is `None`
            - `E_CI`
                - `np.ndarray`, `float`, optional
                - color excess in same color as passed to `CI`
                - the default is `None`
            - `dM`
                - `np.ndarray`, `float`, optional
                - uncertainty of `M`
                - the default is 0
            - `dCI`
                - `np.ndarray`, `float`, optional
                - uncertainty of `CI`
                - the default is 0
            - `dR`
                - `np.ndarray`, `float`, optional
                - uncertainty of `R`
                - the default is 0
            - `dA_M`
                - `np.ndarray`, `float`, optional
                - uncertainty of `A_M`
                - the default is 0
            - `dE_CI`
                - `np.ndarray`, `float`, optional
                - uncertainty of `E_CI`
                - the default is 0

        Raises
        ------
            - `ValueError`
                - if `R` and at least one of `A_M` and `E_CI` are `None` 

        Returns
        -------
            - `w`
                - `np.ndarray`, `float`
                - wesenheit magnitude
            - `dw`
                - `np.ndarra`, `float`
                - uncertainty of `w`

        Dependencies
        ------------
            - numpy

        Comments
        --------
    """

    if R is None and A_M is not None and E_CI is not None:
        R = A_M/E_CI
        dR = dA_M*np.abs(1/dE_CI) + dE_CI * np.abs(A_M/E_CI**2)
    elif R is not None:
        R = R
        dR = dR
    else:
        raise ValueError('Either "R" or both "A_M" and E_CI" have to be not None')


    w = M - R*CI
    
    #uncertainty
    dw = dM + dR*np.abs(CI) + dCI*np.abs(R)

    return w, dw

def mags_sum(
    m:np.ndarray, w:np.ndarray=None,
    dm:np.ndarray=0,
    axis:int=None,
    ) -> Tuple[Union[float,np.ndarray],Union[float,np.ndarray]]:
    """
        - function to calculate the total magnitude of a set of magnitudes

        Parameters
        ----------
            - `m`
                - `np.ndarray`
                - 3d array of shape `(nframes,xpix,ypix)`
                    - nframes denotes the number of frames passed
                    - xpix is the number of pixels in x direction
                    - ypix is the number of pixels in y direction
                - contains magnitudes to add up
            - `w`
                - `np.ndarray`, optional
                - weight for each passed pixel
                    - for example some distance measure
                - has to be of shape `(1,*m.shape[1:])`
                - the default is `None`
                    - will be set to 1 for all elements in `m`
            - `dm`
                - `np.ndarray`, optional
                - uncertainties of `m`
                - the default is 0
            - `axis`
                - `int`, optional
                - axis along which to add up magnitudes
                    - 0     ... pixel wise
                    - 1     ... row wise
                    - 2     ... column wise
                    - (1,2) ... frame wise
                - the default is `None`
                    - will flatten before adding up magnitudes

        Raises
        ------

        Returns
        -------
            - `m_tot`
                - `float`
                - combined (weighted) magnitude
            - `dm_tot`
                - `float`
                - uncertainty of `m_tot`
        
        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------

    """

    if w is None: w = np.ones(m.shape[1:])

    m_exp = 10**(-0.4*m)
    m_sum = np.sum(w*m_exp, axis=axis)
    m_tot = -2.5*np.log10(m_sum)

    #uncertainty
    dm_sum = np.sum(dm * np.abs(w* 10**(-0.4*m)  *(-0.4*np.log(10))), axis=axis)
    dm_tot = dm_sum * np.abs(-2.5*1/(np.log(10)*m_sum))

    return m_tot, dm_tot

def mags_contribution(
    m:Union[float,np.ndarray], m_cont:Union[float,np.ndarray],
    w:np.ndarray=None,
    dm:Union[float,np.ndarray]=None, dm_cont:Union[float,np.ndarray]=None,
    ) -> Tuple[Union[float,np.ndarray],Union[float,np.ndarray]]:
    """
        - function that estimates the contribution in magnitude of target star (m) to a total magnitude
        
        Parameters
        ----------
            - `m`
                - `float`, `np.ndarray`
                - magnitude(s) of the target star(s)
                - if `float`
                    - will calculate and return the contribution of that one target star
                - if `np.ndarray`
                    - will calculate contribution of every single star to the `m_cont` contaminant stars
            - `m_cont`
                - `float`, `np.ndarray`
                - if `float`
                    - total magnitude to compare `m` to
                - if `np.ndarray`
                    - magnitudes to calculate total magnitude from on the fly
                        - will call `mags_sum(m_cont, w=w)`
                    - i.e. magnitudes of all contaminant stars
            - `w`
                - `np.ndarray`, optional
                - weights for each passed magnitude
                    - only applied if `m_cont` is a `np.ndarray`
                    - for example some distance measure
                - the default is `None`
                    - will be set to 1 for all elements in `m`
            - `dm`
                - `float`, `np.ndarray`, optional
                - uncertainty of `m`
                - the default is `None`
                    - will be set to 0 for all entries in `m` (infinite accuracy)
            - `dm_cont`
                - `float`, `np.ndarray`, optional
                - uncertainty of `m_cont`
                - the default is `None`
                    - will be set to 0 for all entries in `m_cont` (infinite accuracy)

        Raises
        ------
            - `ValueError`
                - if quantities and uncertainties don't have the same shape

        Returns
        -------
            - `p`
                - `float`, `np.ndarray`
                - fractional contribution of `m` to `m_cont`
                - `float` if `m` is `float`
                - `np.ndarray` if `m` is a `np.ndarray`
            - `dp`
                - `float`, `np.ndarray`
                - uncertainty of `p`
                - `float` if `m` is `float`
                - `np.ndarray` if `m` is a `np.ndarray`

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------
    """

    m_in = m.copy()

    #default values; convert to np.ndarray
    m       = np.array([m]).flatten()
    m_cont  = np.array([m_cont]).flatten()
    if dm is None:      dm      = np.zeros_like(m)
    else:               dm      = np.array([dm]).flatten()
    if dm_cont is None: dm_cont = np.zeros_like(m_cont)
    else:               dm_cont = np.array([dm_cont]).flatten()

    #check shapes
    if m.shape      != dm.shape:        raise ValueError(f'`m` and `dm` have to have the same shape but have `{m.shape=}` and {dm.shape=}!')
    if m_cont.shape != dm_cont.shape:   raise ValueError(f'`m_cont` and `dm_cont` have to have the same shape but have `{m_cont.shape=}` and {dm_cont.shape=}!')

    #handle arrays of len() == 0
    if len(m_cont) == 0:
        m_cont  = np.append(m_cont, [np.inf])   #set m_cont = inf i.e., infinitely faint object
        dm_cont = np.append(dm_cont, [0])       #set dm_cont = 0 i.e., infinite accuracy

    #calculate total contaminant magnitude if array of magnitudes is provided
    if len(m_cont) > 1:
        m_cont, dm_cont = mags_sum(m=m_cont, w=w, dm=dm_cont)

    ffrac, dffrac = mags2fluxes(m=m_cont, m_ref=m, dm=dm_cont, dm_ref=dm)
    p = 1/(1 + ffrac)
    
    #uncertainty
    dp = dffrac * np.abs(-1/(1+ffrac)**2)

    #convert to float if input was float
    if isinstance(m_in, float):
        p = float(p)
        dp = float(dp)

    return p, dp

def flux_contribution(
    f:Union[float,np.ndarray], f_cont:Union[float,np.ndarray],
    w:np.ndarray=None,
    df:Union[float,np.ndarray]=None, df_cont:Union[float,np.ndarray]=None,
    ) -> Tuple[Union[float,np.ndarray],Union[float,np.ndarray]]:
    """
        - function that estimates the contribution in flux of target star (`f`) to a total flux
        
        Parameters
        ----------
            - `f`
                - `float`, `np.ndarray`
                - flux(s) of the target star(s)
                - if `float`
                    - will calculate and return the contribution of that one target star
                - if `np.ndarray`
                    - will calculate contribution of every single star to the `m_cont` contaminant stars
            - `f_cont`
                - `float`, `np.ndarray`
                - if `float`
                    - total flux to compare `f` to
                - if `np.ndarray`
                    - fluxes to calculate total flux from on the fly
                        - will call `np.sum(w*f_cont)`
                    - i.e. flux of all contaminant stars
            - `w`
                - `np.ndarray`, optional
                - weights for each passed flux
                    - only applied if `f_cont` is a `np.ndarray`
                    - for example some distance measure
                - the default is `None`
                    - will be set to 1 for all elements in `f`
            - `df`
                - `float`, `np.ndarray`, optional
                - uncertainty of `f`
                - the default is `None`
                    - will be set to 0 for all entries in `f` (infinite accuracy)                
            - `df_cont`
                - `float`, `np.ndarray`, optional
                - uncertainty of `f_cont`
                - the default is `None`
                    - will be set to 0 for all entries in `f_cont` (infinite accuracy)                

        Raises
        ------
            - `ValueError`
                - if quantities and uncertainties don't have the same shape

        Returns
        -------
            - `p`
                - `float`, `np.ndarray`
                - fractional contribution of `f` to `f_cont`
                - `float` if `f` is `float`
                - `np.ndarray` if `f` is a `np.ndarray`
            - `dp`
                - `float`, `np.ndarray`
                - uncertainty of `p`
                - `float` if `f` is `float`
                - `np.ndarray` if `f` is a `np.ndarray`

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------
    """
    
    f_in = f.copy()

    #default values; convert to np.ndarray
    f       = np.array([f]).flatten()
    f_cont  = np.array([f_cont]).flatten()
    if w is None: w = 1
    if df is None:      df      = np.zeros_like(f)
    else:               df      = np.array([df]).flatten()
    if df_cont is None: df_cont = np.zeros_like(f_cont)
    else:               df_cont = np.array([df_cont]).flatten()
    
    #check shapes
    if f.shape      != df.shape:        raise ValueError(f'`f` and `df` have to have the same shape but have `{f.shape=}` and {df.shape=}!')
    if f_cont.shape != df_cont.shape:   raise ValueError(f'`f_cont` and `df_cont` have to have the same shape but have `{f_cont.shape=}` and {df_cont.shape=}!')

    #convert to np.ndarray
    f_cont = np.array([f_cont]).flatten()


    #handle arrays of len() == 0
    if len(f_cont) == 0:
        f_cont  = np.append(f_cont, [0])    #set f_cont = 0 i.e., infinitely faint object
        df_cont = np.append(df_cont, [0])   #set df_cont = 0 i.e., infinite accuracy

    #calculate total contaminant flux if array of fluxes is provided
    if len(f_cont) > 1:
        f_cont = np.sum(w*f_cont)
        df_cont = np.sum(df_cont*np.abs(w))

    #contribution factor
    p = 1/(1 + f_cont/f)

    #uncertainty
    dp =  df      * np.abs((f_cont/(f+f_cont)**2)) \
        + df_cont * np.abs(-f/(f+f_cont)**2)


    if isinstance(f_in, float):
        p = float(p)
        dp = float(dp)

    return p, dp
