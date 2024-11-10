
#TODO: add frame with uncertainties to TPF (use `dm_` and `df_`)
#TODO: `TPF.add_custom()`: Add other common trends
#TODO: TPF.add_noise()`: `add different noise parts (photon noise, dead pixels, hot pixels, ...)


#%%imports
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import scipy.stats as sps

from astroLuSt.physics import photometry as alpp
from astroLuSt.monitoring import formatting as almf

from typing import Union, Tuple, Callable, Literal, List, Dict

#%%classes
class TPF:
    """
        - class to generate a simulated Target Pixel File (TPF)
        - assumes stars to have a gaussian Point Spread Function (PSF)

        Attributes
        ----------
            - `size`
                - `tuple`, `int`, optional
                - size of the TPF to generate
                - if `int`
                    - will be interpreted as `(size,size)`
                    - i.e., square frame with `size` pixels in x and y direction
                - if `tuple`
                    - `size[0]` denotes the number of pixels in x direction
                    - `size[1]` denotes the number of pixels in y direction
                - the default is `15`
            - `mode`
                - `Literal["flux","mag"]`, optional
                - whether to generate the frame using magnitudes or fluxes
                - allowed values are
                    - `'flux'`
                        - will use fluxes to generate the frame
                    - `'mag'`
                        - will use magnitudes to generate the frame
                - the default is `None`
                    - will be set to `'flux'`
            - `f_ref`
                - `float`, optional
                - reference flux to use when converting fluxes to magnitudes
                - the default is `1`
            - `m_ref`
                - `float`, optional
                - reference magnitude to use when converting magnitudes to fluxes
                - the default is `0`
            - `store_stars`
                - `bool`, optional
                - wether to store an array of all generated stars including their apertures
                - if `False`
                    - will only store the final (composite) frame
                - the default is `True`
            - `rng`
                - `np.random.default_rng`, `int`, optional
                - if `int`
                    - random seed to use in the random number generator
                - if `np.random.default_rng` instance
                    - random number gnerator to use for random generation
                - the default is `None`
                    - will use `np.random.default_rng(seed=None)`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Infered Attributes
        ------------------
            - `frame`
                - `np.ndarray`
                - final frame when all contributions are added up
                - contains values in flux
                - has shape `(xpix, ypix, 3)`
                    - the first two entries of the last dimension contain the pixel coordinates
                    - the last entry contains the flux value
            - `frame_mag`
                - `np.ndarray`
                - final frame when all contributions are added up
                - contains values in magnitudes (infered from `frame`)
                - has shape `(xpix, ypix, 3)`
                    - the first two entries of the last dimension contain the pixel coordinates
                    - the last entry contains the flux value
            - `starparams`
                - `list`
                - contains as many entries as stars in the frame
                - each entry is a list that contains the star specifications
                    - element 0: position in x direction (`posx`)
                    - element 1: position in y direction (`posy`)
                    - element 2: flux (`f`)
                    - element 3: magnitude (`m`)
                    - element 4: aperture (`aperture`)
            - `stars`
                - `np.ndarray`
                - only stored if `store_stars=True`
                - array of frames of each individual star including its aperture mask
                - has shape `(nstars,xpix,ypix,2)`
                    - first dimension denotes the star
                    - second dimension are the pixels in x direction
                    - third dimension are the pixels in y direction
                    - last dimension contains
                        - as element 0: flux values
                        - as element 1: magnitude values
                        - as element 2: aperture  mask

        Methods
        -------
            - `star()`
            - `add_stars()`
            - `add_custom()`
            - `add_noise()`
            - `aperture_from_mask()`
            - `get_frame()`
            - `rvs()`
            - `plot_result()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`
            - `scipy`

        Comments
        --------
            - all calculations are executed in fluxes
                - in parallel a second frame in magnitudes gets calculated by converting the flux result
    """

    def __init__(self,
        size:Union[Tuple,int]=15,
        mode:Literal['flux','mag']=None,
        f_ref:float=1, m_ref:float=0,
        store_stars:bool=True,
        rng:Union[int,np.random.default_rng]=None,
        verbose:int=0,
        ) -> None:

        if isinstance(size, int):   self.size   = (size,size)
        else:                       self.size   = size
        if mode is None:            self.mode   = 'flux'
        else:                       self.mode   = mode
        if rng is None:             self.rng    = np.random.default_rng(seed=None)
        elif isinstance(rng, int):  self.rng    = np.random.default_rng(seed=rng)
        else:                       self.rng    = rng
        self.verbose                            = verbose
        self.f_ref                              = f_ref
        self.m_ref                              = m_ref

        #frames
        x = np.arange(self.size[0])
        y = np.arange(self.size[1])
        xx, yy = np.meshgrid(x, y)

        #frame in flux
        self.frame = np.zeros(shape=(*self.size,1))
        self.frame = np.concatenate((np.expand_dims(xx,2), np.expand_dims(yy,2), self.frame), axis=2)
        
        #frame in mags
        self.frame_mag = self.frame.copy()
        self.frame_mag[:,:,2], dm_ = alpp.fluxes2mags(self.frame_mag[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)  #reset magnitude values
        #intermediate storage
        self.store_stars = store_stars

        #infered attributes
        self.stars = np.empty((0,self.size[0],self.size[1],3))
        self.starparams = np.empty((0,5))

        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    size={repr(self.size)},\n'
            f'    mode={repr(self.mode)},\n'
            f'    f_ref={repr(self.f_ref)}, m_ref={repr(self.m_ref)},\n'
            f'    store_stars={repr(self.store_stars)},\n'
            f'    rng={repr(self.rng.__class__.__name__)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def clean_frame(self,
        cleanparams:bool=True,
        ) -> None:
        """
            - function to clean the created frames to their initial configuration

            Parameters
            ----------
                - `cleanparams`
                    - `bool`, optional
                    - whether to also clean the parameters (`self.stars`, `self.starparams`)
                    - the default is `True`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        self.frame[:,:,2]       = 0
        self.frame_mag[:,:,2]   = 0
        if cleanparams:
            self.stars          = np.empty((0,self.size[0],self.size[1],3))
            self.starparams     = np.empty((0,5))
        
        return

    def star(self,
        pos:np.ndarray,
        f:float=None, m:float=None,
        aperture:float=1,
        ) -> np.ndarray:
        """
            - method to simulate an individual star on the CCD (in the TPF)
            - assumes gaussian PSF

            Parameters
            ----------
                - `pos`
                    - `np.ndarray`
                    - position of the star in pixels
                        - lower left corner has coordinates `(0,0)`
                - `f`
                    - `float`, optional
                    - flux of the star
                        - implemented as amplitude (scaling factor) to gaussian PSF
                    - used to determine `m` via flux-magnitude relation
                        - calls `astroLuSt.physics.photometry.fluxes2mags()`
                        - will use `self.m_ref` and `self.f_ref` for the conversion
                    - the default is `None`
                        - will be infered from `m`
                - `m`
                    - `float`, optional
                    - magnitude of the star
                    - used to determine `f` via flux-magnitude relation
                        - calls `astroLuSt.physics.photometry.mags2fluxes()`
                        - will use `self.m_ref` and `self.f_ref` for the conversion
                    - will be ignored if `f` is not `None`
                    - the default is `None`
                        - will be ignored
                - `aperture`
                    - `float`, opational
                    - "ground truth" aperture of the star
                        - defined as aperture = 2*FWHM = 2*2*std*sqrt(ln(2))
                        - implemented as proportionality factor to covariance
                    - the default is `1`

            Raises
            ------
                - `ValueError`
                    - if both `f` and `m` are `None`

            Returns
            -------
                - `star`
                    - `np.ndarray`
                    - has shape `(self.size[0],self.size[1],2)`
                        - entry 0 in last dimension contains the fluxes
                        - entry 1 in last dimension contains the aperture mask

            Comments
            --------
        """

        if f is None and m is not None:
            f, df_ =  alpp.mags2fluxes(m=m, m_ref=self.m_ref, f_ref=self.f_ref)
        elif f is not None and m is None:
            m, dm_ =  alpp.fluxes2mags(f=f, f_ref=self.f_ref, m_ref=self.m_ref)
        if f is not None and m is not None:
            m, dm_ =  alpp.fluxes2mags(f=f, f_ref=self.f_ref, m_ref=self.m_ref)
        else:
            raise ValueError(f'At least one of `f` and `m` has to be not `None` but they have values {f} and {m}.')

        cov = (aperture/(2*2*np.sqrt(np.log(2))))**2     #aperture = 2*halfwidth = 2* 2*std*sqrt(ln(2))
        star = f*sps.multivariate_normal(
            mean=pos, cov=cov, allow_singular=True
        ).pdf(self.frame[:,:,:2])

        #star in magnitudes
        star_mag, dm_ = alpp.fluxes2mags(star, f_ref=self.f_ref, m_ref=self.m_ref)
        
        #aperture mask
        aperture_mask = (np.sqrt(np.sum((self.frame[:,:,:2]-pos)**2, axis=2))<aperture)

        #add both to star
        star = np.concatenate(
            (
                np.expand_dims(star, axis=2),
                np.expand_dims(star_mag, axis=2),
                np.expand_dims(aperture_mask, axis=2)
            ),
            axis=2,
        )

        #store generated parameters and clean frames
        self.starparams = np.append(self.starparams, np.array([[pos[0], pos[1], f, m, aperture]]), axis=0)
        if self.store_stars:
            self.stars = np.append(self.stars, np.expand_dims(star,0), axis=0)

        return star

    def add_stars(self,
        nstars:int=1,
        posx:Union[dict,np.ndarray,float,int]=None,
        posy:Union[dict,np.ndarray,float,int]=None,
        f:Union[dict,np.ndarray,float,int]=None,
        m:Union[dict,np.ndarray,float,int]=None,
        aperture:Union[dict,np.ndarray,float,int]=None,
        verbose:int=None,
        ) -> None:
        """
            - method to add stars to the on the CCD (TPF)
            
            Parameters
            ----------
                - `nstars`
                    - `int`, optional
                    - number of stars to generate
                    - will be ignored if all other parameters are specified
                        - i.e., the stars will be generated according to specific parameters and not randomly
                    - the default is `1`
                - `posx`
                    - `dict`, `np.ndarray`, `float`, `int`, optional
                    - x values of the stars position in pixels
                    - if `dict`
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random positions
                            - `'params'`
                                - value is `dict` or `list`
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as `list` of positional args (for list)
                    - if `int`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - if `float`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':0,  'high':self.size[0]})`
                        - i.e. a uniform distribution in the whole range of the frame
                - `posy`
                    - `dict`, `np.ndarray`, `float`, `int`, optional
                    - y values of the stars position in pixels
                    - if `dict`
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a `string`
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random positions
                            - `'params'`
                                - value is `dict` or `list`
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as `list` of positional args (for `list`)
                    - if `int`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - if `float`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':0,  'high':self.size[1]})`
                        - i.e. a uniform distribution in the whole range of the frame
                - `f`
                    - `dict`, `np.ndarray`, `float`, `int`, optional
                    - fluxes of the stars to generate
                    - if `dict`
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random flux values
                            - `'params'`
                                - value is `dict` or `list`
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as `list` of positional args (for `list`)
                    - if `int`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - if `float`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':0.1,  'high':1})`
                        - i.e. a uniform distribution from 0.1 to 1
                - `m`
                    - `dict`, `np.ndarray`, `float`, `int`, optional
                    - magnitudes of the stars to generate
                    - if `dict`
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random magnitudes
                            - `'params'`
                                - value is `dict` or `list`
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for `dict`)
                                    - passed as `list` of positional args (for `list`)
                    - if `int`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - if `float`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':-2.5,  'high':0})`
                        - i.e. a uniform distribution from -2.5 to 0
                - `aperture`
                    - `dict`, `np.ndarray`, `float`, `int`, optional
                    - apertures to use for the stars 
                    - if `dict`
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random apertures
                            - `'params'`
                                - value is `dict` or `list`
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for `dict`)
                                    - passed as `list` of positional args (for `list`)
                    - if `int`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - if `float`
                        - will be interpreted as `list` of length 1 (i.e. one star)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':0.1, 'high':1})`
                        - i.e. a uniform distribution from 0.1 to 1
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`

            Raises
            ------
                - `ValueError`
                    - if to little information has been provided and `random` is set to `False`
                - `ValueError`
                    - if to many stars are requested compared to the resolution of the positions
                    - i.e. `random['nstars']` <= `random['posx_res']*random['posy_res']`
                    - in that case a random sampling without return is not possible!

            Returns
            -------

            Comments
            --------
        """

        #default parameters
        if posx is None:        posx    =dict(dist='uniform', params={'low':0,  'high':self.size[0]})
        if posy is None:        posy    =dict(dist='uniform', params={'low':0,  'high':self.size[1]})
        if f is None:           f       =dict(dist='uniform', params={'low':0.1,'high':1})
        if m is None:           m       =dict(dist='uniform', params={'low':0,  'high':2.5})
        if aperture is None:    aperture=dict(dist='uniform', params={'low':0.1,'high':1})

        #for single value
        if isinstance(posx, (int,float)):     posx     = np.array([posx])
        if isinstance(posy, (int,float)):     posy     = np.array([posy])
        if isinstance(f, (int,float)):        f        = np.array([f])
        if isinstance(m, (int,float)):        m        = np.array([m])
        if isinstance(aperture, (int,float)): aperture = np.array([aperture])

        #generate positions
        ##for randomly generated
        if isinstance(posx, dict):
            if isinstance(posx['params'], dict):        
                posx['params']['size'] = (nstars,1)
                posx = eval(f"self.rng.{posx['dist']}(**{posx['params']})")
            else:
                posx['params'].append((nstars,1))
                posx = eval(f"self.rng.{posx['dist']}(*{posx['params']})")
        if isinstance(posy, dict):
            if isinstance(posy['params'], dict):        
                posy['params']['size'] = (nstars,1)
                posy = eval(f"self.rng.{posy['dist']}(**{posy['params']})")
            else:
                posy['params'].append((nstars,1))
                posy = eval(f"self.rng.{posy['dist']}(*{posy['params']})")

        posx = np.array(posx).reshape(-1,1)
        posy = np.array(posy).reshape(-1,1)
        pos = np.concatenate((posx,posy), axis=1)

        #generate magnitudes/fluxes
        if self.mode == 'flux':
            ##for randomly generated
            if isinstance(f, dict):
                if isinstance(f['params'], dict):        
                    f['params']['size'] = (nstars)
                    f = eval(f"self.rng.{f['dist']}(**{f['params']})")
                else:
                    f['params'].append((nstars))
                    f = eval(f"self.rng.{f['dist']}(*{f['params']})")
            f = np.array(f)
        elif self.mode == 'mag':
            ##for randomly generated
            if isinstance(m, dict):
                if isinstance(m['params'], dict):        
                    m['params']['size'] = (nstars)
                    m = eval(f"self.rng.{m['dist']}(**{m['params']})")
                else:
                    m['params'].append((nstars))
                    m = eval(f"self.rng.{m['dist']}(*{m['params']})")
            m = np.array(m)
            f, df_ = alpp.mags2fluxes(m=m, m_ref=self.m_ref, f_ref=self.f_ref)

        #generate apertures
        ##for randomly generated
        if isinstance(aperture, dict):
            if isinstance(aperture['params'], dict):        
                aperture['params']['size'] = (nstars)
                aperture = eval(f"self.rng.{aperture['dist']}(**{aperture['params']})")
            else:
                aperture['params'].append((nstars))
                aperture = eval(f"self.rng.{aperture['dist']}(*{aperture['params']})")
        aperture = np.array(aperture)

        #add stars to TPF
        for posi, fi, api in zip(pos, f, aperture):
            star = self.star(pos=posi, f=fi, aperture=api)

            self.frame[:,:,2] += star[:,:,0]

        #convert to magnitudes            
        if self.mode == 'mag':
            self.frame_mag[:,:,2], dm_ = alpp.fluxes2mags(self.frame[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)

        return
    
    def add_custom(self,
        trend:Union[np.ndarray,Literal['lineary','linearx']],
        amplitude:float=1,
        ) -> None:
        """
            - method to add custom trends to the final image
            
            Parameters
            ----------
                - `trend`
                    - `np.ndarray`, `Literal["linearx","lineary"]`
                    - trend to add to `self.frame`
                    - if a `np.ndarray`
                        - will be added to `self.frame` via element-wise addition
                        - has to have same shape as `self.frame.shape[:2]`
                            - i.e. `self.size`
                    - options for `Literal`
                        - `lineary`
                            - linear trend in y direction
                            - generated via `np.linspace(0, np.ones((self.frame.shape[0])), self.frame.shape[1], axis=0)`
                        - `linearx`
                            - linear trend in x direction
                            - generated via `np.linspace(0, np.ones((self.frame.shape[0])), self.frame.shape[1], axis=1)`
                - `amplitude`
                    - `float`, optional
                    - scaling factor of the trend
                    - the default is `1`
                        - i.e. no scaling

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if isinstance(trend, str):
            if trend == 'lineary':
                trend = np.linspace(1,2*np.ones((self.frame.shape[0])), self.frame.shape[1], axis=0)
            elif trend == 'linearx':
                trend = np.linspace(1,2*np.ones((self.frame.shape[0])), self.frame.shape[1], axis=1)
            else:
                raise ValueError("`trend` has to be one of 'linearx', 'lineary'!")

        self.frame[:,:,2] += amplitude*trend

        #convert to magnitudes            
        if self.mode == 'mag':
            self.frame_mag[:,:,2], dm_ = alpp.fluxes2mags(self.frame[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)

        return

    def add_noise(self,
        amplitude:float=1, bias:float=1,
        ) -> None:
        """
            - method to add (gaussian) noise to the input frame

            Parameters
            ----------
                - `amplitude`
                    - `float`, optional
                    - amplitude of the added noise
                    - scaled such that `m=0` and `f=1` are likely to give good results for `aperture=1`,  `m_ref=0`, `f_ref=1`
                    - the default is `1`
                - `bias`
                    - `float`, optional
                    - offset of noise from 0 (similar to bias-current)
                    - measured in flux
                    - the default is `1`
                        - such that also for magnitudes are at least 0 for the default `m_ref` and `f_ref`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        noise = amplitude*1E-2*self.rng.standard_normal(size=(self.frame.shape[:2])) + bias
        
        self.frame[:,:,2] += noise

        #convert to magnitudes            
        if self.mode == 'mag':
            self.frame_mag[:,:,2], dm_ = alpp.fluxes2mags(self.frame[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)

        return
    
    def aperture_from_mask(self,
        aperture_mask:np.ndarray,
        ) -> float:
        """
            - method to estimate the aperture radius from a aperture mask

            Parameters
            ----------
                - `aperture_mask`
                    - `np.ndarray`
                    - aperture mask to use for the determination
                    - for this class stored in `TPF.stars[staridx,:,:,1]`
                        - i.e. second entry of last axis if `store_star` is set to `True`
                    - estimation executed via `sqrt(aperture_mask.sum()/pi)`
                        - solving `aperture = pi*r**2`
            
            Raises
            ------

            Returns
            -------
                - `aperture`
                    - `float`
                    - estimated aperture radius

            Comments
            --------
                - only (approximately) correct if the whole aperture is contained within the frame
        """

        aperture = np.sqrt(aperture_mask.sum()/np.pi)

        return aperture

    def get_frame(self,
        ) -> np.ndarray:
        """
            - method that returns the frame at its current state

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `self.frame`
                    - `np.ndarray`
                    - generated frame in fluxes
                    - returned if `self.mode=='flux'`
                - `self.frame_mag`
                    - `np.ndarray`
                    - generated frame in magnitudes
                    - returned if `self.mode=='mag'`
                    
            Comments
            --------
        """

        if self.mode == 'flux':
            return self.frame
        elif self.mode == 'mag':
            return self.frame_mag

    def rvs(self,
        shape:Union[int,tuple]=None,
        add_stars_kwargs:dict=None,
        add_noise_kwargs:dict=None,
        add_custom_kwargs:dict=None,        
        ):
        """
            - method similar to the `scipy.stats` `rvs()` method
            - rvs ... random variates
            - will generate a random frame
            
            Parameters
            ----------
                - `shape`
                    - `int`, `tuple`, optional
                    - NOT USED BECAUSE DEFINED BY `self.size`
                    - number of samples to generate
                    - the default is `None`
                - `add_stars_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroLuSt.synthetic.images.TPF.add_stars()`
                    - the default is `None`
                        - will be set to `dict()`
                - `add_noise_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroLuSt.synthetic.images.TPF.add_noise()`
                    - the default is `None`
                        - will be set to `dict(amplitude=0, bias=0)`
                - `add_custom_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroLuSt.synthetic.images.TPF.add_custom()`
                    - the default is `None`
                        - will be set to `dict(trend='linearx', amplitude=0)`

            Raises
            ------

            Returns
            -------
                - `frame`
                    - `np.ndarray`
                    - generated frame
            
            Comments
            --------
        """

        self.add_stars(**add_stars_kwargs)
        self.add_noise(**add_noise_kwargs)
        self.add_custom(**add_custom_kwargs)

        frame = self.get_frame()
        
        return frame

    def plot_result(self,
        plot_apertures:Union[List[int],Literal['all']]=None,
        fig:Figure=None,
        pcolormesh_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visualize the resulting generated TPF

            Parameters
            ----------
                - `plot_apertures`
                    - `list`, `Literal["all"]`, optional
                    - if `list`
                        - contains indices of apertures to show in the produced figure
                    - options for `Literal`
                        - `'all'`
                            - plots all apertures
                    - the default is `None`
                        - will not plot any apertures
                - `fig`
                    - `Figure`, optional
                    - matplotlib figure to place plots into
                    - the default is `None`
                        - will generate a new figure                        
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - created matplotlib figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`

            Comments
            --------
        """

        if plot_apertures is None: plot_apertures = []
        elif plot_apertures == 'all': plot_apertures = range(len(self.starparams))
        if pcolormesh_kwargs is None: pcolormesh_kwargs = dict()

        if self.mode == 'flux':
            frame2plot = self.frame
            c_lab = 'Flux [-]'
        elif self.mode == 'mag':
            frame2plot = self.frame_mag
            c_lab = 'Magnitude [mag]'

        if fig is None: fig = plt.figure(figsize=(6,5))
        ax1 = fig.add_subplot(111)
        mesh = ax1.pcolormesh(frame2plot[:,:,0], frame2plot[:,:,1], frame2plot[:,:,2], zorder=0, **pcolormesh_kwargs)
        if self.store_stars:
            for idx, apidx in enumerate(plot_apertures):
                try:
                    cont = ax1.contour(self.stars[apidx,:,:,2], levels=0, colors='C0', linewidths=1, zorder=1)
                except IndexError:
                    almf.printf(
                        msg=f'Ignoring `plot_apertures[{idx}]` because the index is out of bounds!',
                        context=f'{self.__class__.__name__}.plot_result()',
                        type='WARNING'
                    )

            #legend entries
            ax1.plot(np.nan, np.nan,  'C0-', label='Aperture')

        #labelling
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')

        cbar = fig.colorbar(mesh, ax=ax1)
        if self.mode == 'mag':
            cbar.ax.invert_yaxis()
            # cbar.ax.set_ylim(0-frame2plot.min()/2-1)
        cbar.set_label(c_lab)

        ax1.legend()

        axs = fig.axes

        return fig, axs


class TPF_Series:
    """
        - class to generate a series TPFs essentially simulating photometric data
        - assumes stars to have a gaussian Point Spread Function (PSF)
        - uses `astroLuSt.synthetic.images.TPF()` to generate individual frames

        Attributes
        ----------
            - `size`
                - `tuple`, `int`
                - size of the TPFs to generate
                - if `int`
                    - will be interpreted as `(size,size)`
                    - i.e., square frame with `size` pixels in x and y direction
                - if `tuple`
                    - `size[0]` denotes the number of pixels in x direction
                    - `size[1]` denotes the number of pixels in y direction
                - the default is `15`
            - `mode`
                - `Literal["flux","mag"]`, optional
                - whether to generate the frames using magnitudes or fluxes
                - allowed values are
                    - `'flux'`
                        - will use fluxes to generate the frames
                    - `'mag'`
                        - will use magnitudes to generate the frames
                - the default is `None`
                    - will be set to `'flux'`
            - `f_ref`
                - `float`, optional
                - reference flux to use when converting fluxes to magnitudes
                - the default is `1`
            - `m_ref`
                - `float`, optional
                - reference magnitude to use when converting magnitudes to fluxes
                - the default is `0`
            - `rng`
                - `np.random.default_rng`, `int`, optional
                - if `int`
                    - random seed to use in the random number generator
                - if `np.random.default_rng` instance
                    - random number gnerator to use for random generation
                - the default is `None`
                    - will use `np.random.default_rng(seed=None)`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `get_frames()`
            - `rvs()`
            - `plot_result()`
                - `init()`
                - `animate()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------
            - all calculations are executed in fluxes
                - in parallel a second frame in magnitudes gets calculated by converting the flux result
    """

    def __init__(self,
        size:Union[Tuple,int]=15,
        mode:Literal['flux','mag']=None,
        f_ref:float=1, m_ref:float=0,
        rng:Union[int,np.random.default_rng]=None,
        verbose:int=0,
        ) -> None:

        if isinstance(size, int):   self.size   = (size,size)
        else:                       self.size   = size
        if mode is None:            self.mode   = 'flux'
        else:                       self.mode   = mode
        if rng is None:             self.rng    = np.random.default_rng(seed=None)
        elif isinstance(rng, int):  self.rng    = np.random.default_rng(seed=rng)
        else:                       self.rng    = rng
        self.verbose                            = verbose
        self.f_ref                              = f_ref
        self.m_ref                              = m_ref
        
        #infered attributes
        self.tpf_s = np.empty((0,*self.size,3))
        self.starparams_s = np.empty((0,0,7))

        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    size={repr(self.size)},\n'
            f'    mode={repr(self.mode)},\n'
            f'    f_ref={repr(self.f_ref)}, m_ref={repr(self.m_ref)},\n'
            f'    rng={repr(self.rng.__class__.__name__)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')\n'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def get_frames(self,
        ) -> np.ndarray:
        """
            - method that returns the generated frames

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `frames`
                    - `np.ndarray`
                    - generated frames
                    - in fluxes/magnitudes according to `self.mode`
                    
            Comments
            --------
        """
        frames = self.tpf_s

        return frames
    

    def rvs(self,
        times:np.ndarray,
        variability:Callable=None,
        shape:Union[int,tuple]=None,
        add_stars_kwargs:dict=None,
        add_noise_kwargs:dict=None,
        add_custom_kwargs:dict=None,
        verbose:int=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to generate a series of frames given `times`
            - similar to the `scipy.stats` `rvs()` method
            - rvs ... random variates
            - will generate a series of random (or nonrandom) frames   

            Parameters
            ----------
                - `times`
                    - `np.ndarray`
                    - timestamps encoding time of each frame
                    - ideally a sorted array
                    - also phases can be passed
                        - then, `variability` has to be adjusted accordingly
                - `variability`
                    - `Callable`
                    - function encoding the variability for every generated star
                    - the function has to take two arguments
                        - `tp`
                            - `float`
                            - a time/phasestamp
                        - `fm`
                            - `np.ndarray`
                            - fluxes/magnitudes for all generated stars
                    - the function shall return one value
                        - `fm`
                            - `np.ndarray`
                            - the flux/magnitude value for the current time/phasestamp
                    - the function can implement variability for every single star separately
                    - an example for strong sinusoidal variability of the last generated star and weak sinusoidal variability (with random periods) of all other stars can be found below
                        >>> def mf_var(tp, fm):
                        >>>     period = 100
                        >>>     amp    = 7
                        >>>     rperiods = np.random.uniform(40, 50, size=fm[:-1].shape)
                        >>>     ramps    = np.random.uniform(.1, 2,  size=rperiods.shape)
                        >>>     fm[-1]  *= amp*(np.sin(2*np.pi*tp/period)+3) + 2E-3*np.random.randn()
                        >>>     fm[:-1] *= ramps*(np.sin(2*np.pi*tp/rperiods)+2) + 1E-3*np.random.randn(*fm[:-1].shape)
                        >>>     return fm
                    - the default is `None`
                        - will be implemented as `lambda tp, fm: fm`
                        - i.e., no variability at all
                - `shape`
                    - `int`, `tuple`, optional
                    - NOT USED BECAUSE DEFINED BY `self.size`
                    - number of samples to generate
                    - the default is `None`
                - `add_stars_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroLuSt.synthetic.images.TPF.add_stars()`
                    - the default is `None`
                        - will be set to `dict()`
                - `add_noise_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroLuSt.synthetic.images.TPF.add_noise()`
                    - the default is `None`
                        - will be set to `dict(amplitude=0, bias=0)`
                - `add_custom_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroLuSt.synthetic.images.TPF.add_custom()`
                    - the default is `None`
                        - will be set to `dict(trend='linearx', amplitude=0)`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`

            Raises
            ------

            Returns
            -------
                - `tpf_s`
                    - `np.ndarray`
                    - target pixel file series
                    - i.e. series of generated frames
                    - has shape `(nframes,self.size[0],self.size[1],3)`
                        - the last dimension contains
                            element 0: pixel coordinate in x direction
                            element 1: pixel coordinate in y direction
                            element 2: pixel flux/magnitude values
                - `starparams_s`
                    - `np.ndarray`
                    - contains for every frame for physical parameters for all targets
                    - has shape `(nframes,nstars,7)`
                        - the last dimension contains
                            - element 0: position in x direction (`posx`)
                            - element 1: position in y direction (`posy`)
                            - element 2: flux (`f`)
                            - element 3: magnitude (`m`)
                            - element 4: aperture (`aperture`)
                            - element 5: time/phase (`tp`)
                            - element 6: frame number (`idx`)
                             
            Comments
            --------
        """

        #default values
        if variability is None: variability = lambda tp, fm: fm
        if add_stars_kwargs is None:    add_stars_kwargs    = dict()
        if add_noise_kwargs is None:    add_noise_kwargs    = dict(amplitude=0, bias=0)
        if add_custom_kwargs is None:   add_custom_kwargs   = dict(trend='linearx', amplitude=0)
        if verbose is None:             verbose             = self.verbose

        #store times
        self.times = np.insert(times, 0, 0) #append zero-timestamp for rest frame

        #initialize TPF generator
        tpf = TPF(
            size=self.size,
            mode=self.mode,
            f_ref=self.f_ref, m_ref=self.m_ref,
            rng=self.rng,
            verbose=verbose,
        )

        #generate one frame for each timestamp
        for idx, tp in enumerate(self.times):

            #initial (rest) frame (will be ignored in the end)
            if idx == 0:
                #generate new instance for first added objects (rest frame)
                tpf.add_stars(
                    **add_stars_kwargs
                )

                #set rest params for next frames
                params = tpf.starparams.T
                
                #clean the frame for next timestamp ("readout")
                tpf.clean_frame(cleanparams=True)
            else:
                #modify rest params (simulates reference luminosities)
                tpf.add_stars(
                    nstars=-1,  #will be ignored anyways
                    posx=params[0],
                    posy=params[1],
                    f=variability(tp, params[2].copy()),
                    m=variability(tp, params[3].copy()),
                    aperture=params[4],
                )
                #noise and trends will be regenerated every frame
                tpf.add_noise(**add_noise_kwargs)
                tpf.add_custom(**add_custom_kwargs)

                #add new frame to array of frames
                frame = tpf.get_frame() #automaticall returns correct frame according to tpf.mode
                self.tpf_s = np.append(self.tpf_s, np.expand_dims(frame,0), axis=0)
                # if self.mode == 'flux':
                #     self.tpf_s = np.append(self.tpf_s, np.expand_dims(tpf.frame,0), axis=0)
                # elif self.mode == 'mag':
                #     self.tpf_s = np.append(self.tpf_s, np.expand_dims(tpf.frame_mag,0), axis=0)
                
                #add physical params to array of params (adding time, frame,)
                tpf_starparams = np.append(tpf.starparams,[[tp,idx]]*len(tpf.starparams),axis=1)
                tpf_starparams = np.expand_dims(tpf_starparams,0)   #reshaping for concatenation
                self.starparams_s = self.starparams_s.reshape(-1,tpf_starparams.shape[1],7) #reshaping for concatenation
                self.starparams_s = np.append(self.starparams_s, tpf_starparams, axis=0)    #appending

                #clean the frame for next timestamp ("readout")
                tpf.clean_frame(cleanparams=True)

        return self.tpf_s, self.starparams_s
    
    def plot_result(self,
        save:str=False,
        fig:Figure=None,
        pcolormesh_kwargs:dict=None,
        funcanim_kwargs:dict=None,
        save_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes,FuncAnimation]:
        """
            - method to render the animation resulting from the generated frames

            Parameters
            ----------
                - `save`
                    - `str`, optional
                    - path to the location of where to save the created animation
                    - if `str`
                        - will be interpreted as savepath
                    - the default is `False`
                        - will not save the animation
                - `fig`
                    - `Figure`, optional
                    - matplotlib figure to place plots into
                    - the default is `None`
                        - will generate a new figure
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict()`
                - `funcanim_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `matplotlib.animation.FuncAnimation()`
                    - the default is `None`
                        - will be set to `dict(frames=len(self.tpf_s))`
                - `save_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `matplotlib.animation.FuncAnimation().save()`
                    - the default is `None`
                        - will be set to `dict(fps=15)`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - created matplotlib figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
                - `anim`
                    - `matplotlib.animation.FuncAnimation()`
                    - created animation

            Comments
            --------
        """

        def init(mesh):
            """
                - function to initialize the animation
            """
            mesh.set_array(self.tpf_s[0,:,:,2])
            return
        
        def animate(frame, mesh, title):
            """
                - function to be executed at every new frame
            """
            mesh.set_array(self.tpf_s[frame,:,:,2])
            
            title.set_text(f'Frame {frame:.0f}')
            return

        cur_cmap = plt.rcParams["image.cmap"]

        #default parameters
        if pcolormesh_kwargs is None:   pcolormesh_kwargs = dict()
        if funcanim_kwargs is None:     funcanim_kwargs = dict()
        if 'frames' not in funcanim_kwargs.keys(): funcanim_kwargs['frames'] = len(self.tpf_s)
        if save_kwargs is None:         save_kwargs = dict()
        if 'fps' not in save_kwargs.keys(): save_kwargs['fps'] = 15
        if self.mode == 'flux':
            c_lab = 'Flux [-]'
        elif self.mode == 'mag':
            c_lab = 'Magnitude [mag]'

        #initialize figure
        if fig is None: fig = plt.figure(figsize=(6,5))
        ax1 = fig.add_subplot(111)
        title = fig.suptitle(f'Frame 0')
        
        #labelling
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')

        #plot first frame
        mesh = ax1.pcolormesh(
            self.tpf_s[0,:,:,0],
            self.tpf_s[0,:,:,1],
            self.tpf_s[0,:,:,2],
            zorder=0, **pcolormesh_kwargs
        )

        #add colorbar
        cbar = fig.colorbar(mesh, ax=ax1)
        if self.mode == 'mag':
            cbar.ax.invert_yaxis()
        cbar.set_label(c_lab)

        #animate plot (call init() at the start and animate() for every frame)
        anim = FuncAnimation(
            fig,
            partial(animate, mesh=mesh, title=title),
            init_func=partial(init, mesh=mesh),
            **funcanim_kwargs,   
        )

        #save if desired
        if isinstance(save, str):
            anim.save(save, **save_kwargs)
        
        axs = fig.axes

        return fig, axs, anim
    

#%%definitions
