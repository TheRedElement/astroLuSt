#TODO: Allow passing of weights for random.choice

#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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
                - tuple, int
                - size of the TPF to generate
                - if int
                    - will be interpreted as `(size,size)`
                    - i.e., square frame with `size` pixels in x and y direction
                - if tuple
                    - `size[0]` denotes the number of pixels in x direction
                    - `size[y]` denotes the number of pixels in y direction
            - `mode`
                - Literal, optional
                - whether to generate the frame using magnitudes or fluxes
                - allowed values are
                    - `'flux'`
                        - will use fluxes to generate the frame
                    - `'mag'`
                        - will use magnitudes to generate the frame
                - the default is `None`
                    - will be set to `'flux'`
            - `f_ref`
                - float, optional
                - reference flux to use when converting fluxes to magnitudes
                - the default is 1
            - `m_ref`
                - reference magnitude to use when converting magnitudes to fluxes
                - the default is 0
            - `store_stars`
                - bool, optional
                - wether to store an array of all generated stars including their apertures
                - if False
                    - will only store the final (composite) frame
                - the default is False
            - `rng`
                - np.random.default_rng, int, optional
                - if int
                    - random seed to use in the random number generator
                - if np.random.default_rng instance
                    - random number gnerator to use for random generation
                - the default is `None`
                    - will use `np.random.default_rng(seed=None)`
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Methods
        -------
            - `star()`
            - `add_stars()`
            - `add_custom()`
            - `add_noise()`
            - `aperture_from_mask()`
            - `plot_result()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing
            - scipy

        Comments
        --------
            - all calculations are executed in fluxes
                - in parallel a second frame in magnitudes gets calculated by converting the flux result
    """

    def __init__(self,
        size:Union[Tuple,int],
        mode:Literal['flux','mag']=None,
        f_ref:float=1, m_ref:float=0,
        store_stars:bool=False,
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
        self.frame_mag[:,:,2] = alpp.fluxes2mags(self.frame_mag[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)           #reset magnitude values

        #intermediate storage
        self.store_stars = store_stars

        #infered attributes
        self.stars = np.empty((0,self.size[0],self.size[1],2))
        self.starparams = []

        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    size={repr(self.size)},\n'
            f'    mode={repr(self.mode)},\n'
            f'    f_ref={repr(self.f_ref)}, m_ref={repr(self.m_ref)},\n'
            f'    store_stars={repr(self.store_stars)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

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
                    - np.ndarray
                    - position of the star in pixels
                        - lower left corner has coordinates `(0,0)`
                - `f`
                    - float, optional
                    - flux of the star
                        - implemented as amplitude (scaling factor) to gaussian PSF
                    - used to determine `m` via flux-magnitude relation
                        - calls `astroLuSt.physics.photometry.fluxes2mags()`
                        - will use `self.m_ref` and `self.f_ref` for the conversion
                    - the default is `None`
                        - will be infered from `m`
                - `m`
                    - float, optional
                    - magnitude of the star
                    - used to determine `f` via flux-magnitude relation
                        - calls `astroLuSt.physics.photometry.mags2fluxes()`
                        - will use `self.m_ref` and `self.f_ref` for the conversion
                    - will be ignored if `f` is not `None`
                    - the default is `None`
                        - will be ignored
                - `aperture`
                    - float, opational
                    - "ground truth" aperture of the star
                        - defined as aperture = 2*FWHM = 2*2*std*sqrt(ln(2))
                        - implemented as proportionality factor to covariance
                    - the default is 1

            Raises
            ------
                - `ValueError`
                    - if both `f` and `m` are `None`

            Returns
            -------
                - `star`
                    - np.ndarray
                    - has shape `(self.size[0],self.size[1],2)`
                        - entry 0 in last dimension contains the fluxes
                        - entry 1 in last dimension contains the aperture mask

            Comments
            --------
        """

        if f is None and m is not None:
            f =  alpp.mags2fluxes(m=m, m_ref=self.m_ref, f_ref=self.f_ref)
        elif f is not None and m is None:
            m =  alpp.fluxes2mags(f=f, f_ref=self.f_ref, m_ref=self.m_ref)
        if f is not None and m is not None:
            m =  alpp.fluxes2mags(f=f, f_ref=self.f_ref, m_ref=self.m_ref)
        else:
            raise ValueError(f'At least one of `f` and `m` has to be not `None` but they have values {f} and {m}.')

        cov = (aperture/(2*2*np.sqrt(np.log(2))))**2     #aperture = 2*halfwidth = 2* 2*std*sqrt(ln(2))
        star = f*sps.multivariate_normal(
            mean=pos, cov=cov, allow_singular=True
        ).pdf(self.frame[:,:,:2])

        b = (np.sqrt(np.sum((self.frame[:,:,:2]-pos)**2, axis=2))<aperture)
        
        star = np.append(np.expand_dims(star,axis=2), np.expand_dims(b,axis=2), axis=2)

        #store generated parameters and clean frames
        self.starparams.append([pos, f, m, aperture])
        if self.store_stars:
            self.stars = np.append(self.stars, np.expand_dims(star,0), axis=0)

        return star

    def add_stars(self,
        nstars:int=1,
        posx:Union[dict,np.ndarray]=None,
        posy:Union[dict,np.ndarray]=None,
        f:Union[dict,np.ndarray]=None,
        m:Union[dict,np.ndarray]=None,
        aperture:Union[dict,np.ndarray]=None,
        verbose:int=None,
        ) -> None:
        """
            - method to add stars to the on the CCD (TPF)
            
            Parameters
            ----------
                - `nstars`
                    - int, optional
                    - number of stars to generate
                    - the default is 1 
                - `posx`
                    - dict, np.ndarray, optional
                    - x values of the stars position in pixels
                    - if dict
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random positions
                            - `'params'`
                                - value is dict or list
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as list of positional args (for list)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':0,  'high':100})`
                        - i.e. a uniform distribution from 0 to 100
                - `posy`
                    - dict, np.ndarray, optional
                    - y values of the stars position in pixels
                    - if dict
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random positions
                            - `'params'`
                                - value is dict or list
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as list of positional args (for list)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':0,  'high':100})`
                        - i.e. a uniform distribution from 0 to 100
                - `f`
                    - dict, np.ndarray, optional
                    - fluxes of the stars to generate
                    - if dict
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random flux values
                            - `'params'`
                                - value is dict or list
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as list of positional args (for list)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':1,  'high':100})`
                        - i.e. a uniform distribution from 1 to 100
                - `m`
                    - dict, np.ndarray, optional
                    - magnitudes of the stars to generate
                    - if dict
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random magnitudes
                            - `'params'`
                                - value is dict or list
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as list of positional args (for list)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':-4,  'high':4})`
                        - i.e. a uniform distribution from -4 to 4
                - `aperture`
                    - dict, np.ndarray, optional
                    - apertures to use for the stars 
                    - if dict
                        - has to contain 2 keys
                            - `'dist'`
                                - value is a string
                                - specifies distribution implemented for `np.random.default_rng()`
                                    - this distribution will be used to generate random apertures
                            - `'params'`
                                - value is dict or list
                                - parameters of `'dist'`
                                    - passed as `key:value`-pairs (kwargs, for dict)
                                    - passed as list of positional args (for list)
                    - the default is `None`
                        - will be set to `dict(dist='uniform', params={'low':-4,  'high':4})`
                        - i.e. a uniform distribution from 1 to 20
                - `verbose`
                    - int, optional
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
        if posx is None:        posx    =dict(dist='uniform', params={'low':0,  'high':100})
        if posy is None:        posy    =dict(dist='uniform', params={'low':0,  'high':100})
        if f is None:           f       =dict(dist='uniform', params={'low':1,  'high':100})
        if m is None:           m       =dict(dist='uniform', params={'low':-4, 'high':4})
        if aperture is None:    aperture=dict(dist='uniform', params={'low':1,  'high':20})


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
            f = alpp.mags2fluxes(m=m, m_ref=self.m_ref, f_ref=self.f_ref)

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
            self.frame_mag[:,:,2] = alpp.fluxes2mags(self.frame[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)

        return
    
    def add_custom(self,
        trend:Union[np.ndarray,Literal['lineary','linearx']],
        amplitude:float=1,
        ) -> None:
        #TODO: Add other common trends
        """
            - method to add custom trends to the final image
            
            Parameters
            ----------
                - `trend`
                    - np.ndarray, Literal
                    - trend to add to `self.frame`
                    - if a np.ndarray
                        - will be added to `self.frame` via element-wise addition
                        - has to have same shape as `self.frame.shape[:2]`
                            - i.e. `self.size`
                    - options for Literal
                        - `lineary`
                            - linear trend in y direction
                            - generated via `np.linspace(0, np.ones((self.frame.shape[0])), self.frame.shape[1], axis=0)`
                        - `linearx`
                            - linear trend in x direction
                            - generated via `np.linspace(0, np.ones((self.frame.shape[0])), self.frame.shape[1], axis=1)`
                - `amplitude`
                    - float, optional
                    - scaling factor of the trend
                    - the default is 1
                        - i.e. no scaling

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if trend in ['lineary', 'linearx']:
            if trend == 'lineary':
                trend = np.linspace(1,2*np.ones((self.frame.shape[0])), self.frame.shape[1], axis=0)
            if trend == 'linearx':
                trend = np.linspace(1,2*np.ones((self.frame.shape[0])), self.frame.shape[1], axis=1)

        self.frame[:,:,2] += amplitude*trend

        #convert to magnitudes            
        if self.mode == 'mag':
            self.frame_mag[:,:,2] = alpp.fluxes2mags(self.frame[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)

        return

    def add_noise(self,
        amp:float=1E-3, bias:float=1E-1,
        ) -> None:
        #TODO: add different noise parts (photon noise, dead pixels, hot pixels, ...)
        """
            - method to add (gaussian) noise to the input frame

            Parameters
            ----------
                - `amp`
                    - float, optional
                    - amplitude of the added noise
                    - the default is 1E-3
                - `bias`
                    - float, optional
                    - offset of noise from 0 (similar to bias-current)
                    - the default is 1E-1

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        noise = amp*self.rng.standard_normal(size=(self.frame.shape[:2])) + bias
        
        self.frame[:,:,2] += noise

        #convert to magnitudes            
        if self.mode == 'mag':
            self.frame_mag[:,:,2] = alpp.fluxes2mags(self.frame[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)

        return
    
    def aperture_from_mask(self,
        aperture_mask:np.ndarray,
        ) -> float:
        """
            - method to estimate the aperture radius from a aperture mask

            Parameters
            ----------
                - `aperture_mask`
                    - np.ndarray
                    - aperture mask to use for the determination
                    - for this class stored in `TPF.stars[staridx,:,:,1]`
                        - i.e. second entry of last axis if `store_star` is set to `True`
                    - estimation executed via sqrt(aperture_mask.sum()/pi)
                        - solving aperture = pi*r**2
            
            Raises
            ------

            Returns
            -------
                - `aperture`
                    - float
                    - estimated aperture radius

            Comments
            --------
                - only (approximately) correct if the whole aperture is contained within the frame
        """

        aperture = np.sqrt(aperture_mask.sum()/np.pi)

        return aperture

    def plot_result(self,
        plot_apertures:List[int]=None,
        pcolormesh_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visualize the resulting generated TPF

            Parameters
            ----------
                - `plot_apertures`
                    - list, optional
                    - contains indices of apertures to show in the produced figure
                    - the default is `None`
                        - will not plot any apertures
                - `pcolormesh_kwargs`
                    - dict, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict('viridis')`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - Figure
                    - created matplotlib figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------
        """

        if plot_apertures is None: plot_apertures = []
        if pcolormesh_kwargs is None: pcolormesh_kwargs = dict()

        if self.mode == 'flux':
            frame2plot = self.frame
            c_lab = 'Flux [-]'
            if 'cmap' not in pcolormesh_kwargs.keys():
                pcolormesh_kwargs['cmap'] = 'viridis'
        elif self.mode == 'mag':
            frame2plot = self.frame_mag
            c_lab = 'Magnitude [mag]'
            if 'cmap' not in pcolormesh_kwargs.keys():
                pcolormesh_kwargs['cmap'] = 'viridis_r'

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        mesh = ax1.pcolormesh(frame2plot[:,:,0], frame2plot[:,:,1], frame2plot[:,:,2], zorder=0, **pcolormesh_kwargs)
        if self.store_stars:
            for idx, apidx in enumerate(plot_apertures):
                try:
                    cont = ax1.contour(self.stars[apidx,:,:,1], levels=[0], colors='r', linewidths=1, zorder=1)
                except IndexError:
                    almf.printf(
                        msg=f'Ignoring `plot_apertures[{idx}]` because the index is out of bounds!',
                        context=f'{self.__class__.__name__}.plot_result()',
                        type='WARNING'
                    )

        #legend entries
        ax1.plot(np.nan, np.nan,  'r-', label='Aperture')

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

    def __init__(self) -> None:
        pass


#%%definitions
