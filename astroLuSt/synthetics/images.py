#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import scipy.stats as sps

from astroLuSt.physics import photometry as alpp
from astroLuSt.monitoring import formatting as almf

from typing import Union, Tuple, Callable, Literal, List

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
            - `f_ref`
            - `m_ref`
            - `store_stars`
                - bool, optional
                - wether to store an array of all generated stars including their apertures
                - if False
                    - will only store the final (composite) frame
                - the default is False
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
    """

    def __init__(self,
        size:Union[Tuple,int],
        mode:Literal['flux','mag']=None,
        f_ref:float=1, m_ref:float=0,
        store_stars:bool=False,
        verbose:int=0,
        ) -> None:

        if isinstance(size, int):   self.size   = (size,size)
        else:                       self.size   = size
        if mode is None:            self.mode   = 'flux'
        else:                       self.mode   = mode
        self.verbose                            = verbose
        self.f_ref                              = f_ref
        self.m_ref                              = m_ref

        x = np.arange(self.size[0])
        y = np.arange(self.size[1])
        xx, yy = np.meshgrid(x, y)

        self.frame = np.zeros(shape=(*self.size,1))
        self.frame = np.concatenate((np.expand_dims(xx,2), np.expand_dims(yy,2), self.frame), axis=2)
        self.store_stars = store_stars

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
        random:Union[dict,bool],
        pos:np.ndarray=None,
        f:np.ndarray=None,
        aperture:np.ndarray=None,
        verbose:int=None,
        ) -> None:
        """
            - method to add stars to the on the CCD (TPF)
            
            Parameters
            ----------
                - `random`
                    - dict, bool
                    - if dict
                        - configurations for the generation of random stars
                        - allowed keys
                            - `nstars`
                                - int
                                - number of stars to generate
                            - `sizex`
                                - int
                                - maximum number of pixels in x direction
                            - `sizey`
                                - int
                                - maximum number of pixels in y direction
                            - `fmin`
                                - float
                                - minimum flux value
                            - `fmax`
                                - float
                                - maximum flux value
                            - `apmin`
                                - float
                                - minimum aperture size
                            - `apmax`
                                - float
                                - maximum aperture size
                - `pos`
                    - np.ndarray, optional
                    - 2d array of shape `(nstars,2)`
                        - element 0 of second axis is the x position in pixels
                        - element 1 of second axis is the y position in pixels
                    - the default is `None`
                - `f`
                    - np.ndarray, optional
                    - fluxes of the stars to generate
                    - the default is `None`
                - `aperture`
                    - np.ndarray, optional
                    - apertures of the stars to generate
                    - the default is `None`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`

            Raises
            ------
                - `ValueError`
                    - if too little information has been provided and `random` is set to `False`

            Returns
            -------

            Comments
            --------
        """


        #default parameters
        default_random_config = {
            'nstars':1,
            'sizex':self.frame.shape[0],
            'sizey':self.frame.shape[1],
            'fmin':0,
            'fmax':100,
            'apmin':1,
            'apmax':10,
        }
        if random == False:
            random_config = default_random_config
        else:
            rand_keys = np.array(list(random.keys()))
            invalid_keys = ~np.isin(rand_keys, list(default_random_config.keys()))
            if np.any(invalid_keys):
                almf.printf(
                    msg=f'Some keys ({rand_keys[invalid_keys]}) are not valid and will be ignored. Allowed are {default_random_config.keys()}!',
                    context=f'{self.__class__.__name__}.add_stars()',
                    type='WARNING'
                )
            for k in random.keys():
                default_random_config[k] = random[k]
            
            random_config = default_random_config.copy()
        if verbose is None: verbose = self.verbose

        #generate random stars if requested
        if random != False:
            #get all possible combinations of posx and posy
            posx        = range(random_config['sizex'])
            posy        = range(random_config['sizey'])
            pos         = np.array(np.meshgrid(posx, posy)).T.reshape(-1,2)
            
            #get random indices that each position occurs once
            randidxs    = np.random.choice(range(random_config['sizex']*random_config['sizey']), size=random_config['nstars'], replace=False)
            
            #choose random parameters for stars
            pos         = pos[randidxs]
            f           = np.random.choice(range(random_config['fmin'],  random_config['fmax'], 1), size=(random_config['nstars']))
            aperture    = np.random.choice(range(random_config['apmin'], random_config['apmax'],1), size=(random_config['nstars']))

            #report generated parameters
            if verbose > 2:
                almf.printf(
                    msg=(
                        f'Generated Parameters:'
                        f'pos:{pos.tolist()}, f:{f}, aperture:{aperture}'
                    ),
                    context=f'{self.__class__.__name__}.add_stars()'
                )                
        elif random == False and f is None:
            raise ValueError("`f` has to be provided if `random` is set to `False`")
        elif random == False and aperture is None:
            raise ValueError("`aperture` has to be provided if `random` is set to `False`")
        
        #add stars to TPF
        for posi, fi, api in zip(pos, f, aperture):
            star = self.star(pos=posi, f=fi, aperture=api)

            self.frame[:,:,2] += star[:,:,0]

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

        if trend == 'lineary':
            trend = np.linspace(0,np.ones((self.frame.shape[0])), self.frame.shape[1], axis=0)
        if trend == 'linearx':
            trend = np.linspace(0,np.ones((self.frame.shape[0])), self.frame.shape[1], axis=1)

        self.frame[:,:,2] += amplitude*trend

        return

    def add_noise(self,
        amp:float=1E-3,
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

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        self.frame[:,:,2] += amp*np.random.randn(*self.frame.shape[:2])

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
        plot_apertures:List[int]=None
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

        if self.mode == 'flux':
            frame2plot = self.frame
            c_lab = 'Flux [-]'
            cmap = 'viridis'
        elif self.mode == 'mag':
            frame2plot = self.frame
            frame2plot[:,:,2] = alpp.fluxes2mags(frame2plot[:,:,2], f_ref=self.f_ref, m_ref=self.m_ref)
            c_lab = 'Magnitude [mag]'
            cmap = 'viridis_r'

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        mesh = ax1.pcolormesh(self.frame[:,:,0], self.frame[:,:,1], self.frame[:,:,2], cmap=cmap, zorder=0)
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
        # cbar = fig.colorbar(cont, ax=ax1)
        if self.mode == 'mag':
            cbar.ax.invert_yaxis()
        # cbar.ax.set_ylim(0-mag_range/2-1)
        cbar.set_label('Flux [-]')

        ax1.legend()

        axs = fig.axes

        return fig, axs


class TPF_Series:

    def __init__(self) -> None:
        pass


#%%definitions
