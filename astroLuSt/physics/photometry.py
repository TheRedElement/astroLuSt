#TODO: Documentation
#TODO: Calculation with Magnitudes

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
                - float, np.ndarray, optional
                - apparent magnitude(s)
                - will infer `M` if both `M` and `m` are not `None`
                - the default is `None`
                    - will be infered from `M`
            - `M`
                - float, np.ndarray, optional
                - absolute magnitude(s)
                - will infer `M` if both `M` and `m` are not `None`
                - the default is `None`
                    - will be infered from `m`
            - `d`
                - float, np.ndarray, optional
                - distance(s) in parsec [pc]
                - will use `d` if both `d` and `plx` are not `None`
                - the default is `None`
                    - will try to infer distances via `plx`
            - `plx`
                - float, np.ndarray, optional
                - parallax(es) in arcseconds [arcsec]
                - will use `d` if both `d` and `plx` are not `None`
                - the default is `None`
                    - will try to infer parallaces via `d`

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
        ) -> None:


        self.m      = m
        self.M      = M
        self.d      = d
        self.plx    = plx

        #calculate parameters
        self.infer_d_plx()
        self.infer_m_M()


        pass

    def __repr__(self) -> str:
        return (
            f'DistanceModule(\n'
            f'    m={repr(self.m)}, M={repr(self.M)},\n'
            f'    d={repr(self.d)}, plx={repr(self.plx)},\n'
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
        #infer plx from d
        elif self.d is not None and self.plx is None:
            self.plx = 1/self.d
        #use d if both provided
        elif self.d is not None and self.plx is not None:
            self.plx = 1/self.d


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

        self.M = self.m - 5*np.log10(self.d/10)

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

        return

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Literal

from astroLuSt.visualization import plotting as alvp

class BestAperture:
    """
        - class to exectue an analysis for the determination of the best aperture

        Attributes
        ----------
            - `mode`
                - Literal, optional
                - mode to use
                - can be one of
                    - `'flux'`
                        - will consider equations for fluxes to execute calculations
                    - `'mag'`
                        - will consider equations for magnitudes to execute calculations
            - `store_ring_masks`
                - bool, optional
                - whether to store all sky-ring masks created during calculations
                    - those will be used during i.e., plotting
                - the default is True
            - `store_aperture_masks`
                - bool, optional
                - whether to store all aperture masks created during calculations
                    - those will be used during i.e., plotting
                - the default is True
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Infered Attributes
        ------------------
            - `aperture_res`
                - np.ndarray
                - contains results for the analysis of the aperture
                - has shape `(nradii,3)`
                    - `nradii` is the number of tested radii
                    - the last axis contains
                        - element 0: tested radius
                        - element 1: total flux within aperture
                        - element 2: number of enclosed pixels
            - `ring_res`
                - np.ndarray
                - contains results for the analysis of the sky-ring
                - has shape `(nradii*nwidths,4)`
                    - `nradii` is the number of tested radii
                    - `nwidths` is the number of tested widths for the sky-ring
                    - the last axis contains
                        - element 0: tested radius
                        - element 1: tested width
                        - element 2: total flux within aperture
                        - element 3: number of enclosed pixels
            - `aperture_masks`
                - np.ndarray
                - contains boolean arrays of all tested apertures
            - `ring_masks`
                - np.ndarray
                - contains boolean arrays of all tested sky-rings

        Methods
        -------
            - `__check_frames_shape()`
            - `__store_frames()`
            - `get_sum_frame()`
            - `test_aperture()`
            - `test_background_skyring()`
            - `fit()`
            - `predict()`
            - `fit_predict()`
            - `plot_results()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing

        Comments
        --------

    """

    def __init__(self,
        mode:Literal['flux','mag']=None,
        store_aperture_masks:bool=True,
        store_ring_masks:bool=True,
        verbose:int=0,
        ) ->  None:

        if mode is None:    self.mode   = 'flux'
        else:               self.mode   = mode
        self.store_aperture_masks       = store_aperture_masks
        self.store_ring_masks           = store_ring_masks
        self.verbose                    = verbose
        
        #infered attributes
        self.aperture_res   = np.empty((0,3))
        self.ring_res       = np.empty((0,4))
        self.aperture_masks = np.empty((0,0,0))
        self.ring_masks     = np.empty((0,0,0))
        

        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'mode={repr(self.mode)}),\n'
            f'store_ring_mask={repr(self.store_ring_masks)}),\n'
            f'store_aperture_masks={repr(self.store_aperture_masks)}),\n'
            f'verbose={self.verbose},\n'
            f'\n)'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def __check_frames_shape(self,
        frames
        ) -> None:
        """
            - private method to check if `frames` is of the correct shape

            Parameters
            ----------
                - `frames`
                    - np.ndarray
                    - series of frames to consider for the analysis
                    - has to have shape `(nframes,npixels,npixels,3)`
                        - the last axis contains
                            - element 0: posistion in x direction
                            - element 1: posistion in y direction
                            - element 2: flux/magnitude values

            Raises
            ------
                - `ValueError`
                    - if `frames` has a wrong shape

            Returns
            -------

            Comments
            --------
        """

        if len(frames.shape) == 4:
            self.frames                 = frames
        elif len(frames.shape) == 3:
            self.frames                 = np.expand_dims(frames,0)
        else:
            raise ValueError(f'`frames` has to be 4d but has shape {frames.shape}!')
        
        return

    def __store_frames(self,
        frames:np.ndarray,
        context:Literal['aperture','skyring']
        ) -> None:
        """
            - private method to store passed frames
            - also initializes storage arrays for aperture and sky-ring masks

            Parameters
            ----------
                - `frames`
                    - np.ndarray
                    - series of frames to consider for the analysis
                    - has to have shape `(nframes,npixels,npixels,3)`
                        - the last axis contains
                            - element 0: posistion in x direction
                            - element 1: posistion in y direction
                            - element 2: flux/magnitude values
                - `context`
                    - Literal
                    - context on where the method was called
                        - necessary to know to not override some previously initialized infered attribute
                    - options are
                        - `'aperture'`
                            - if apertures gets tested
                        - `'skyring'`
                            - if sky-rings gets tested (for background)
            Raises
            ------

            Returns
            -------

            Comments
            --------

        """

        self.frames = frames

        #get frame of pixel-wise sum
        self.get_sum_frame()

        #update frame-dependent attributes (only when in respective functions)
        if context == 'aperture':
            self.aperture_masks = np.empty((0,*self.frames.shape[1:3]))
        elif context == 'skyring':
            self.ring_masks     = np.empty((0,*self.frames.shape[1:3]))

        return

    def get_sum_frame(self,
        ) -> np.ndarray:
        """
            - method to get the pixel-wise sum of all frames `self.frames`
            - used to visually determine location and size of aperture

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `sum_frame`
                    - np.ndarray
                    - frame consisting of the pixel wise total flux/brightness

            Comments
            --------
        """

        self.sum_frame = self.frames[0].copy()
        
        if self.mode == 'flux':
            self.sum_frame[:,:,2] = np.nansum(self.frames[:,:,:,2], axis=0)
        elif self.mode == 'mag':
            self.sum_frame[:,:,2] = mags_sum(self.frames[:,:,:,2], axis=0)

        return self.sum_frame

    def test_aperture(self,
        frames:np.ndarray,
        posx:float, posy:float,
        r_aperture:np.ndarray,
        ) -> None:
        """
            - method to test different apertures
        
            Parameters
            ----------
                - `frames`
                    - np.ndarray
                    - series of frames to consider for the analysis
                    - has to have shape `(nframes,npixels,npixels,3)`
                        - the last axis contains
                            - element 0: posistion in x direction
                            - element 1: posistion in y direction
                            - element 2: flux/magnitude values
                - `posx`
                    - float
                    - position of the aperture in x direction
                - `posy`
                    - float
                    - position of the aperture in y direction
                - `r_aperture`
                    - np.ndarray
                    - test radii of the aperture
                    - will test every radius and calculate total flux/magnitude contained within that radius
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        
        """

        #initial checks and saves
        self.__check_frames_shape(frames)
        self.__store_frames(frames, context='aperture')

        #construct position array
        pos = np.array([posx,posy])

        #check all apertures
        for r in r_aperture:
            
            #create aperture mask (boolean array)
            aperture_mask = (np.sqrt(np.sum((self.sum_frame[:,:,:2]-pos)**2, axis=2)) < r)

            #get flux contained in aperture
            aperture_flux = np.sum(self.sum_frame[aperture_mask,2])#/np.sum(aperture_mask)**2
            
            #store result
            self.aperture_res = np.append(self.aperture_res, np.array([[r, aperture_flux, aperture_mask.sum()]]), axis=0)

            #store aperture_mask for current r
            if self.store_aperture_masks:
                self.aperture_masks = np.append(self.aperture_masks, np.expand_dims(aperture_mask,0), axis=0)

        return
    
    def test_background_skyring(self,
        frames:np.ndarray,
        posx:float, posy:float,
        rw_sky:np.ndarray,# w_sky:np.ndarray,
        ) -> None:
        """
            - method to test various sky-rings for background determination
            - a sky ring is described by its radius (`r_sky`) and width (`w_sky`)

            Parameters
            ----------
                - `frames`
                    - np.ndarray
                    - series of frames to consider for the analysis
                    - has to have shape `(nframes,npixels,npixels,3)`
                        - the last axis contains
                            - element 0: posistion in x direction
                            - element 1: posistion in y direction
                            - element 2: flux/magnitude values
                - `posx`
                    - float
                    - position of the sky-ring in x direction
                - `posy`
                    - float
                    - position of the sky-ring in y direction
                - `rw_sky`
                    - np.ndarray
                    - test specifications for sky-rings to test
                    - has to have shape `(nrings,2)`
                        - element 0: radii of the sky-rings
                        - element 1: widths of the sky-rings
                                
            Raises
            ------

            Returns
            -------

            Comments
            --------

        """

        #initial checks and saves
        self.__check_frames_shape(frames)
        self.__store_frames(frames, context='skyring')
            
        #generate position array
        pos = np.array([posx,posy])
        
        #test all test sky-rings (radii-width combinations)
        for (r, w) in rw_sky:
            
            #create ring-mask (everything inside interval [r, r+w])
            ring_mask = \
                    (np.sqrt(np.sum((self.sum_frame[:,:,:2]-pos)**2, axis=2)) >r)\
                &(np.sqrt(np.sum((self.sum_frame[:,:,:2]-pos)**2, axis=2)) < r+w)

            #calculate flux within sky-ring
            ring_flux = np.sum(self.sum_frame[ring_mask,2])#/np.sum(ring_mask)**2

            #get sky-ring results
            self.ring_res = np.append(self.ring_res, np.array([[r, w, ring_flux, ring_mask.sum()]]), axis=0)
            
            #store ring_mask for current w
            if self.store_ring_masks:
                self.ring_masks = np.append(self.ring_masks, np.expand_dims(ring_mask,0), axis=0)

        return
    
    def fit(self,
        frames:np.ndarray,
        posx:float,
        posy:float,
        r_aperture:np.ndarray,
        rw_sky:np.ndarray,
        test_aperture_kwargs:dict=None,
        test_background_kwargs:dict=None,
        ):
        """
            - method to fit the estimator
            
            Parameters
            ----------
                - 
        """
        #TODO: Add alternate background estimate options

        if test_aperture_kwargs is None:    test_aperture_kwargs    = dict()
        if test_background_kwargs is None:  test_background_kwargs  = dict()

        self.test_aperture(
            frames=frames,
            posx=posx,
            posy=posy,
            r_aperture=r_aperture,
            **test_aperture_kwargs
        )

        self.test_background_skyring(
            frames=frames,
            posx=posx,
            posy=posy,
            rw_sky=rw_sky,
            **test_background_kwargs
        )

        return
    
    def predict(self,
        ):

        almf.printf(
            msg=f'Not implemented yet. Call `{self.__class__.__name__}.{self.plot_result.__name__}()` to visualize the executed analysis.',
            context=f'{self.__class__.__name__}.{self.predict.__name__}()',
            type='WARNING',
        )

        return

    def fit_predict(self,
        frames:np.ndarray,
        posx:float,
        posy:float,
        r_aperture:np.ndarray,
        rw_sky:np.ndarray,
        # r_sky:np.ndarray,
        # w_sky:np.ndarray,
        fit_kwargs:dict=None,
        predict_kwargs:dict=None,
        ):

        if fit_kwargs is None:      fit_kwargs      = dict()
        if predict_kwargs is None:  predict_kwargs  = dict()

        self.fit(
            frames=frames,
            posx=posx,
            posy=posy,
            r_aperture=r_aperture,
            rw_sky=rw_sky
            **fit_kwargs
        )
        self.predict(**predict_kwargs)

        return

    def plot_result(self,
        plot_aperture_r:np.ndarray=None,
        plot_sky_rings_rw:np.ndarray=None,
        aperture_cmap:str=None,
        sky_rings_cmap:str=None,
        fig:Figure=None,
        sort_rings_apertures:bool=True,
        plot_kwargs:dict=None,
        scatter_kwargs:dict=None,
        ) -> Union[Figure,plt.Axes]:

        #default values
        if plot_aperture_r is None:   plot_aperture_r   = np.empty((0))
        if plot_sky_rings_rw is None: plot_sky_rings_rw = np.empty((0,2))
        if sky_rings_cmap is None:    sky_rings_cmap    = 'autumn'
        if aperture_cmap is None:     aperture_cmap     = 'winter'
        if plot_kwargs is None:       plot_kwargs       = dict(lw=1)
        if scatter_kwargs is None:    scatter_kwargs    = dict(s=5, cmap='viridis')

        #kwargs of outline for star aperture plot
        outline_kwargs = plot_kwargs.copy()
        outline_kwargs['lw'] = plot_kwargs['lw']*4
        outline_kwargs['color'] = 'w'

        #sorting
        if sort_rings_apertures:
            plot_aperture_r   = np.sort(plot_aperture_r)[::-1]
            plot_sky_rings_rw = np.sort(plot_sky_rings_rw)[::-1]

        #plotting
        if fig is None: fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        #plot sum frame
        mesh = ax1.pcolormesh(self.sum_frame[:,:,0], self.sum_frame[:,:,1], self.sum_frame[:,:,2], zorder=0)
        
        #plot some selected sky rings
        if len(plot_sky_rings_rw) > 0:
            colors_sky_ring = alvp.generate_colors(len(plot_sky_rings_rw), cmap=sky_rings_cmap)
            for idx, (wsr, rsr) in enumerate(plot_sky_rings_rw):

                br = (rsr==self.ring_res[:,0])
                bw = (wsr==self.ring_res[:,1])
                try:
                    mesh_sr = ax1.pcolormesh(self.sum_frame[:,:,0], self.sum_frame[:,:,1], self.ring_masks[bw&br][0], zorder=2, edgecolor=colors_sky_ring[idx], facecolors='none')
                    mesh_sr.set_alpha(self.ring_masks[bw&br][0])
                except IndexError as i:
                    alme.LogErrors().print_exc(
                        e=i,
                        prefix=(
                            f'EXCEPTION({self.__class__.__name__}.plot_results()).\n'
                            f'    Ignoring plotting of sky-ring...\n'
                            f'    Original ERROR:'
                        )
                    )
                
                if idx == 0: lab = 'Radius Skyring'
                else: lab = None
                ax2.axvline(rsr, color=colors_sky_ring[idx], linestyle='--', label=lab)
        
        #plot some selected apertures
        if len(plot_aperture_r) > 0:
            colors_aperture = alvp.generate_colors(len(plot_aperture_r), cmap=aperture_cmap)
            for idx, ra in enumerate(plot_aperture_r):
                br = (ra==self.aperture_res[:,0])
                try:
                    mesh_a = ax1.pcolormesh(self.sum_frame[:,:,0], self.sum_frame[:,:,1], self.aperture_masks[br][0], zorder=2, edgecolor=colors_aperture[idx], facecolors='none')
                    mesh_a.set_alpha(self.aperture_masks[br][0])
                except IndexError as i:
                    alme.LogErrors().print_exc(
                        e=i,
                        prefix=(
                            f'EXCEPTION({self.__class__.__name__}.plot_results()).\n'
                            f'    Ignoring plotting of aperture...\n'
                            f'    Original ERROR:'
                        )
                    )

                #show which apertures are plotted
                if idx == 0: lab = 'Radius Aperture'
                else: lab = None
                ax2.axvline(ra, color=colors_aperture[idx], label=lab)
                


        #plot star aperture and sky ring
        ax2.plot(self.aperture_res[:,0], self.aperture_res[:,1], label=None,            **outline_kwargs)
        ax2.plot(self.aperture_res[:,0], self.aperture_res[:,1], label='Aperture Curve Growth', **plot_kwargs)
        sctr = ax2.scatter(self.ring_res[:,:1], self.ring_res[:,2], c=self.ring_res[:,1], label='Sky Ring Curve Growth', **scatter_kwargs)


        #add colorbars
        cmap1 = fig.colorbar(mesh, ax=ax1)
        cmap2 = fig.colorbar(sctr, ax=ax2)

        ax2.legend(loc='upper right')


        #labelling
        cmap2.set_label('Sky Ring Width')

        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')
        ax2.set_xlabel('Radius')
        if self.mode == 'flux':
            cmap1.set_label('Total Flux (Pixel-Wise)')
            ax2.set_ylabel('Aperture Flux')
        elif self.mode == 'mag':
            cmap1.set_label('Total Magnitude (Pixel-Wise)')
            ax2.set_ylabel('Aperture Magnitude')

        plt.show()

        axs = fig.axes
        
        return fig, axs
    

#%%definitions

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

def mags_sum(m:np.ndarray, w:np.ndarray=None, axis:int=None):
    """
        - function to calculate the total magnitude of a set of magnitudes

        Parameters
        ----------
            - `m`
                - np.ndarray
                - 3d array of shape (nframes,xpix,ypix)
                    - nframes denotes the number of frames passed
                    - xpix is the number of pixels in x direction
                    - ypix is the number of pixels in y direction
                - contains magnitudes to add up
            - `w`
                - np.ndarray, optional
                - weight for each passed pixel
                    - for example some distance measure
                - has to be of shape `(1,*m.shape[1:])`
                - the default is `None`
                    - will be set to 1 for all elements in `m`
            - `axis`
                - int, optional
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
                - float
                - combined (weighted) magnitude
        
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

    return m_tot

def mags_contribution(
    m:Union[float,np.ndarray], m_cont:Union[float,np.ndarray],
    w:np.ndarray=None,
    ) -> Union[float,np.ndarray]:
    """
        - function that estimates the contribution in magnitude of target star (m) to a total magnitude
        
        Parameters
        ----------
            - `m`
                - float, np.ndarray
                - magnitude(s) of the target star(s)
                - if float
                    - will calculate and return the contribution of that one target star
                - if np.ndarray
                    - will calculate contribution of every single star to the `m_cont` contaminant stars
            - `m_cont`
                - float, np.ndarray
                - if float
                    - total magnitude to compare `m` to
                - if np.ndarray
                    - magnitudes to calculate total magnitude from on the fly
                        - will call `mags_sum(m_cont, w=w)`
                    - i.e. magnitudes of all contaminant stars
            - `w`
                - np.ndarray, optional
                - weights for each passed magnitude
                    - only applied if `m_cont` is a np.ndarray
                    - for example some distance measure
                - the default is `None`
                    - will be set to 1 for all elements in `m`


        Raises
        ------

        Returns
        -------
            - `p`
                - float, np.ndarray
                - fractional contribution of `m` to `m_cont`
                - float if `m` is float
                - np.ndarray if `m` is a np.ndarray
        
        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------
    """
    
    #calculate total contaminant magnitude if array of magnitudes is provided
    if isinstance(m_cont, np.ndarray):
        if len(m_cont) > 1:
            m_cont = mags_sum(m_cont, w=w)

    ffrac = mags2fluxes(m=m_cont, m_ref=m)
    p = 1/(1 + ffrac)

    return p

