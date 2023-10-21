
#%%imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Literal

from astroLuSt.monitoring import errorlogging as alme, formatting as almf
from astroLuSt.physics import photometry as alpp
from astroLuSt.visualization import plotting as alvp

#%%classes
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
        self.frames         = None          
        self.aperture_res   = np.empty((0,3))
        self.ring_res       = np.empty((0,4))
        self.aperture_masks = np.empty((0,0,0))
        self.ring_masks     = np.empty((0,0,0))
        

        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    mode={repr(self.mode)},\n'
            f'    store_ring_mask={repr(self.store_ring_masks)},\n'
            f'    store_aperture_masks={repr(self.store_aperture_masks)},\n'
            f'    verbose={self.verbose},\n'
            f')'
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
                            - element 0: number of frames
                            - element 1: posistion in x direction
                            - element 2: posistion in y direction
                            - element 3: flux/magnitude values

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
                            - element 0: number of frames
                            - element 1: posistion in x direction
                            - element 2: posistion in y direction
                            - element 3: flux/magnitude values

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
        self.aperture_masks = np.empty((0,*self.frames.shape[1:3]))
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
            self.sum_frame[:,:,2] = alpp.mags_sum(self.frames[:,:,:,2], axis=0)

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
        if self.frames is None:
            self.__check_frames_shape(frames)
            self.__store_frames(self.frames)

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
        rw_sky:np.ndarray,
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
        if self.frames is None:
            self.__check_frames_shape(frames)
            self.__store_frames(self.frames)
            
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
        x:np.ndarray,
        posx:float,
        posy:float,
        r_aperture:np.ndarray,
        rw_sky:np.ndarray,
        y:np.ndarray=None,
        test_aperture_kwargs:dict=None,
        test_background_kwargs:dict=None,
        ) -> None:
        """
            - method to fit the estimator
            
            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to apply estimator on
                    - i.e. frames of the timeseries
                - `posx`
                    - float
                    - position of aperture and sky-ring in x direction
                - `posy`
                    - float
                    - position of aperture and sky-ring in y direction
                - `r_aperture`
                    - np.ndarray
                    - test radii of the aperture
                    - will test every radius and calculate total flux/magnitude contained within that radius
                - `rw_sky`
                    - np.ndarray
                    - test specifications for sky-rings to test
                    - has to have shape `(nrings,2)`
                        - element 0: radii of the sky-rings
                        - element 1: widths of the sky-rings
                - `y`
                    - np.ndarray, optional
                    - y values
                    - just here for consistency
                    - not used in calculation
                    - the default is `None`
                - `test_aperture_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.test_aperture()`
                    - the default is `None`
                        - will be set to `dict()`
                - `test_background_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.test_background()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        #TODO: Add alternate background estimate options

        if test_aperture_kwargs is None:    test_aperture_kwargs    = dict()
        if test_background_kwargs is None:  test_background_kwargs  = dict()

        self.test_aperture(
            frames=x,
            posx=posx,
            posy=posy,
            r_aperture=r_aperture,
            **test_aperture_kwargs
        )

        self.test_background_skyring(
            frames=x,
            posx=posx,
            posy=posy,
            rw_sky=rw_sky,
            **test_background_kwargs
        )

        return
    
    def predict(self,
        ) -> None:
        """
            - NOT IMPLEMENTED
            - method to predict with the fitted estimator

            Parameters
            ----------
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

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
        ) -> None:
        """
            - method to fit the estimator and and predict with it

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to apply estimator on
                    - i.e. frames of the timeseries
                - `posx`
                    - float
                    - position of aperture and sky-ring in x direction
                - `posy`
                    - float
                    - position of aperture and sky-ring in y direction
                - `r_aperture`
                    - np.ndarray
                    - test radii of the aperture
                    - will test every radius and calculate total flux/magnitude contained within that radius
                - `rw_sky`
                    - np.ndarray
                    - test specifications for sky-rings to test
                    - has to have shape `(nrings,2)`
                        - element 0: radii of the sky-rings
                        - element 1: widths of the sky-rings
                - `y`
                    - np.ndarray, optional
                    - y values
                    - just here for consistency
                    - not used in calculation
                    - the default is `None`
                - `test_aperture_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.test_aperture()`
                    - the default is `None`
                        - will be set to `dict()`
                - `test_background_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.test_background()`
                    - the default is `None`
                        - will be set to `dict()`
                        
            Raises
            ------

            Returns
            -------

            Comments
            --------
        
        """

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
        # self.predict(**predict_kwargs)

        return

    def plot_result(self,
        plot_aperture_r:np.ndarray=None,
        plot_sky_rings_rw:np.ndarray=None,
        aperture_cmap:str=None,
        sky_rings_cmap:Union[str,mcolors.Colormap]=None,
        fig:Figure=None,
        sort_rings_apertures:bool=True,
        pcolormesh_kwargs:dict=None,
        plot_kwargs:dict=None,
        scatter_kwargs:dict=None,
        ) -> Union[Figure,plt.Axes]:
        """
            - mathod to visualize the result

            Parameters
            ----------
                - `plot_aperture_r`
                    - np.ndarray, optional
                    - radii to visualize the aperture for in the frame of total magnitudes/fluxes
                    - the default is `None`
                        - will be set to `[]`
                - `plot_sky_rings_rw`
                    - np.ndarray, optional
                    - radii-width combinations to visualize the sky-ring for in the frame of total magnitudes/fluxes
                    - has to have shape `(n2plot,2)`
                        - element 0: sky-ring radius
                        - emement 1: sky-ring width
                    - the default is `None`
                        - will be set to `[]`
                - `aperture_cmap`
                    - str, mcolors.Colormap, optional
                    - colormap to use for colorcoding apertures
                    - the default is `None`
                        - will be set to `autumn`
                - `sky_rings_cmap`
                    - str, mcolors.Colormap, optional
                    - colormap to use for colorcoding apertures
                    - the default is `None`
                        - will be set to `winter`
                - `fig`
                    - Figure, optional
                    - figure to plot into
                    - the default is `None`
                        - will create a new figure
                - `sort_rings_apertures`
                    - bool, optional
                    - whether to sort the passed rings and apertures before plotting
                    - recommended because then it is more likely that all requested apertures/sky-rings are visible
                        - if they are not sorted, it could be that a large aperture is plot on top of a smaller one, essentially covering it
                - `plot_kwargs`
                    - dict, optional
                    - kwargs to pass to ´ax.plot()`
                    - the default is `None`
                        - will be set to `dict()`
                - `scatter_kwargs`
                    - dict, optional
                    - kwargs to pass to ´ax.scatter()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - Figure
                    - created figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------
        """

        #default values
        if plot_aperture_r is None:   plot_aperture_r   = np.empty((0))
        if plot_sky_rings_rw is None: plot_sky_rings_rw = np.empty((0,2))
        if sky_rings_cmap is None:    sky_rings_cmap    = 'autumn'
        if aperture_cmap is None:     aperture_cmap     = 'winter'
        if pcolormesh_kwargs is None: pcolormesh_kwargs = dict()
        if plot_kwargs is None:       plot_kwargs       = dict(lw=1)
        if scatter_kwargs is None:    scatter_kwargs    = dict(s=5, cmap=sky_rings_cmap)

        #set some fixed kwargs
        scatter_kwargs['cmap'] = sky_rings_cmap     #to ensure that cmap of scatter and skyrings matches
        if 'cmap' not in pcolormesh_kwargs.keys():
            pcolormesh_kwargs['cmap'] = 'viridis' + '_r'*(self.mode=='mag')

        #kwargs of outline for star aperture plot
        outline_kwargs = plot_kwargs.copy()
        outline_kwargs['lw'] = plot_kwargs['lw']*4
        outline_kwargs['color'] = 'w'

        #sorting
        if sort_rings_apertures:
            sortidxs_skyring = np.argsort(plot_sky_rings_rw[:,1])
            plot_aperture_r   = np.sort(plot_aperture_r)[::-1]
            plot_sky_rings_rw = plot_sky_rings_rw[sortidxs_skyring][::-1]

        #plotting
        if fig is None: fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        #plot sum frame
        mesh = ax1.pcolormesh(self.sum_frame[:,:,0], self.sum_frame[:,:,1], self.sum_frame[:,:,2], **pcolormesh_kwargs)
        
        #plot some selected sky rings
        if len(plot_sky_rings_rw) > 0:
            colors_sky_ring = alvp.generate_colors(len(plot_sky_rings_rw), cmap=sky_rings_cmap)
            for idx, (rsr, wsr) in enumerate(plot_sky_rings_rw):

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
                            f'    Ignoring plotting of sky-ring for combination `(r,w)={rsr,wsr}`.\n'
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
        cbar1 = fig.colorbar(mesh, ax=ax1)
        cbar2 = fig.colorbar(sctr, ax=ax2)

        ax2.legend()

        #labelling
        cbar2.set_label('Sky Ring Width')

        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')
        ax2.set_xlabel('Radius')
        if self.mode == 'flux':
            cbar1.set_label('Total Flux (Pixel-Wise)')
            ax2.set_ylabel('Aperture Flux')
        elif self.mode == 'mag':
            ax2.invert_yaxis()
            cbar1.ax.invert_yaxis()
            cbar1.set_label('Total Magnitude (Pixel-Wise)')
            ax2.set_ylabel('Aperture Magnitude')

        axs = fig.axes
        
        return fig, axs
    