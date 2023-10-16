
#%%imports
import numpy as np

from typing import Union, Tuple, Callable

from astroLuSt.monitoring import errorlogging as alme

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

    def __init__(self,
        frames:np.ndarray,
        mode:Literal['flux','mag']=None,
        store_ring_masks:bool=True,
        store_aperture_masks:bool=True,
        verbose:int=0,
        ) ->  None:

        if len(frames.shape) == 4:
            self.frames                 = frames
        elif len(frames.shape) == 3:
            self.frames                 = np.expand_dims(frames,0)
        else:
            raise ValueError(f'`frames` has to be 4d but has shape {frames.shape}!')
        if mode is None:    self.mode   = 'flux'
        else:               self.mode   = mode
        self.store_ring_masks           = store_ring_masks
        self.store_aperture_masks       = store_aperture_masks
        self.verbose                    = verbose
        
        self.get_sum_frame()


        #infered attributes
        self.aperture_res   = np.empty((0,2))
        self.ring_res       = np.empty((0,3))
        self.aperture_masks = np.empty((0,*self.frames.shape[1:3]))
        self.ring_masks     = np.empty((0,*self.frames.shape[1:3]))
        

        pass

    def __repr__(self) -> str:
        return

    def __dict__(self) -> dict:
        return
    
    def get_sum_frame(self,
        ) -> np.ndarray:

        self.sum_frame = self.frames[0].copy()
        self.sum_frame[:,:,2] = np.nansum(self.frames[:,:,:,2], axis=0)

        """
        fig = plt.figure()
        m = plt.pcolormesh(self.sum_frame[:,:,-1])
        plt.colorbar(m)
        plt.show()
        """

        return self.sum_frame

    def get_star(self,
        posx:float, posy:float,
        r_star:np.ndarray,
        ):

        pos = np.array([posx,posy])

        for r in r_star:
            aperture_mask = (np.sqrt(np.sum((self.sum_frame[:,:,:2]-pos)**2, axis=2)) < r)

            aperture_flux = np.sum(self.sum_frame[aperture_mask,2])#/np.sum(aperture_mask)**2
            # print(aperture_flux)
            
            self.aperture_res = np.append(self.aperture_res, np.array([[r, aperture_flux]]), axis=0)

            # fig = plt.figure()
            # plt.pcolormesh(self.sum_frame[:,:,0], self.sum_frame[:,:,1], aperture_mask)
            # plt.scatter(*pos)
            # plt.show()



        return
    
    def get_background(self,
        posx:float, posy:float,
        r_sky:np.ndarray,
        w_sky:np.ndarray,
        ):

        pos = np.array([posx,posy])


        
        for r in r_sky:
            for w in w_sky:
    
                ring_mask = \
                     (np.sqrt(np.sum((self.sum_frame[:,:,:2]-pos)**2, axis=2)) >r)\
                    &(np.sqrt(np.sum((self.sum_frame[:,:,:2]-pos)**2, axis=2)) < r+w)

                ring_flux = np.sum(self.sum_frame[ring_mask,2])#/np.sum(ring_mask)**2
                # print(ring_flux)

                self.ring_res = np.append(self.ring_res, np.array([[r, w, ring_flux]]), axis=0)
                

                #store ring_mask for current w
                if self.store_ring_masks:
                    self.ring_masks = np.append(self.ring_masks, np.expand_dims(ring_mask,0), axis=0)

                # fig = plt.figure()
                # plt.pcolormesh(self.sum_frame[:,:,0], self.sum_frame[:,:,1], ring_mask)
                # plt.scatter(*pos)
                # plt.show()


        return
    
    
    def plot_result(self,
        plot_sky_rings_r:np.ndarray=None,
        plot_sky_rings_w:np.ndarray=None,
        plot_kwargs:dict=None,
        scatter_kwargs:dict=None,
        figure_kwargs:dict=None,
        ) -> Union[Figure,plt.Axes]:

        #default values
        if plot_sky_rings_r is None:  plot_sky_rings_r  = np.empty((0))
        if plot_sky_rings_w is None:  plot_sky_rings_w  = np.empty((0))
        if plot_kwargs is None:     plot_kwargs         = dict(lw=1)
        if scatter_kwargs is None:  scatter_kwargs      = dict(s=5, cmap='viridis')
        if figure_kwargs is None:   figure_kwargs       = dict(figsize=(10,5))

        #kwargs of outline for star aperture plot
        outline_kwargs = plot_kwargs.copy()
        outline_kwargs['lw'] = plot_kwargs['lw']*4
        outline_kwargs['color'] = 'w'


        fig = plt.figure(**figure_kwargs)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        #plot sum frame
        mesh = ax1.pcolormesh(self.sum_frame[:,:,0], self.sum_frame[:,:,1], self.sum_frame[:,:,2], zorder=0)
        
        #plot some selected sky rings
        for idx, (r, w) in enumerate(zip(plot_sky_rings_r, plot_sky_rings_w)):
            br = (r==self.ring_res[:,0])
            bw = (w==self.ring_res[:,1])
            try:
                ax1.contour(self.sum_frame[:,:,0], self.sum_frame[:,:,1], self.ring_masks[bw&br][0], levels=0, colors='r', zorder=2)
            except IndexError as i:
                alme.LogErrors().print_exc(
                    e=i,
                    prefix=f'{self.__class__.__name__}.plot_results()'
                )
                


        #plot star aperture and sky ring
        ax2.plot(self.aperture_res[:,0], self.aperture_res[:,1], label=None,            **outline_kwargs)
        ax2.plot(self.aperture_res[:,0], self.aperture_res[:,1], label='Star Aperture', **plot_kwargs)
        sctr = ax2.scatter(self.ring_res[:,:1], self.ring_res[:,2], c=self.ring_res[:,1], label='Sky Ring', **scatter_kwargs)

        #show which sky rings are plotted
        ax2

        #add colorbars
        cmap1 = fig.colorbar(mesh, ax=ax1)
        cmap2 = fig.colorbar(sctr, ax=ax2)
        cmap2.set_label('Sky Ring Width')

        ax2.legend()


        #labelling
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')

        ax2.set_xlabel('Radius')
        if self.mode == 'flux':
            ax2.set_ylabel('Aperture Flux')
        elif self.mode == 'mag':
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

