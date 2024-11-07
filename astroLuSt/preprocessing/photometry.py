#TODO: `Aperture`: Implement `water_shed_mask()`
#TODO: `BestAperture.fit()`: add alternate background estimate options


#%%imports
from joblib.parallel import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib.colors as mcolors
from matplotlib import patheffects
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Literal, Tuple, Callable, Any, List
import warnings

from astroLuSt.monitoring import errorlogging as alme, formatting as almf
from astroLuSt.physics import photometry as alpp
from astroLuSt.visualization import plotting as alvp

#%%classes
class Aperture:
    """
        - class to generate a variety of aperture masks

        Attributes
        ----------
            - `size`
                - `Tuple[int,int]`
                - size of the aperture in pixels
                - will generate frame (`np.ndarray`) with shape = `size`
            - `npixels`
                - `float`, optional
                - to how many pixels the aperture shall be normalized to
                    - i.e. sum over all aperture-mask elements will add up to this value
                - if differential photometry by means of a sky-ring is used, it is suggested to normalize the aperture to at least 2 pixels
                    - i.e. set `npixels >= 2`
                - the default is `None`
                    - no normalization
            - `position`
                - `np.ndarray[float,float]`, optional
                - position of the aperture on the frame relative to the frame's center
                - the default is `None`
                    - will be set to `np.array([0,0])`
                    - right in the center of the frame
            - `outside`
                - `float`, optional
                - value to set pixels outside of the aperture to
                - common choices
                    - `np.nan`
                    - `0.0`
                - the default is `0.0`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
        
        Methods
        -------
            - `get_coordinates()`
            - `post_process()`
            - `lp_aperture()`
            - `rect_aperture()`
            - `gauss_aperture()`
            - `lorentz_aperture()`
            - `plot_result()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------

    """

    def __init__(self,
        size:Tuple[int,int],
        npixels:float=None,
        position:np.ndarray[float,float]=None,
        outside:float=0.0,
        verbose:int=0,
        ) -> None:
        
        self.size       = (size[1],size[0])
        self.npixels    = npixels
        if position is None:    self.position = np.array([0,0])
        else:                   self.position = np.array(position)
        self.outside    = outside
        self.verbose    = verbose

        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    size={repr(self.size)},\n'
            f'    npixels={repr(self.npixels)},\n'
            f'    position={repr(self.position)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f'    outside={repr(self.outside)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def get_coordinates(self
        ) -> np.ndarray:
        """
            - method to generate a grid of coordinates used to define the aperture mask

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `coords`
                    - `np.ndarray`
                    - grid of coordinates of shape `(self.size[0],self.size[1],2)` 

            Comments
            --------
        """
        
        size = self.size

        #get coordinates
        apxx, apyy = np.meshgrid(
            np.linspace(-size[0]/2,size[0]/2,size[0]),
            np.linspace(-size[1]/2,size[1]/2,size[1]),
        )
        coords = np.append(apxx.reshape(size[1], size[0], 1), apyy.reshape(size[1], size[0], 1), axis=2)

        return coords

    def post_process(self,
        aperture:np.ndarray,
        npixels:float=None,
        outside:float=None,
        ) -> np.ndarray:
        """
            - function to execute postprocessing on the generated aperture
                - i.e.
                    - normalizing to some pixel value
                    - setting out-of-aperture values

            Parameters
            ----------
                - `aperture`
                    - `np.ndarray`
                    - generated aperture to be postprocessed
                - `npixels`
                    - `float`, optional
                    - to how many pixels the aperture shall be normalized to
                        - i.e. sum over all aperture-mask elements will add up to this value
                    - if differential photometry by means of a sky-ring is used, it is suggested to normalize the aperture to at least 2 pixels
                        - i.e. set `npixels >= 2`
                    - overrides `self.npixels`
                    - the default is `None`
                        - will fall back to `self.npixels`
                - `outside`
                    - `float`, optional
                    - value to set pixels outside of the aperture to
                    - common choices
                        - `np.nan`
                        - `0.0`
                    - overrides `self.outside`
                    - the default is `None`
                        - will fall back to `self.outside`

            Raises
            ------

            Returns
            -------
                - `aperture`
                    - `np.ndarray`
                    - postprocessed version of `aperture`


            Comments
            --------  
        """

        #default parameters
        if npixels is None: npixels = self.npixels
        if outside is None: outside = self.outside

        #normalize aperture
        if npixels is not None:
            aperture = aperture/np.sum(aperture) * npixels

        #set out-of-aperture value
        aperture[(aperture<=0)] = self.outside

        return aperture

    def lp_aperture(self,
        radius:float,
        p:float=2,
        poly_coeffs:np.ndarray=None,
        radius_inner:float=0,
        ) -> np.ndarray:
        """
            - method to generate apertures based on the L_p-norms
            - will generate mask with maximum value of 1

            Parameters
            ----------
                - `radius`
                    - `float`
                    - radius of the aperture in a L_p-norm sense
                - `p`
                    - `int`, optional
                    - p-parameter in the L_p norm
                    - will be passed to `np.linalg.norm()` as `ord`
                        - `np.linalg.norm(..., ord=p, ...)`
                    - the default is `2`
                        - L2-norm
                        - circular aperture
                - `poly_coeffs
                    - `np.ndarray`, optional
                    - polynomial coefficients to describe the radial aperture falloff/gradient
                    - will be passed to `np.polyval()`
                    - the default is `None`
                        - will be set to `np.array([1])
                        - constant (step-function)
                - `radius_inner`
                    - `float`, optional
                    - inner radius of sky ring
                    - will create a sky-ring (donut-like shape) with
                        - inner radius = `inner_radius`
                        - outer radius = `radius`
                    - the default is `0`
                        - creates standard aperture

            Raises
            ------

            Returns
            -------
                - `aperture`
                    - `np.ndarray`
                    - generated aperture mask
                    - has shape `(self.size[0],self.size[1])`

            Comments
            --------
        """
        
        #get position and coordinates
        position = self.position
        npixels = self.npixels
        outside = self.outside
        coords = self.get_coordinates()

        #default parameters
        if poly_coeffs is None: poly_coeffs = np.array([1])

        #generate mask (sky ring for nonzero radius_inner)
        abs_coords = np.linalg.norm(coords-position.reshape(1,1,-1), ord=p, axis=2)
        aperture = ((radius_inner <= abs_coords)&(abs_coords <= radius)).astype(np.float64)


        #polynomial radial gradient
        poly = np.polyval(poly_coeffs, abs_coords)
        aperture *= poly
        
        #postprocess aperture
        aperture = self.post_process(aperture=aperture, npixels=npixels, outside=outside)
        
        return aperture
    
    def rect_aperture(self,
        width:float, height:float,
        width_inner:float=0.0, height_inner:float=0.0,
        ) -> np.ndarray:
        """
            - method to generate rectangular apertures
            - aperture will be mask of zeros and ones scaled by `self.npixels`

            Parameters
            ----------
                - `width`
                    - `float`
                    - width of the aperture
                - `height`
                    - `float`
                    - height of the aperture
                - `width_inner`
                    - `float`, optional
                    - width of the inner bound of the sky ring
                    - will create a sky-ring (donut-like shape) with
                        - inner width = `inner_width`
                        - outer width = `width`
                    - the default is `0`
                        - creates standard aperture
                - `height_inner`
                    - `float`, optional
                    - height of the inner bound of the sky ring
                    - will create a sky-ring (donut-like shape) with
                        - inner height = `inner_height`
                        - outer height = `height`
                    - the default is `0`
                        - creates standard aperture

            Raises
            ------

            Returns
            -------
                - `aperture`
                    - `np.ndarray`
                    - generated aperture mask
                    - has shape `(self.size[0],self.size[1])`

            Comments
            --------
        """
        #get position and coordinates
        position = self.position
        npixels = self.npixels
        outside = self.outside
        coords = self.get_coordinates()

        #generate mask
        aperture = (coords - position.reshape(1,1,-1))
        
        ##outer aperture
        aperture_outer = (-np.array([width,height])/2<=aperture)&(aperture<=np.array([width,height])/2)
        aperture_outer = aperture_outer[:,:,0]&aperture_outer[:,:,1]
        
        ##inner aperture
        aperture_inner = (-np.array([width_inner,height_inner])/2>=aperture)|(aperture>=np.array([width_inner,height_inner])/2)
        aperture_inner = aperture_inner[:,:,0]|aperture_inner[:,:,1]
        
        ##combine
        aperture = aperture_inner & aperture_outer
        aperture = aperture.astype(np.float64)
        
        #postprocess aperture
        aperture = self.post_process(aperture=aperture, npixels=npixels, outside=outside)

        return aperture

    def gauss_aperture(self,
        radius:float=np.inf,
        p:float=2,
        covariance:Union[float,np.ndarray]=1,
        lp:bool=False,
        radius_inner:float=0.0,
        ) -> np.ndarray:
        """
            - method to generate apertures based on gaussian distributions
            - returns aperture normalized to `self.npixels`
                - normalized to `1` in case `self.npixels` is `None`

            Parameters
            ----------
                - `radius`
                    - `float`
                    - constraint space for the aperture
                    - anything outside `radius` will be set to zero
                    - the default is `np.inf`
                        - unconstrained
                - `p`
                    - `int`, optional
                    - p-parameter in the L_p norm
                    - will be passed to `np.linalg.norm()` as `ord`
                        - `np.linalg.norm(..., ord=p, ...)`
                    - utilized in
                        - computing the contraint space
                            - i.e. everything outside `radius` w.r.t. the L-p norm will be set to zero
                        - as norm in the exponent of the gaussian if `lp==True`
                            - will use L-p norm at expense of utilizing a covariance matrix
                            - useful to generate balls with gaussian-like decaying mask-values
                    - the default is `2`
                        - L2-norm
                        - circular aperture
                - `covariance`
                    - `float`, `np.ndarray`, optional
                    - covariance matrix in the exponent of the 2d gaussian
                    - has to have shape `(2,2)`
                    - if `float`
                        - will be interpreted as matrix with diagnoal values set to `covariance`
                    - otherwise
                        - will be interpreted as the covariance matrix
                    - the default is `1`
                - `lp`
                    - `bool`, optional
                    - whether to use the L-p norm in the exponent of the gaussian instead of the standard expression
                    - the defaul is `False`
                - `radius_inner`
                    - `float`, optional
                    - inner radius of sky ring
                    - will create a sky-ring (donut-like shape) with
                        - inner radius = `inner_radius`
                        - outer radius = `radius`
                    - only applied if `p!=0`
                    - the default is `0`
                        - creates standard aperture                 

            Raises
            ------
                - `TypeError`
                    - in case `covariance` is a `np.ndarray` and `lp==True`

            Returns
            -------
                - `aperture`
                    - `np.ndarray`
                    - generated aperture mask
                    - has shape `(self.size[0],self.size[1])`

            Comments
            --------
        """
        #get position and coordinates
        position = self.position
        npixels = self.npixels
        outside = self.outside
        coords = self.get_coordinates()        

        if isinstance(covariance, np.ndarray) and lp:
            raise TypeError((
                f'if `lp==True` `covariance` has to be a number!', 
            ))
        elif not isinstance(covariance, np.ndarray) and not lp:
            covariance = np.eye(2)*covariance
        
        #generate aperture (no constraints)
        if lp:
            exp = np.linalg.norm(coords-position.reshape(1,1,-1), ord=p, axis=-1)/covariance
        else:
            exp = np.einsum('ijk,kl,ijl -> ij', coords-position.reshape(1,1,-1),np.linalg.inv(covariance),coords-position.reshape(1,1,-1))
        aperture = np.exp(-exp/2)

        #constrain via L_p norm
        if p != 0:
            lp_mask_outer = (np.linalg.norm(coords-position.reshape(1,1,-1), ord=p, axis=-1) <= radius)         #outer bound
            lp_mask_inner = (radius_inner <= np.linalg.norm(coords-position.reshape(1,1,-1), ord=p, axis=-1))   #inner bound
            lp_mask = lp_mask_inner&lp_mask_outer
            aperture[~lp_mask] = 0


        #postprocess aperture
        aperture = self.post_process(aperture=aperture, npixels=npixels, outside=outside)

        return aperture
    
    def lorentz_aperture(self,
        fwhm:Union[float,np.ndarray],
        radius:float=np.inf,
        p:float=0,
        radius_inner:float=0.0,
        ) -> np.ndarray:
        """
            - method to generate apertures following a 2d Lorentzian-profile
            - returns aperture normalized to `self.npixels`
                - normalized to `1` in case `self.npixels` is `None`

            Parameters
            ----------
                - `fwhm`
                    - `float`, `np.ndarray`
                    - full-width-half-maximum of the lorentzian
                    - determines spread of the distribution in x1 and x2 directions
                    - if `np.ndarray`
                        - has to have shape `(2,)`
                    - if `float`
                        - will be interpreted as fwhm for all directions
                - `radius`
                    - `float`
                    - constraint space for the aperture
                    - anything outside `radius` will be set to zero
                    - the default is `np.inf`
                        - unconstrained
                - `p`
                    - `int`, optional
                    - p-parameter in the L_p norm
                    - will be passed to `np.linalg.norm()` as `ord`
                        - `np.linalg.norm(..., ord=p, ...)`
                    - the default is `2`
                        - L2-norm
                        - circular aperture
                - `radius_inner`
                    - `float`, optional
                    - inner radius of sky ring
                    - will create a sky-ring (donut-like shape) with
                        - inner radius = `inner_radius`
                        - outer radius = `radius`
                    - only applied if `p!=0`
                    - the default is `0`
                        - creates standard aperture

            Raises
            ------

            Returns
            -------
                - `aperture`
                    - `np.ndarray`
                    - generated aperture mask
                    - has shape `(self.size[0],self.size[1])`

            Comments
            --------
        """        
        #get position and coordinates
        position = self.position
        npixels = self.npixels
        outside = self.outside
        coords = self.get_coordinates()

        #compute 2D lorentzian profile
        x = (coords-position.reshape(1,1,-1))/(fwhm/2)
        aperture = 1/(1+x**2)   #lorentz profile
        aperture = np.prod(aperture, axis=-1)   #combine dimensions

        #constrain via L_p norm
        if p != 0:
            lp_mask_outer = (np.linalg.norm(coords-position.reshape(1,1,-1), ord=p, axis=-1) <= radius)         #outer bound
            lp_mask_inner = (radius_inner <= np.linalg.norm(coords-position.reshape(1,1,-1), ord=p, axis=-1))   #inner bound
            lp_mask = lp_mask_inner&lp_mask_outer
            aperture[~lp_mask] = 0

        #postprocess aperture
        aperture = self.post_process(aperture=aperture, npixels=npixels, outside=outside)

        return aperture

    def water_shed_mask(self,
        X:np.ndarray
        ) -> np.ndarray:
        """
            - algorithm to separate source from nearby stars

            Parameters
            ----------

            Raiese
            ------

            Returns
            -------

            Comments
            --------
        """
        warnings.warn("Not implemented yet", UserWarning)
        return    

    def plot_result(self,
        X:np.ndarray,
        ax:plt.Axes=None,
        contour_path_effects:list=None,
        pcolormesh_kwargs:dict=None,
        contour_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the generated aperture

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - aperture to plot
                    - has to have shape `(self.size[0], self.size[1]`)
                - `ax`
                    - `plt.Axes`, optional
                    - axes to plot into
                    - the default is `None`
                        - will generate new figure containing one axis
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict()`
                - `contour_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.contour()`
                    - the default is `None`
                        - will be set to `dict(colors='r', levels=0)`
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `matplotlib.figure.Figure`
                    - created figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`

            Comments
            --------

        """

        #default parameters
        if contour_path_effects is None: 
            contour_path_effects =  [
                patheffects.Stroke(linewidth=5, foreground='white'),
                patheffects.Normal(),
            ]
        if pcolormesh_kwargs is None:   pcolormesh_kwargs   = dict()
        if contour_kwargs is None:      contour_kwargs      = dict()
        if 'colors' not in contour_kwargs.keys(): contour_kwargs['colors'] = 'r'
        if 'levels' not in contour_kwargs.keys(): contour_kwargs['levels'] = 0


        #get coords
        coords = self.get_coordinates()

        if ax is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
            ax1 = ax
        ax1.set_aspect('equal')
        
        #plotting
        mesh = ax1.pcolormesh(coords[:,:,0], coords[:,:,1], X, **pcolormesh_kwargs)
        cont = ax1.contour(coords[:,:,0], coords[:,:,1], X, **contour_kwargs)
        cont.set_path_effects(contour_path_effects)

        cbar = fig.colorbar(mesh, ax=ax1)
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')
        cbar.set_label('Aperture Transmissivity')

        axs = fig.axes


        return fig, axs

class AperturePhotometry:
    """
        - class to apply aperture photometry on series of frames
        - transforms frames to lightcurves (LCs)

        Attributes
        ----------
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `reduce_func_nansum()`
            - `lc_from_frames()`
            - `fit()`
            - `transform()`
            - `fit_transform()`
            - `plot_result()`
        
        Dependencies
        ------------
            - `joblib`
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------

    """
    def __init__(self,
        verbose:int=0,
        ) -> None:
        
        self.verbose    = verbose

        self.X_transformed = None

        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
        
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def reduce_func_nansum(self,
        p:np.ndarray, p_e:np.ndarray=None,
        **kwargs
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to reduce frames to fluxes
            - based on `np.nansum()`

            Parameters
            ----------
                - `p`
                    - `np.ndarray`
                    - has shape `(nobservations, npixelsx, npixelsy)`
                    - pixel flux weighted by aperture-mask
                - `p_e`
                    - `np.ndarray`, optional
                    - has shape `(nobservations, npixelsx, npixelsy)`
                    - pixel flux error weighted by aperture-mask
                    - the default is `None`
                        - no errors
            
            Raises
            ------

            Returns
            -------
                - `flux`
                    - `np.ndarray`
                    - has shape `(nobservations, 1)`
                    - flux values representing each observation
                - `flux_e`
                    - `np.ndarray`
                    - has shape `(nobservations, 1)`
                    - errors corresponding to `flux`
                    - if `p_e` is `None`
                        - returns array filled with `np.nan`

            Comments
            --------
        """
        flux = np.nansum(p, axis=(1,2))
        if p_e is not None:
            flux_e = np.sqrt(np.nansum(p_e**2, axis=(1,2)))
        else:
            flux_e = np.zeros_like(flux)*np.nan
        return flux, flux_e

    def fit(self,
        X:np.ndarray, X_e:np.ndarray=None,
        aperture:Union[float,np.ndarray]=2.0,
        reduce_func:Callable[[np.ndarray,np.ndarray,Any],Tuple[np.ndarray,np.ndarray]]=None,
        fluxvars:Callable[[np.ndarray,Any],np.ndarray]=None,
        verbose:int=None,
        reduce_func_kwargs:dict=None,
        fluxvars_kwargs:dict=None,
        *args, **kwargs,
        ) -> None:
        """
            - method to generate a lightcurve from a series of frames

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - series of frames to generate lightcurve from
                    - has shape `(nframes, npixelsx, npixelsy)`
                - `X_e`
                    - `np.ndarray`, optional
                    - series of frames containing errors for each pixel corresponding to `X`
                    - has shape `(nframes, npixelsx, npixelsy)`
                        - same as `X`
                    - the default is `None`
                        - no errors considered
                - `aperture`
                    - `float`, `np.ndarray`, optional
                    - aperture to use for the creation of the lightcurve
                    - if `float`
                        - will be interpreted as the radius of a circular aperture
                        - passed as `radius` to `astroLuSt.Aperture().lp_aperture()`
                            - other than `radius` will be called with default parameters
                    - if `np.ndarray`
                        - has to have the shape `X.shape[1:3]`
                        - will be interpreted as the actual aperture mask to be used
                    - the default is `2.0`
                - `reduce_func`
                    - `Callable(np.ndarray,Any) -> Tuple[np.ndarray,np.ndarray]`, optional
                    - function to use for reducing
                        - `X*aperture` to a 1-d series of fluxes
                        - `X_e*aperture` to a 1-d series of flux errors
                    - has to take at least two arguments
                        - `X*aperture`
                        - `X_e*aperture`
                    - has to return two 1-d arrays
                        - `X*aperture` reduced to a 1-d series of fluxes
                        - `X_e*aperture` reduced to a 1-d series of flux errors corresponding to `X`
                    - common options are
                        - `self.reduce_func_nansum`
                            - total flux in aperture
                    - the default is `None`
                        - will be set `self.reduce_func_nansum()`
                - `fluxvars`
                    - `Callable(np.ndarray,np.ndarray,Any) -> Tuple[np.ndarray,np.ndarray]`, optional
                    - function to use for calculating variations of the extracted flux (and corresponding errors)
                    - useful for i.e.
                        - normalizing the flux
                        - modifying other parameters within the parallelized extraction loop
                            - calculating custom quality flags
                            - normalizing fluxes w.r.t. another quantity (i.e., sectorwise)
                                - one has to pass the sector-mask via `fluxvars_kwargs` in that case
                            - converting times to phases
                                - one has to pass the times via `fluxvars_kwargs` in that case
                    - has to take 2 arguments
                        - `x`
                            - `np.ndarray`
                            - array of flux values (LC)
                        - `x_e`
                            - `np.ndarray`
                            - array of flux errors (LC)
                    - has to return two `np.ndarray`s (flux variations, errors of flux variations)
                        - have to have shape `(len(X),nfluxvars)`
                            - `nfluxvars` contains the number of variations that have been calculated
                    - an example for returning two variations (globally normalized and sector-wise normalized)
                        >>> def sectornorm(X:np.ndarray, X_e:np.ndarray, sector:np.ndarray):
                        >>>     X_norm = np.empty((X.shape[0],2))
                        >>>     X_norm_e = np.zeros_like(X_norm)*np.nan
                        >>>     X_norm[:,0] = X/np.nanmedian(X)
                        >>>     X_norm_e[:,0] = X_e/np.nanmedian(X)
                        >>>     for s in np.unique(sector):
                        >>>         sbool = (sector==s)
                        >>>         X_norm[sbool,1] = X[sbool] / np.median(X[sbool])
                        >>>         X_norm_e[sbool,1] = X_e[sbool] / np.median(X[sbool])
                        >>>     return X_norm, X_norm_e
                    - if you want to return only the raw flux pass `lambda x, x_e, **kwargs:(None, None)`
                    - the default is `None`
                        - will be set to `lambda x, x_e, **kwargs: ((x/np.nanmedian(x)).reshape(-1,1), (x_e/np.nanmedian(x)).reshape(-1,1))`
                            - i.e. global normalization by the median
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`
                - `reduce_func_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `reduce_func`
                    - the default is `None`
                        - will be set to `dict(axis=(1,2)`)
                - `fluxvars_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `fluxvars()`
                    - useful for i.e. including additional quantities in the normalization
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #default parameters
        if X_e is None:                         X_e                 = np.zeros_like(X)*np.nan
        if fluxvars is None:                    fluxvars            = lambda x, x_e, **kwargs: ((x/np.nanmedian(x)).reshape(-1,1), (x_e/np.nanmedian(x)).reshape(-1,1))    #global median
        if reduce_func is None:                 reduce_func         = self.reduce_func_nansum
        if verbose is None:                     verbose             = self.verbose
        if reduce_func_kwargs is None:          reduce_func_kwargs  = dict(axis=(1,2))
        if fluxvars_kwargs is None:             fluxvars_kwargs     = dict()

        if not isinstance(aperture, np.ndarray):
            #generate aperture
            AP = Aperture(size=X.shape[1:])
            aperture = AP.lp_aperture(radius=aperture)

        #calculate flux timeseries based on aperture and frame-series
        flux, flux_e = reduce_func(
            X*aperture, X_e*aperture,
            **reduce_func_kwargs,
        )
        flux_norm, flux_norm_e  = fluxvars(flux.copy(), flux_e.copy(), **fluxvars_kwargs)

        #reshape correctly
        flux    = flux.reshape(-1,1)
        flux_e  = flux_e.reshape(-1,1)

        #add custom computations        
        if flux_norm is not None:
            flux    = np.append(flux,   flux_norm,  axis=1)
            flux_e  = np.append(flux_e, flux_norm_e,axis=1)

        self.X_transformed      = flux
        self.X_transformed_e    = flux_e
        self.aperture          = aperture

        return
   
    def transform(self,
        verbose:int=None,
        *args, **kwargs,
        ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
            - method to transform a given set of samples `X` to lightcurves
            - does not execute computation but only returns the results from fitting

            Parameters
            ----------
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`

            Raises
            ------

            Returns
            -------
                - `X_transformed`
                    - `np.ndarray`
                    - fluxes constituting the lightcurve resulting from the series of frames `X`
                    - has shape `(len(X),nfeatures)`
                        - the first feature is hereby the unnormalized extracted flux
                        - all other features are the ones computed via `fluxvars()`
                - `X_transformed_e`
                    - `np.ndarray`
                    - errors corresponding to `X_transformed`
                    - has shape `(len(X),nfeatures)` (same as `X_transformed`)
                        - the first feature is hereby the error corresponding to the unnormalized extracted flux
                        - all other features are the ones computed via `fluxvars()`
                - `aperture`
                    - `np.ndarray`
                    - the ultimately used aperture for the creation of LC                    

            Comments
            --------                    
        
        """
        if verbose is None:             verbose             = self.verbose

        if self.X_transformed is None:
            raise ValueError((
                f'You have to call `{self.__class__.__name__}.fit()` before being able to transform.'
                f'    Another option is to call `{self.__class__.__name__}.fit_transform()` from the start.'
            ))

        X_transformed   = self.X_transformed
        X_transformed_e = self.X_transformed_e
        aperture        = self.aperture

        return X_transformed, X_transformed_e, aperture
    
    def fit_transform(self,
        X:List[np.ndarray], X_e:List[np.ndarray]=None,
        verbose:int=None,
        fit_kwargs:dict=None,
        *args, **kwargs,
        ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
            - method to transform a given set of samples `X` to lightcurves
            - fits the transformer and transforms the data in one go

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - series of frames to generate lightcurve from
                    - has shape `(nframes, npixelsx, npixelsy)`
                - `X_e`
                    - `np.ndarray`, optional
                    - series of frames containing errors for each pixel corresponding to `X`
                    - has shape `(nframes, npixelsx, npixelsy)`
                        - same as `X`
                    - the default is `None`
                        - no errors considered                   
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`
                - `fit_kwargs`
                    - `dict`, optional
                    -  kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `X_transformed`
                    - `np.ndarray`
                    - fluxes constituting the lightcurve resulting from the series of frames `X`
                    - has shape `(len(X),nfeatures)`
                        - the first feature is hereby the unnormalized extracted flux
                        - all other features are the ones computed via `fluxvars()`
                - `X_transformed_e`
                    - `np.ndarray`
                    - errors corresponding to `X_transformed`
                    - has shape `(len(X),nfeatures)` (same as `X_transformed`)
                        - the first feature is hereby the error corresponding to the unnormalized extracted flux
                        - all other features are the ones computed via `fluxvars()`
                - `aperture`
                    - `np.ndarray`
                    - the ultimately used aperture for the creation of LC   

            Comments
            --------                    
        
        """

        if verbose is None:             verbose             = self.verbose
        if fit_kwargs is None:          fit_kwargs          = dict()

        self.fit(X=X, X_e=X_e, verbose=verbose, **fit_kwargs)
        X_transformed, X_transformed_e, apertures = self.transform(X=X, verbose=verbose)

        return X_transformed, X_transformed_e, apertures

    def plot_result(self,
        X:np.ndarray, X_e:np.ndarray=None,
        x_vals:np.ndarray=None, X_in:np.ndarray=None, aperture:np.ndarray=None,
        fig=None,
        animate:bool=True,
        verbose:int=None,
        pcolormesh_kwargs:dict=None,
        pcolormesh_ap_kwargs:dict=None,
        sctr_kwargs:dict=None,
        errorbar_kwargs:dict=None,
        func_animation_kwargs:dict=None,
        ) -> Tuple[Figure, plt.Axes, manimation.FuncAnimation]:
        """
            - method to visualize the generated result

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - output `X_transformed` of the transformer
                    - i.e., the transformed version of `X_in`
                    - has shape `(nsamples, nfeatures)`
                        - `nsamples`
                            - number of datapoints
                        - `nfeatures`
                            - number of generated transformation
                            - will create one panel for each feature
                            - the standard output of `self.transform()` has `nfeatures=2`
                - `X_e`
                    - `np.ndarray`, optional
                    - errors corresponding to `X`
                    - has shape `(nsamples, nfeatures)` (same as `X`)
                        - `nsamples`
                            - number of datapoints
                        - `nfeatures`
                            - number of generated transformation
                            - will create one panel for each feature
                            - the standard output of `self.transform()` has `nfeatures=2`
                    - the default is `None`
                        - no errors visualized
                - `x_vals`
                    - `np.ndarray`, optional
                    - has to have same length as `X`
                    - series of datapoints to use as x values for plotting
                    - typical choices are
                        - `times`
                        - `phases`
                        - `periods`
                    - the default is `None`
                        - interpreted as `np.range(X.shape[0])`
                - `X_in`
                    - `np.ndarray`, optional
                    - contains input passed to `self.fit()` to obtain `X_transformed`
                        - i.e. a series of frames
                    - has to be 3D
                    - if passed
                        - will plot the frame(s) as well
                    - the default is `None`
                        - no frames will be plotted
                - `aperture`
                    - `np.ndarray`, optional
                    - aperture used to generate `X_transformed` from `X_in`
                    - if passed will overplot `aperture` over frame(s)
                    - the default is `None`
                        - no aperture plotted
                - `fig`
                    - `matplotib.figure.Figure`, optional
                    - figure to plot into
                    - the default is `None`
                        - will generate a new figure
                - `animate`
                    - `bool`, optional
                    - whether to create an animation out of the passed quantities to plot
                    - the default is `True`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`
                - pcolormesh_kwargs
                    - `dict`, optional
                    - kwargs to be passed to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict(vmin=np.nanmin(X_in), vmax=np.nanmax(X_in))`
                - pcolormesh_ap_kwargs
                    - `dict`, optional
                    - kwargs to be passed to `ax.pcolormesh()` called to plot the aperture
                    - NOTE, that the parameters will be set using `ax.pcolormesh().set_<parameter>(value)`
                    - the default is `None`
                        - will be set to `dict(alpha=aperture, zorder=2, edgecolor='r', facecolor='none')`
                - sctr_kwargs
                    - `dict`, optional
                    - kwargs to be passed to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict(cmap='nipy_spectral', c=np.ones(X.shape[0]))`
                            - `'c'` set by default to ensure same colors of errorbars and datapoints
                - errorbar_kwargs
                    - `dict`, optional
                    - kwargs to be passed to `ax.errorbar()`
                    - the default is `None`
                        - will be set to `dict(ls='', marker=None, zorder=-1)`
                        - `'ecolor'` will be inherited from the datapoints in `ax.scatter()`, if not provided
                - func_animation_kwargs
                    - `dict`, optional
                    - kwargs to be passed to `matplotlib.animation.FuncAnimation()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `matplotlib.figure.Figure`
                    - generated figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
                - `anim`
                    - `matplotlib.animation.FuncAnimation`
                    - generated animation
                    - returns `None` if `animate==False`

            Comments
            --------
        """


        def update(
            frame:int,
            ) -> None:
            
            mesh.update(dict(array=X_in[frame]))
            for vline in vlines: vline.set_xdata([x_vals[frame]])

            return

        #default parameters
        if x_vals is None:                          x_vals                              = np.arange(X.shape[0])
        if verbose is None:                         verbose                             = self.verbose
        if pcolormesh_kwargs is None:               pcolormesh_kwargs                   = dict()
        if 'vmin' not in pcolormesh_kwargs:         pcolormesh_kwargs['vmin']           = np.nanmin(X_in)
        if 'vmax' not in pcolormesh_kwargs:         pcolormesh_kwargs['vmax']           = np.nanmax(X_in)
        if pcolormesh_ap_kwargs is None:            pcolormesh_ap_kwargs                = dict(alpha=aperture, zorder=2, edgecolor='r', facecolor='none')
        if 'alpha' not in pcolormesh_ap_kwargs:     pcolormesh_ap_kwargs['alpha']       = aperture
        if 'zorder' not in pcolormesh_ap_kwargs:    pcolormesh_ap_kwargs['zorder']      = 2
        if 'edgecolor' not in pcolormesh_ap_kwargs: pcolormesh_ap_kwargs['edgecolor']   = 'r'
        if 'facecolor' not in pcolormesh_ap_kwargs: pcolormesh_ap_kwargs['facecolor']   = 'none'
        if sctr_kwargs is None:                     sctr_kwargs                         = dict()
        if 'cmap' not in sctr_kwargs.keys():        sctr_kwargs['cmap']                 = 'nipy_spectral'
        if 'c' not in sctr_kwargs.keys():           sctr_kwargs['c']                    = np.ones(X.shape[0])   #set default color to ensure same coloring of datapoints and error bars
        if errorbar_kwargs is None:                 errorbar_kwargs                     = dict()
        if 'ls' not in errorbar_kwargs.keys():      errorbar_kwargs['ls']               = ''
        if 'marker' not in errorbar_kwargs.keys():  errorbar_kwargs['marker']           = None
        if 'zorder' not in errorbar_kwargs.keys():  errorbar_kwargs['zorder']           = -1
        if func_animation_kwargs is None:           func_animation_kwargs               = dict()
        
        #plotting
        if fig is None:
            fig = plt.figure()
        if X_in is not None or aperture is not None:
            ax2 = fig.add_subplot(121)
            ax2.set_aspect('equal')
            ax2.set_xlabel(r'Pixel')
            ax2.set_ylabel(r'Pixel')
        lc_axs = []

        #plot frame
        if X_in is not None:
            mesh    = ax2.pcolormesh(X_in[0], **pcolormesh_kwargs)
            cbar = fig.colorbar(mesh, ax=ax2)
            cbar.set_label('Color')
        if aperture is not None:
            #plot aperture
            mesh_ap = ax2.pcolormesh(aperture)
            #adjust parameters
            for param, value in pcolormesh_ap_kwargs.items():
                setter = getattr(mesh_ap, f'set_{param}')
                setter(value)
            ax2.plot(np.nan, np.nan, '-r', label='Aperture')
            ax2.legend(framealpha=0.5)

        for idx, xi in enumerate(X.T):
            if X_in is not None:    ax1 = fig.add_subplot(X.shape[1],2,2*(idx+1))
            else:                   ax1 = fig.add_subplot(X.shape[1],1,idx+1)
            
            #plot datapoints
            sctr    = ax1.scatter(x_vals, xi, **sctr_kwargs)
            if X_e is not None:
                #get colors of datapoints to assign to errors
                c = sctr.to_rgba(sctr.get_array())
                if 'ecolor' not in errorbar_kwargs.keys(): errorbar_kwargs['ecolor'] = c
                #plot errorbars
                errbar  = ax1.errorbar(x_vals, xi, yerr=X_e.T[idx], **errorbar_kwargs)
            
            ax1.set_xlabel('x')
            ax1.set_ylabel(f'X[:,{idx}]')
            #append for animation
            lc_axs.append(ax1)
        
        fig.tight_layout()

        if animate:
            vlines = [axi.axvline(x_vals[0], color='tab:grey', ls='--') for axi in lc_axs]
            anim = manimation.FuncAnimation(
                fig,
                func=update,
                **func_animation_kwargs
            )
        else:
            anim = None
        
        axs = fig.axes

        return fig, axs, anim

class BestAperture:
    """
        - class to exectue an analysis for the determination of the best aperture

        Attributes
        ----------
            - `mode`
                - `Literal["flux","mag"]`, optional
                - mode to use
                - can be one of
                    - `'flux'`
                        - will consider equations for fluxes to execute calculations
                    - `'mag'`
                        - will consider equations for magnitudes to execute calculations
            - `store_ring_masks`
                - `bool`, optional
                - whether to store all sky-ring masks created during calculations
                    - those will be used during i.e., plotting
                - the default is `True`
            - `store_aperture_masks`
                - `bool`, optional
                - whether to store all aperture masks created during calculations
                    - those will be used during i.e., plotting
                - the default is `True`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Infered Attributes
        ------------------
            - `aperture_res`
                - `np.ndarray`
                - contains results for the analysis of the aperture
                - has shape `(nradii,3)`
                    - `nradii` is the number of tested radii
                    - the last axis contains
                        - element 0: tested radius
                        - element 1: total flux/magnitude within aperture
                        - element 2: number of enclosed pixels
            - `ring_res`
                - `np.ndarray`
                - contains results for the analysis of the sky-ring
                - has shape `(nradii*nwidths,4)`
                    - `nradii` is the number of tested radii
                    - `nwidths` is the number of tested widths for the sky-ring
                    - the last axis contains
                        - element 0: tested radius
                        - element 1: tested width
                        - element 2: total flux/magnitude within aperture (dependent on `self.mode`)
                        - element 3: number of enclosed pixels
            - `aperture_masks`
                - `np.ndarray`
                - contains boolean arrays of all tested apertures
            - `ring_masks`
                - `np.ndarray`
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
            - `matplotlib`
            - `numpy`
            - `typing`

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
        

        return

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
                    - `np.ndarray`
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
                    - `np.ndarray`
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
                    - `np.ndarray`
                    - frame consisting of the pixel wise total flux/magnitude

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
                    - `np.ndarray`
                    - series of frames to consider for the analysis
                    - has to have shape `(nframes,npixels,npixels,3)`
                        - the last axis contains
                            - element 0: posistion in x direction
                            - element 1: posistion in y direction
                            - element 2: flux/magnitude values
                - `posx`
                    - `float`
                    - position of the aperture in x direction
                - `posy`
                    - `float`
                    - position of the aperture in y direction
                - `r_aperture`
                    - `np.ndarray`
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

            #get total flux/magnitude contained in aperture
            if self.mode == 'flux':
                aperture_brightness = np.sum(self.sum_frame[aperture_mask,2])#/np.sum(aperture_mask)**2
            elif self.mode == 'mag':
                aperture_brightness = alpp.mags_sum(self.sum_frame[aperture_mask,2])
            
            #store result
            self.aperture_res = np.append(self.aperture_res, np.array([[r, aperture_brightness, aperture_mask.sum()]]), axis=0)

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
                    - `np.ndarray`
                    - series of frames to consider for the analysis
                    - has to have shape `(nframes,npixels,npixels,3)`
                        - the last axis contains
                            - element 0: posistion in x direction
                            - element 1: posistion in y direction
                            - element 2: flux/magnitude values
                - `posx`
                    - `float`
                    - position of the sky-ring in x direction
                - `posy`
                    - `float`
                    - position of the sky-ring in y direction
                - `rw_sky`
                    - `np.ndarray`
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
            if self.mode == 'flux':
                ring_brightness = np.sum(self.sum_frame[ring_mask,2])#/np.sum(ring_mask)**2
            elif self.mode == 'mag':
                ring_brightness = alpp.mags_sum(self.sum_frame[ring_mask,2])
            

            #get sky-ring results
            self.ring_res = np.append(self.ring_res, np.array([[r, w, ring_brightness, ring_mask.sum()]]), axis=0)
            
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
        test_aperture_kwargs:dict=None,
        test_background_kwargs:dict=None,
        *args, **kwargs
        ) -> None:
        """
            - method to fit the estimator
            
            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - input values to apply estimator on
                    - i.e. frames of the timeseries
                - `posx`
                    - `float`
                    - position of aperture and sky-ring in x direction
                - `posy`
                    - `float`
                    - position of aperture and sky-ring in y direction
                - `r_aperture`
                    - `np.ndarray`
                    - test radii of the aperture
                    - will test every radius and calculate total flux/magnitude contained within that radius
                - `rw_sky`
                    - `np.ndarray`
                    - test specifications for sky-rings to test
                    - has to have shape `(nrings,2)`
                        - element 0: radii of the sky-rings
                        - element 1: widths of the sky-rings
                - `test_aperture_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.test_aperture()`
                    - the default is `None`
                        - will be set to `dict()`
                - `test_background_kwargs`
                    - `dict`, optional
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
        *args, **kwargs
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
        fit_kwargs:dict=None,
        predict_kwargs:dict=None,
        ) -> None:
        """
            - method to fit the estimator and and predict with it

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - input values to apply estimator on
                    - i.e. frames of the timeseries
                - `posx`
                    - `float`
                    - position of aperture and sky-ring in x direction
                - `posy`
                    - `float`
                    - position of aperture and sky-ring in y direction
                - `r_aperture`
                    - `np.ndarray`
                    - test radii of the aperture
                    - will test every radius and calculate total flux/magnitude contained within that radius
                - `rw_sky`
                    - `np.ndarray`
                    - test specifications for sky-rings to test
                    - has to have shape `(nrings,2)`
                        - element 0: radii of the sky-rings
                        - element 1: widths of the sky-rings
                - `test_aperture_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.test_aperture()`
                    - the default is `None`
                        - will be set to `dict()`
                - `test_background_kwargs`
                    - `dict`, optional
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
                    - `np.ndarray`, optional
                    - radii to visualize the aperture for in the frame of total magnitudes/fluxes
                    - the default is `None`
                        - will be set to `[]`
                - `plot_sky_rings_rw`
                    - `np.ndarray`, optional
                    - radii-width combinations to visualize the sky-ring for in the frame of total magnitudes/fluxes
                    - has to have shape `(n2plot,2)`
                        - element 0: sky-ring radius
                        - emement 1: sky-ring width
                    - the default is `None`
                        - will be set to `[]`
                - `aperture_cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to use for colorcoding apertures
                    - the default is `None`
                        - will be set to `autumn`
                - `sky_rings_cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to use for colorcoding apertures
                    - the default is `None`
                        - will be set to `winter`
                - `fig`
                    - `Figure`, optional
                    - figure to plot into
                    - the default is `None`
                        - will create a new figure
                - `sort_rings_apertures`
                    - `bool`, optional
                    - whether to sort the passed rings and apertures before plotting
                    - recommended because then it is more likely that all requested apertures/sky-rings are visible
                        - if they are not sorted, it could be that a large aperture is plot on top of a smaller one, essentially covering it
                - `plot_kwargs`
                    - `dict`, optional
                    - kwargs to pass to`ax.plot()`
                    - the default is `None`
                        - will be set to `dict()`
                - `scatter_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - created figure
                - `axs`
                    - `plt.Axes`
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
        cbar2.set_label('Sky Ring Width [pixel]')

        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')
        ax2.set_xlabel('Radius [pixel]')
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

class DifferentialPhotometryImage:
    """
        - class to execute differential photometry on a image-level

        Attributes
        ----------
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `fit()`
            - `transform()`
            - `fit_transform()`
            - `plot_result()`

        Dependencies
        ------------
            - `joblib`
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------
    """

    def __init__(self,
        verbose:int=0,
        ) -> None:
        
        self.verbose    = verbose

        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def fit(self,
        X:np.ndarray, X_ref:Union[np.ndarray,int]=None,
        X_e:np.ndarray=None, X_ref_e:np.ndarray=None,
        strategy:Union[Callable,Literal['previous']]=None,
        verbose:int=None,
        strategy_kwargs:dict=None,
        *args, **kwargs
        ) -> None:
        """
            - method to execute differential photometry on one particular target `X`

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - series of frames of the science target
                    - has shape `(nframes,xpixels,ypixels)`
                - `X_ref`
                    - `np.ndarray`, `int`, optional
                    - reference frame or series to utilize for executing differential photometry
                    - if `np.ndarray`
                        - calls `X - X_ref`
                        - will subtract reference frame(s) from `X`
                    - if `int`
                        - calls `X - X[X_ref]`
                        - will be interpreted as the index of the reference frame in `X`
                    - the default is `None`
                        - will be ignored
                - `X_e`
                    - `np.ndarray`, optional
                    - errors corresponding to `X`
                    - has shape `(nframes,xpixels,ypixels)`
                    - the default is `None`
                        - will be set to array containing only `np.nan`
                        - no errors considered
                - `X_ref_e`
                    - `np.ndarray`, optional
                    - errors corresponding to `X_ref`
                    - has to have same shape as `X_ref`
                    - the default is `None`
                        - will be set to array containing only `np.nan`
                        - no errors considered
                - `strategy`
                    - `Callable(np.ndarray,np.ndarray,np.ndarray,np.ndarray,Any) -> (np.ndarray,np.ndarray)`, `Literal['previous']`, optional
                    - strategy to envoke for executing the differential photometry
                    - if `Callable`
                        - has to take 4 arguments
                            - `X`
                            - `X_ref`
                            - `X_e`
                            - `X_ref_e`
                        - has to return 2 `np.ndarray`s
                            - frames after applying differential photmetry
                            - propagated uncertainties for the resulting frame after differential photometry
                                - return array filled with `np.nan` in case no errors are computed
                        - will call `strategy(X, X_ref, X_e, X_ref_e, **strategy_kwargs)` instead of the default implementation
                        - used to execute custom strategies
                    - if `previous`
                        - only executed if `X_ref` is `None`
                        - will call `np.diff(X, axis=0)`
                        - will consider the previous frame as the reference
                    - the default is `None`
                        - will be set to `previous`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `strategy_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `strategy`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if strategy is None:        strategy        = 'previous'
        if strategy_kwargs is None: strategy_kwargs = dict()
        if X_e is None:             X_e             = np.zeros_like(X)*np.nan
        if X_ref_e is None:         X_ref_e         = np.zeros_like(X)*np.nan

        #init output
        
        #execute custom strategy if specified
        if callable(strategy):
            X_dp, X_dp_e = strategy(X, X_ref, X_e, X_ref_e, **strategy_kwargs)
        #use predefined strategies
        else:
            #check for if `X_ref` has been passsed
            if isinstance(X_ref,int):
                X_dp = X - X[X_ref]             #subtract frame with index `X_ref` from whole series
                X_dp_e = np.sqrt(X_e**2 + X_e[X_ref]**2)
            elif isinstance(X_ref, np.ndarray):
                X_dp = X - X_ref                #subtract complete frame/series of frames
                X_dp_e = np.sqrt(X_e**2 + X_ref_e**2)
            #check for strategy
            elif strategy == 'previous':
                X_dp = np.diff(X, axis=0)
                X_dp_e = np.sqrt(X_e[:-1]**2 + X_e[1:]**2)
            else:
                X_dp = None
                X_dp_e = None
        
        self.X_dp   = X_dp
        self.X_dp_e = X_dp_e

        return

    def transform(self,
        *args, **kwargs
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to transform the input

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `X_dp`
                    - `np.ndarray`
                    - frames after applying differential photometry
                    - has same shape as `X` (in general)
                        - if `strategy=='previous'`
                            - has shape `(X.shape[0]-1,X.shape[1],X-shape[2])`
                            - one frame less than `X`
                        - if `strategy` is a `Callable`
                            - has custom output shape
                            - entries `X`
                - `X_dp_e`
                    - 'np.ndarray`
                    - uncertainty estimate for frames after applying differential photometry
                    - same shape as `X_dp`

            Comments
            --------
        """
        X_dp    = self.X_dp
        X_dp_e  = self.X_dp_e
        
        return X_dp, X_dp_e
    
    def fit_transform(self,
        X:np.ndarray, X_ref:Union[np.ndarray,int]=None,
        X_e:np.ndarray=None, X_ref_e:np.ndarray=None,
        verbose:int=None,
        fit_kwargs:dict=None,
        *args, **kwargs
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - series of frames of the science target
                    - has shape `(nframes,xpixels,ypixels)`
                - `X_ref`
                    - `np.ndarray`, `int`, optional
                    - reference frame or series to utilize for executing differential photometry
                    - if `np.ndarray`
                        - calls `X - X_ref`
                        - will subtract reference frame(s) from `X`
                    - if `int`
                        - calls `X - X[X_ref]`
                        - will be interpreted as the index of the reference frame in `X`
                    - the default is `None`
                        - will be ignored
                - `X_e`
                    - `np.ndarray`, optional
                    - errors corresponding to `X`
                    - has shape `(nframes,xpixels,ypixels)`
                    - the default is `None`
                        - will be set to array containing only `np.nan`
                        - no errors considered
                - `X_ref_e`
                    - `np.ndarray`, optional
                    - errors corresponding to `X_ref`
                    - has to have same shape as `X_ref`
                    - the default is `None`
                        - will be set to array containing only `np.nan`
                        - no errors considered
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `fit_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`
            
            Raises
            ------
            
            Returns
            -------
                - `X_dp`
                    - `np.ndarray`
                    - frames after applying differential photometry
                    - has same shape as `X` (in general)
                        - if `strategy=='previous'`
                            - has shape `(X.shape[0]-1,X.shape[1],X-shape[2])`
                            - one frame less than `X`
                        - if `strategy` is a `Callable`
                            - has custom output shape
                            - entries `X`
                - `X_dp_e`
                    - 'np.ndarray`
                    - uncertainty estimate for frames after applying differential photometry
                    - same shape as `X_dp`

            Comments
            --------
        """
        if verbose is None:     verbose     = self.verbose
        if fit_kwargs is None:  fit_kwargs  = dict()
        
        self.fit(X, X_ref, X_e, X_ref_e, verbose=verbose, **fit_kwargs)
        X_dp, X_dp_e = self.transform()

        return X_dp, X_dp_e
    
    def plot_result(self,
        X:np.ndarray, y:np.ndarray=None, X_e:np.ndarray=None,
        X_in:np.ndarray=None, X_in_e:np.ndarray=None,
        X_ref:np.ndarray=None, X_ref_e:np.ndarray=None,
        x_vals:np.ndarray=None, aperture:Union[np.ndarray,float]=2.0,
        fig:Figure=None,
        animate:bool=True,
        verbose:int=None,
        pcolormesh_kwargs:dict=None,
        sctr_kwargs:dict=None,
        errorbar_kwargs:dict=None,
        func_animation_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes,manimation.FuncAnimation]:
        """
            - method to visulize the result
            - will animate the series of frames if requested
            - will generate some exemplary LCs

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - transformed input
                    - has shape `(nframes,xpixels,ypixels)`
                - `X_e`
                    - `np.ndarray`, optional
                    - pixel-wise errors corresponding to `X`
                    - has shape `(nframes,xpixels,ypixels)` (same as `X`)
                    - the default is `None`
                        - ignored
                - `X_in`
                    - `np.ndarray`, optional
                    - original input
                    - same shape as `X`
                    - the default is `None`
                        - will be ignored
                - `X_in_e`
                    - `np.ndarray`, optional
                    - pixel-wise errors corresponding to `X_in`
                    - has shape `same shaep as `X_in`
                    - the default is `None`
                        - ignored
                - `X_ref`
                    - `np.ndarray`, optional
                    - reference frame-series
                    - same shape as `X`
                    - the default is `None`
                        - will be ignored
                - `X_ref_e`
                    - `np.ndarray`, optional
                    - pixel-wise errors corresponding to `X_ref`
                    - has shape `same shaep as `X_ref`
                    - the default is `None`
                        - ignored
                - `x_vals`
                    - `np.ndarray`, optional
                    - x-values to use for plotting
                    - useful if one wants to plot i.e., in phase space
                    - has to have shape `(nframes)`
                    - the default is `None`
                        - will be generated automatically based on indices
                - `aperture`
                    - `np.ndarray`, `float`, optional
                    - aperture to use for generating LCs corresponding to the frame-series
                    - will be passed to `astroLuSt.AperturePhotometry().fit()`
                        - passed further down to `astroLuSt.Aperture()`
                    - the default is `2.0`
                - `fig`
                    - `matplotlib.figure.Figure`, optional
                    - figure to plot into
                    - the default is `None`
                    - will generate a new figure
                - `animate`
                    - `bool`, optional
                    - whether to animate the visualization
                    - the default is `True`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to be passed to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict(vmin=np.nanmin(X_in), vmax=np.nanmax(X_in))`
                - `sctr_kwargs`
                    - `dict`, optional
                    - kwargs to be passed to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict(cmap='nipy_spectral')`
                - `errorbar_kwargs`
                    - `dict`, optional
                    - kwargs to be passed to `ax.errorbar()`
                    - the default is `None`
                        - will be set to `dict(ls='', marker=None, zorder=-1)`
                        - `'ecolor'` will be inherited from the datapoints in `ax.scatter()`, if not provided
                - `func_animation_kwargs`
                    - `dict`, optional
                    - kwargs to be passed to `matplotlib.animation.FuncAnimation()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `matplotlib.figure.Figure`
                    - generated figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
                - `anim`
                    - `matplotlib.animation.FuncAnimation`
                    - generated animation
                    - returns `None` if `animate==False`

            Comments
            --------
        """

        def update(
            frame:int,
            ) -> None:
            
            for idx, mesh in enumerate(meshes): mesh.update(dict(array=frames2plot[idx][frame]))
            for vline in vlines: vline.set_xdata([x_vals[frame]])

            return


        #default parameters
        if x_vals is None:                  x_vals                  = np.arange(X.shape[0])
        if verbose is None:                 verbose                 = self.verbose
        if pcolormesh_kwargs is None:       pcolormesh_kwargs       = dict()
        if sctr_kwargs is None:             sctr_kwargs             = dict()
        if 'cmap' not in sctr_kwargs.keys():sctr_kwargs['cmap']     = 'nipy_spectral'
        if errorbar_kwargs is None:                 errorbar_kwargs                     = dict()
        if 'ls' not in errorbar_kwargs.keys():      errorbar_kwargs['ls']               = ''
        if 'marker' not in errorbar_kwargs.keys():  errorbar_kwargs['marker']           = None
        if 'zorder' not in errorbar_kwargs.keys():  errorbar_kwargs['zorder']           = -1
        if func_animation_kwargs is None:   func_animation_kwargs   = dict()

        
        if fig is None:
            fig = plt.figure()

        #frames and labels
        ylabs       = ['X']
        frames2plot = [X]
        errs2plot   = [X_e]
        if X_in is not None:
            ylabs.append('X_in')
            frames2plot.append(X_in)
            errs2plot.append(X_in_e)
        if X_ref is not None:
            ylabs.append('X_ref')
            frames2plot.append(X_ref)
            errs2plot.append(X_ref_e)

        #LCs
        AP = AperturePhotometry(verbose=0)
        lcs = []
        lcs_e = []
        aps = []
        for f2p, f2pe in zip(frames2plot, errs2plot):
            lcs_i, lcs_i_e, aps_i = AP.fit_transform(
                X=f2p, X_e=f2pe,
                fit_kwargs=dict(
                    aperture=aperture,
                    # fluxvars=lambda x, x_e, **kwargs: ((x/np.nanmedian(x)).reshape(-1,1),(x_e/np.nanmedian(x)).reshape(-1,1)),
                ),
            )
            lcs.append(lcs_i)
            lcs_e.append(lcs_i_e)
            aps.append(aps_i)
        

        meshes = []
        lc_axs = []
        ncols = 2
        for idx, (Xi, lcsi) in enumerate(zip(frames2plot, lcs)):
            ax1 = fig.add_subplot(len(frames2plot),ncols, ncols*(idx+1)-1)
            ax2 = fig.add_subplot(len(frames2plot),ncols, ncols*(idx+1)-0)
            ax1.set_aspect('equal')
            ax1.set_xlabel('Pixels')
            ax1.set_ylabel('Pixels')
            ax2.set_xlabel('x')
            ax2.set_ylabel(ylabs[idx])
            
            if 'vmin' not in pcolormesh_kwargs: vmin = np.nanmin(Xi)
            else:                               vmin = pcolormesh_kwargs.pop('vmin')
            if 'vmax' not in pcolormesh_kwargs: vmax = np.nanmax(Xi)
            else:                               vmax = pcolormesh_kwargs.pop('vmax')
            
            #plot frame
            mesh = ax1.pcolormesh(Xi[0], vmin=vmin, vmax=vmax, **pcolormesh_kwargs)
            
            #plot lc
            sctr = ax2.scatter(x_vals, lcsi[:,0], **sctr_kwargs)

            #plot corresponding uncertainties
            if errs2plot[idx] is not None:
                #get colors of datapoints to assign to errors
                c = sctr.to_rgba(sctr.get_array())
                if 'ecolor' not in errorbar_kwargs.keys(): errorbar_kwargs['ecolor'] = c
                #plot errorbars
                errbar  = ax2.errorbar(x_vals, lcsi[:,0], yerr=lcs_e[idx][:,0], **errorbar_kwargs)
                        

            #plot aperture
            mesh_ap = ax1.pcolormesh(aps[idx], zorder=2, edgecolor='r', facecolors='none')
            mesh_ap.set_alpha(aps[idx])
            ax1.plot(np.nan, np.nan, '-r', label='Aperture')
            ax1.legend(framealpha=0.5)

            cbar = fig.colorbar(mesh, ax=ax1)
            cbar.set_label(ylabs[idx])

            #append for animation
            meshes.append(mesh)
            lc_axs.append(ax2)


        fig.tight_layout()

        if animate:
            vlines = [axi.axvline(x_vals[0], color='tab:grey', ls='--') for axi in lc_axs]
            anim = manimation.FuncAnimation(
                fig,
                func=update,
                # fargs=[times, sectors, ax1, pcolormesh_kwargs],
                **func_animation_kwargs
            )
        else:
            anim = None

        axs = fig.axes

        return fig, axs, anim
    
class DifferentialPhotometryLC:
    """
        - class to execute differential photometry on lightcurve-level

        Attributes
        ----------
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `fit()`
            - `transform()`
            - `fit_transform()`
            - `plot_result()`

        Dependencies
        ------------

        Comments
        --------
    """

    def __init__(self,
        verbose:int=0,
        ) -> None:
        
        self.verbose    = verbose

        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def fit(self,
        X:np.ndarray, X_ref:np.ndarray,
        X_e:np.ndarray=None, X_ref_e:np.ndarray=None,
        strategy:Callable[[np.ndarray,np.ndarray,np.ndarray,np.ndarray,Any],Tuple[np.ndarray,np.ndarray]]=None,
        verbose:int=None,
        strategy_kwargs:dict=None,
        *args, **kwargs,
        ) -> None:
        """
            - method to execute differential photometry on one particular target `X`

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input lightcurve to apply differential photometry to
                    - has shape `(nobservations)`
                - `X_ref`
                    - `np.ndarray`
                    - reference lightcurve to utilize for executing differential photometry
                        - calls `X - X_ref`
                        - will subtract reference LC from `X`
                - `X_e`
                    - `np.ndarray`, optional
                    - uncertainties of observations in `X`
                    - has shape `(nobservations)`
                    - the default is `None`
                        - will not consider errors in computation
                        - outputs array filled with `np.nan` for `X_dp_e`
                - `X_ref_e`
                    - `np.ndarray`, optional
                    - uncertainties of observations in `X_ref`
                    - has shape `(nobservations)`
                    - the default is `None`
                        - will not consider errors in computation
                        - outputs array filled with `np.nan` for `X_dp_e`
                - `strategy`
                    - `Callable(np.ndarray,np.ndarray,np.ndarray,np.ndarray,Any) -> (np.ndarray,np.ndarray)`, optional
                    - strategy to envoke for executing the differential photometry
                    - has to take 4 arguments
                        - `X`
                        - `X_ref`
                        - `X_e`
                        - `X_ref_e`
                    - has to return two `np.ndarray`s
                        - LC after applying differential photmetry
                        - Errors related to LC after differential photometry
                            - return array filled with `np.nan` in case no errors are computed
                    - will call `strategy(X, X_ref, X_e, X_ref_e, **strategy_kwargs)` instead of the default implementation
                    - used to execute custom strategies
                    - the default is `None`
                        - will use default implementation instead
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `strategy_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `strategy`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                        
            Comments
            --------
        """

        if X_e is None:             X_e             = np.zeros_like(X)*np.nan
        if X_ref_e is None:         X_ref_e         = np.zeros_like(X_ref)*np.nan
        if strategy_kwargs is None: strategy_kwargs = dict()


        #execute custom strategy if specified
        if callable(strategy):
            X_dp, X_dp_e = strategy(X, X_ref, X_e, X_ref_e, **strategy_kwargs)
        #use predefined strategies
        else:
            X_dp = X - X_ref                #element-wise subtract reference LC from input
            X_dp_e = np.sqrt(X_e**2 + X_ref_e**2)          #uncertainty propagation for X_dp

        self.X_dp   = X_dp
        self.X_dp_e = X_dp_e

        return
    
    def transform(self,
        *args, **kwargs,
        ) -> Tuple[List[np.ndarray],List[np.ndarray]]:
        """
            - method to transform the input

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `X_dp`
                    - `np.ndarray`
                    - LC after applying differential photometry
                    - has same shape as `X` (in general)
                    - if `strategy` is a `Callable`
                        - has custom output shape
                -`X_dp_e`
                    - `np.ndarray`
                    - propagated uncertainties corresponding to `X_dp`
                    - has same shape as `X_dp`
                    - will be filled with `np.nan` in case one of the following applies
                        - `X_e is None`
                        - `X_ref_e is None`            

            Comments
            --------
        """
        X_dp    = self.X_dp
        X_dp_e  = self.X_dp_e
        
        return X_dp, X_dp_e
    
    def fit_transform(self,
        X:np.ndarray, X_ref:np.ndarray=None,
        X_e:np.ndarray=None, X_ref_e:np.ndarray=None,
        verbose:int=None,
        fit_kwargs:dict=None,
        *args, **kwargs,
        ) -> np.ndarray:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input lightcurve to apply differential photometry to
                    - has shape `(nobservations)`
                - `X_ref`
                    - `np.ndarray`
                    - reference lightcurve to utilize for executing differential photometry
                        - calls `X - X_ref`
                        - will subtract reference LC from `X`
                - `X_e`
                    - `np.ndarray`, optional
                    - uncertainties of observations in `X`
                    - has shape `(nobservations)`
                    - the default is `None`
                        - will not consider errors in computation
                        - outputs array filled with `np.nan` for `X_dp_e`
                - `X_ref_e`
                    - `np.ndarray`, optional
                    - uncertainties of observations in `X_ref`
                    - has shape `(nobservations)`
                    - the default is `None`
                        - will not consider errors in computation
                        - outputs array filled with `np.nan` for `X_dp_e`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `fit_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`
            
            Raises
            ------
            
            Returns
            -------
                - `X_dp`
                    - `np.ndarray`
                    - LC after applying differential photometry
                    - has same shape as `X` (in general)
                    - if `strategy` is a `Callable`
                        - has custom output shape
                -`X_dp_e`
                    - `np.ndarray`
                    - propagated uncertainties corresponding to `X_dp`
                    - has same shape as `X_dp`
                    - will be filled with `np.nan` in case one of the following applies
                        - `X_e is None`
                        - `X_ref_e is None`  

            Comments
            --------
        """

        if verbose is None:     verbose     = self.verbose
        if fit_kwargs is None:  fit_kwargs  = dict()
        
        self.fit(X, X_ref, X_e, X_ref_e, verbose=verbose, **fit_kwargs)
        X_dp, X_dp_e = self.transform()

        return X_dp, X_dp_e
    
    def plot_result(self,
        X:np.ndarray, X_e:np.ndarray=None,
        X_in:np.ndarray=None, X_in_e:np.ndarray=None,
        X_ref:np.ndarray=None, X_ref_e:np.ndarray=None,
        x_vals:np.ndarray=None,
        fig:Figure=None,
        verbose:int=None,
        sctr_kwargs:dict=None,
        errorbar_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visulize the result

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - transformed input
                    - has shape `(nobservations)`
                - `X_e`
                    - `np.ndarray`, optional
                    - uncertainties associated with `X`
                    - has shape `(nobservations)` (same as `X`)
                    - the default is `None`
                        - no uncertainties plotted
                - `X_in`
                    - `np.ndarray`, optional
                    - original input
                    - same shape as `X`
                    - the default is `None`
                        - will be ignored
                - `X_in_e`
                    - `np.ndarray`, optional
                    - uncertainties associated with `X_in`
                    - same shape as `X_in`
                    - the default is `None`
                        - no uncertainties plotted
                - `X_ref`
                    - `np.ndarray`, optional
                    - reference lc
                    - same shape as `X`
                    - the default is `None`
                        - will be ignored
                - `X_ref_e`
                    - `np.ndarray`, optional
                    - uncertainties associated with `X_ref`
                    - same shape as `X_ref`
                    - the default is `None`
                        - no uncertainties plotted
                - `x_vals`
                    - `np.ndarray`, optional
                    - x-values to use for plotting
                    - useful if one wants to plot i.e., in phase space
                    - has to have shape `(nobservations)`
                    - the default is `None`
                        - will be generated automatically based on indices
                - `fig`
                    - `matplotlib.figure.Figure`, optional
                    - figure to plot into
                    - the default is `None`
                    - will generate a new figure
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`
                - sctr_kwargs
                    - `dict`, optional
                    - kwargs to be passed to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict(cmap='nipy_spectral')`
                - `errorbar_kwargs`
                    - `dict`, optional
                    - kwargs to be passed to `ax.errorbar()`
                    - the default is `None`
                        - will be set to `dict(ls='', marker=None, zorder=-1)`
                        - `'ecolor'` will be inherited from the datapoints in `ax.scatter()`, if not provided


            Raises
            ------

            Returns
            -------
                - `fig`
                    - `matplotlib.figure.Figure`
                    - generated figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`

            Comments
            --------
        """
        #default parameters
        if x_vals is None:                          x_vals                              = np.arange(X.shape[0])
        if verbose is None:                         verbose                             = self.verbose
        if sctr_kwargs is None:                     sctr_kwargs                         = dict()
        if 'cmap' not in sctr_kwargs.keys():        sctr_kwargs['cmap']                 = 'nipy_spectral'
        if errorbar_kwargs is None:                 errorbar_kwargs                     = dict()
        if 'ls' not in errorbar_kwargs.keys():      errorbar_kwargs['ls']               = ''
        if 'marker' not in errorbar_kwargs.keys():  errorbar_kwargs['marker']           = None
        if 'zorder' not in errorbar_kwargs.keys():  errorbar_kwargs['zorder']           = -1

        
        if fig is None:
            fig = plt.figure()

        #frames and labels
        ylabs       = ['X']
        frames2plot = [X]
        errs2plot   = [X_e]
        if X_in is not None:
            ylabs.append('X_in')
            frames2plot.append(X_in)
            errs2plot.append(X_in_e)
        if X_ref is not None:
            ylabs.append('X_ref')
            frames2plot.append(X_ref)
            errs2plot.append(X_ref_e)


        for idx, Xi in enumerate(frames2plot):
            ax1 = fig.add_subplot(len(frames2plot),1, idx+1)
            ax1.set_xlabel('x')
            ax1.set_ylabel(ylabs[idx])
            
            #plot lc
            sctr = ax1.scatter(x_vals, Xi, **sctr_kwargs)
            
            #plot corresponding uncertainties
            if errs2plot[idx] is not None:
                #get colors of datapoints to assign to errors
                c = sctr.to_rgba(sctr.get_array())
                if 'ecolor' not in errorbar_kwargs.keys(): errorbar_kwargs['ecolor'] = c
                #plot errorbars
                errbar  = ax1.errorbar(x_vals, Xi, yerr=errs2plot[idx][idx], **errorbar_kwargs)
            

        fig.tight_layout()

        axs = fig.axes

        return fig, axs

