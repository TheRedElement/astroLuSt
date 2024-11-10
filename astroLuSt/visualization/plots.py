#TODO: `ParallelCoordinates`: gap in categorical if 'nan' alphabetically after something else

#%%imports
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from typing import Union, Tuple, List, Callable, Literal
import warnings

from astroLuSt.visualization import plotting as alvp
from astroLuSt.monitoring import formatting as almf


#%%classes

class CornerPlot:
    """
        - class to generate a corner plot given some data and potentially labels

        Attributes
        ----------

        Methods
        -------
            - `__2standardnormal()`
            - `__2d_distributions()`
            - `__1d_distributions()`
            - `plot()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `scipy`
            - `typing`

        Comments
        --------
    """

    def __init__(self) -> None:
        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def __2standardnormal(self,
        d1:np.ndarray, mu1:float, sigma1:float,
        d2:np.ndarray, mu2:float, sigma2:float,
        ) -> Tuple[np.ndarray,float,float,np.ndarray,float,float]:
        """
            - private method to convert the input to a standard normal distibution
                - zero mean
                - unit variance
            
            Parameters
            ----------
                - `d1`
                    - `np.ndarray`
                    - data of the first coordinate to plot
                - `mu1`
                    - `np.ndarray`
                    - mean of the first coordinate
                - `sigma1`
                    - `np.ndarray`
                    - standard deviation of the first coordinate
                - `d2`
                    - `np.ndarray`
                    - data of the second coordinate to plot
                - `mu2`
                    - `np.ndarray`
                    - mean of the second coordinate
                - `sigma2`
                    - `np.ndarray`
                    - standard deviation of the second coordinate

            Raises
            ------

            Returns
            -------
                - `d1`
                    - `np.ndarray`
                    - normalized input `d1`
                - `mu1`
                    - `float`
                    - mean of `d1`
                - `sigma1`
                    - `float`
                    - standard deviation of `d1`
                - `d2`
                    - `np.ndarray`
                    - normalized input `d2`
                - `mu2`
                    - `float`
                    - mean of `d2`
                - `sigma2`
                    - `float`
                    - standard deviation of `d2`

            Comments
            --------
        """

        d1 = (d1-mu1)/sigma1
        d2 = (d2-mu2)/sigma2
        mu1, mu2 = 0, 0
        sigma1, sigma2 = 1, 1

        return (
            d1, mu1, sigma1,
            d2, mu2, sigma2,       
        )
    
    def __2d_distributions(self,
        idx1:int, idx2:int, idx:int,
        d1:np.ndarray, mu1:float, sigma1:float, l1:str,
        d2:np.ndarray, mu2:float, sigma2:float, l2:str,
        corrmat:np.ndarray,
        y:np.ndarray,
        cmap:Union[str,mcolors.Colormap],
        xvals:np.ndarray, yvals:np.ndarray,
        fig:Figure, nrowscols:int,
        sctr_kwargs:dict=None,
        contour_kwargs:dict=None,
        axvline_kwargs:dict=None,
        ) -> plt.Axes:
        """
            - method to generate the (off-diagonal) 2d distributions

            Parameters
            ----------
                - `idx1`
                    - `int`
                    - index of y coordinate in use
                - `idx2`
                    - `int`
                    - index of x coordinate in use
                - `idx`
                    - `int`
                    - current subplot index
                - `d1`
                    - `np.ndarray`
                    - data of the first coordinate to plot
                - `mu1`
                    - `np.ndarray`
                    - mean of the first coordinate
                - `sigma1`
                    - `np.ndarray`
                    - standard deviation of the first coordinate
                - `l1`
                    - `str`
                    - label to apply to y coordinate in use
                - `d2`
                    - `np.ndarray`
                    - data of the second coordinate to plot
                - `mu2`
                    - `np.ndarray`
                    - mean of the second coordinate
                - `sigma2`
                    - `np.ndarray`
                    - standard deviation of the second coordinate
                - `l2`
                    - `str`
                    - label to apply to x coordinate in use
                - `corrmat`
                    - `np.ndarray`
                    - correlation matrix for all passed coordinates
                - `y`
                    - `np.ndarray`
                    - labels for each sample
                - `cmap`
                    - `str`, `Colormap`
                    - name of colormap or `Colormap` instance to color the datapoints
                - `xvals`
                    - `np.ndarray`
                    - x-values to use for plotting
                    - used for generating the normal distribution estimate
                    - used for defining x-axis limits
                - `yvals`
                    - `np.ndarray`
                    - y-values to use for plotting
                    - used for generating the normal distribution estimate
                    - used for defining y-axis limits
                - `fig`
                    - `Figure`
                    - figure to plot into
                - `nrowscols`
                    - `int`
                    - number of rows and columns of the corner-plot
                - `sctr_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict(s=1, alpha=0.5, zorder=2)`
                - `countour_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.contour()`
                    - the default is `None`
                        - will be set to `dict(cmap=cur_cmap)`
                        - default cmap

            Raises
            ------

            Returns
            -------
                - `ax`
                    - `plt.Axes`
                    - created axes

            Comments
            --------
        """

        cur_cmap = plt.rcParams["image.cmap"]


        #default values
        if sctr_kwargs is None:
            sctr_kwargs = dict(s=1, alpha=0.5, zorder=2)
        if 's' not in sctr_kwargs.keys():       sctr_kwargs['s']        = 1
        if 'alpha' not in sctr_kwargs.keys():   sctr_kwargs['alpha']    = 0.5
        if 'zorder' not in sctr_kwargs.keys():  sctr_kwargs['zorder']   = 2
        if contour_kwargs is None:              contour_kwargs          = dict(cmap=cur_cmap)
        if 'cmap' not in contour_kwargs.keys(): contour_kwargs['cmap']  = cur_cmap
        if axvline_kwargs is None:                  axvline_kwargs              = dict(color='C0', linestyle='--')
        if 'color' not in axvline_kwargs.keys():    axvline_kwargs['color']     = 'C0'
        if 'linestyle' not in axvline_kwargs.keys():axvline_kwargs['linestyle'] = '--'
        
        #add new panel
        ax = fig.add_subplot(nrowscols, nrowscols, idx)
        
        #lines for means
        if mu1 is not None: ax.axhline(mu1, **axvline_kwargs)
        if mu2 is not None: ax.axvline(mu2, **axvline_kwargs)

        #data
        sctr = ax.scatter(
            d2, d1,
            c=y,
            cmap=cmap,
            **sctr_kwargs,
        )
             
        if mu1 is not None and sigma1 is not None:
            
            covmat = np.cov(np.array([d1,d2]))

            xx, yy = np.meshgrid(xvals, yvals)
            mesh = np.dstack((xx, yy))
            
            norm = stats.multivariate_normal(
                mean=np.array([mu1, mu2]),
                cov=covmat,
                allow_singular=True
            )
            cont = ax.contour(yy, xx, norm.pdf(mesh), zorder=1, **contour_kwargs)
        

        #labelling
        if idx1 == nrowscols-1:
            ax.set_xlabel(l2)
        else:
            ax.set_xticklabels([])
        if idx2 == 0:
            ax.set_ylabel(l1)
        else:
            ax.set_yticklabels([])
        ax.tick_params()

        ax.set_xlim(np.nanmin(xvals), np.nanmax(xvals))
        ax.set_ylim(np.nanmin(yvals), np.nanmax(yvals))

        ax.margins(x=0,y=0)

        #add corrcoeff in legend
        ax.errorbar(np.nan, np.nan, color="none", label=r"$r_\mathrm{P}=%.4f$"%(corrmat[idx1, idx2]))
        ax.legend()


        return ax

    def __1d_distributions(self,
        idx:int,
        d1:np.ndarray, mu1:float, sigma1:float,
        y:np.ndarray,
        cmap:Union[str,mcolors.Colormap],
        bins:np.ndarray,
        fig:Figure, nrowscols:int,
        hist_kwargs:dict=None,
        sctr_kwargs:dict=None,
        plot_kwargs:dict=None,
        axvline_kwargs:dict=None,
        ) -> plt.Axes:
        """
            - method to generate (on-diagonal) 1d distributions (i.e. histograms)

            Parameters
            ----------
                - `idx`
                    - `int`
                    - current subplot index
                - `d1`
                    - `np.ndarray`
                    - data of the first coordinate to plot
                - `mu1`
                    - `np.ndarray`
                    - mean of the first coordinate
                - `sigma1`
                    - `np.ndarray`
                    - standard deviation of the first coordinate
                - `y`
                    - `np.ndarray`
                    - labels for each sample
                - `cmap`
                    - `str`, `Colormap`
                    - name of colormap or `Colormap` instance to color the datapoints
                - `bins`
                    - `np.ndarray`
                    - bins to use in the histogram
                - `fig`
                    - `Figure`
                    - figure to plot into
                - `nrowscols`
                    - `int`
                    - number of rows and columns of the corner-plot
                - `hist_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.hist()`
                - `sctr_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.scatter()`
                    - will use only part of the information to format 1d-histograms similarly to the scatters
                - `plot_kwargs`
                    - ``dict`` optional
                    - kwargs to pass to `ax.plot()`
                    - the default is `None`
                        - will be set to `dict(color='C2')`
                - `axvline_kwargs`
                    - `dict` optional
                    - kwargs to pass to `ax.axvline()`
                    - the default is `None`
                        - will be set to `dict(color='C0', linestyle='--')`
                    
            Raises
            ------

            Returns
            -------
                - `ax`
                    - `plt.Axes`
                    - created axes

            Comments
            --------

        """

        #default parameters
        if hist_kwargs is None:                     hist_kwargs                 = dict(density=True, alpha=0.5, zorder=2)
        if 'density' not in hist_kwargs.keys():     hist_kwargs['density']      = True
        if 'alpha' not in hist_kwargs.keys():       hist_kwargs['alpha']        = 0.5
        if 'zorder' not in hist_kwargs.keys():      hist_kwargs['zorder']       = 2
        if sctr_kwargs is None:                     sctr_kwargs                 = dict(s=1, alpha=0.5, zorder=2)
        if 's' not in sctr_kwargs.keys():           sctr_kwargs['s']            = 1
        if 'alpha' not in sctr_kwargs.keys():       sctr_kwargs['alpha']        = 0.5
        if 'zorder' not in sctr_kwargs.keys():      sctr_kwargs['zorder']       = 2        
        if plot_kwargs is None:                     plot_kwargs                 = dict(color='C2')
        if 'color' not in plot_kwargs.keys():       plot_kwargs['color']        = 'C2'
        if axvline_kwargs is None:                  axvline_kwargs              = dict(color='C0', linestyle='--')
        if 'color' not in axvline_kwargs.keys():    axvline_kwargs['color']     = 'C0'
        if 'linestyle' not in axvline_kwargs.keys():axvline_kwargs['linestyle'] = '--'
        
        
        if 'vmin' in sctr_kwargs.keys(): vmin = sctr_kwargs['vmin']
        else:                            vmin = None
        if 'vmax' in sctr_kwargs.keys(): vmax = sctr_kwargs['vmax']
        else:                            vmax = None

        #get colors for distributions
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        if isinstance(y, str):  colors = [y]    ##if no classes got passed
        else:
            ##generate colormap if classes are passed
            colors = cmap(mcolors.Normalize(vmin=vmin, vmax=vmax)(np.unique(y).astype(np.float64)))

        #add panel
        ax = fig.add_subplot(nrowscols, nrowscols, idx)

        #plot histograms
        if 'density' in hist_kwargs.keys():
            if hist_kwargs['density']:  countlab = 'Normalized Counts'
            else:                       countlab = 'Counts'
        else:
            countlab = 'Counts'


        if idx != 1:
            orientation = 'horizontal'
            ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
            ax.xaxis.set_label_position('top')
            ax.set_xlabel(countlab)
            ax.set_yticklabels(ax.get_yticklabels(), visible=False)
            ax.set_ymargin(0)
        else:
            orientation = 'vertical'
            ax.set_ylabel(countlab)
            ax.set_xticklabels(ax.get_xticklabels(), visible=False)
            ax.set_xmargin(0)

        #plot histograms (color each class in y)
        for yu, c in zip(np.unique(y), colors):
            ax.hist(
                d1[(y==yu)].flatten(),
                orientation=orientation,
                color=c,
                bins=bins,
                **hist_kwargs
            )


        #normal distribution estimate
        if mu1 is not None and sigma1 is not None:
            normal = stats.norm.pdf(bins, mu1, sigma1)
            
            if orientation == 'horizontal':
                ax.plot(normal, bins, **plot_kwargs)
                ax.axhline(mu1, label=r'$\mu=%.2f$'%(mu1), **axvline_kwargs)
            
            elif orientation == 'vertical':
                ax.plot(bins, normal, **plot_kwargs)
                ax.axvline(mu1, label=r'$\mu=%.2f$'%(mu1), **axvline_kwargs)
        
            ax.errorbar(np.nan, np.nan, color='none', label=r'$\sigma=%.2f$'%(sigma1))
            ax.legend()

        return ax
   
    def plot(self,
        X:np.ndarray, y:Union[np.ndarray,str]=None, featurenames:np.ndarray=None,
        mus:np.ndarray=None, sigmas:np.ndarray=None, corrmat:np.ndarray=None,
        bins:Union[int,np.ndarray]=100,
        cmap:Union[str,mcolors.Colormap]=None,
        asstandardnormal:bool=False,
        fig:Figure=None,
        sctr_kwargs:dict=None,
        contour_kwargs:dict=None,
        hist_kwargs:dict=None,
        plot_kwargs:dict=None,
        axvline_kwargs:dict=None,
        ):
        """
            - method to create the corner-plot

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - contains samples as rows
                    - contains features as columns
                - `y`
                    - `np.ndarray`, `str`, optional
                    - contains labels corresponding to `X`
                    - if a `np.ndarray` is passed
                        - will be used as the colormap
                    - if a string is passed
                        - will be interpreted as the actual color
                    - the default is `None`
                        - will default to `'C0'`
                - `featurenames`
                    - `np.ndarray`, optional
                    - names to give to the features present in `X`
                    - the default is `None`
                        - will initialize with `'Feature i'`, where `i` is the index at which the feature appears in `X`
                - `mus`
                    - `np.ndarray`, optional
                    - contains the mean value estimates corresponding to `X`
                    - the default is `None`
                        - will be ignored
                - `sigmas`
                    - `np.ndarray`, optional
                    - contains the standard deviation estimates corresponding to `X`
                    - the default is `None`
                        - will be ignored
                - `corrmat`
                    - `np.ndarray`, optional
                    - correlation matrix for `X`
                    - has to have shape `(X.shape[1],X.shape[1])`
                    - the default is `None`
                        - will infer the correlation coefficients
                - `bins`
                    - `int`, `np.ndarray`, optional
                    - number of bins to use in
                        - `ax.histogram()`
                        - `np.meshgrid()` in `self.__2d_distributions()`
                    - will be passed to 
                        - `self.__2d_distributions()`
                        - `hist_kwargs`
                            - if not overwritten
                    - if `np.ndarray`
                        - will be used as axis limits for ALL axis as well
                        - will use those exact bins for ALL uninque values in `y`
                    - if `int`
                        - will automatically calculate the bins
                        - will use the calculated bins for ALL uninque values in `y`
                    - to enforce equal ranges for all panels use `bins=np.array(X.min(), X.max(), 100)`
                    - the default is `100`
                - `cmap`
                    - `str`, `mcolors.Colormap`
                    - name of the colormap to use or `Colormap` instance
                    - used to color the 1d and 2d distributions according to `y`
                    - the default is `None`
                        - will use current default `cmap`
                - `asstandardnormal`
                    - `bool`, optional
                    - whether to plot the data rescaled to zero mean and unit variance
                    - the default is `False`
                - `fig`
                    - `Figure`, optional
                    - figure to plot into
                    - the default is `None`
                        - will create a new figure
                - `sctr_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict(s=1, alpha=0.5, zorder=2)`
                - `countour_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.contour()`
                    - the default is `None`
                        - will be set to `dict(cmap=cur_cmap)`                        
                        - will use current default `cmap`
                - `hist_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.hist()`
                    - the default is `None`
                        - will be set to `dict(bins=bins, density=True, alpha=0.5, zorder=2)`
                - `plot_kwargs`
                    - `dict` optional
                    - kwargs to pass to `ax.plot()`
                    - the default is `None`
                        - will be set to `dict(color='C2')`
                - `axvline_kwargs`
                    - `dict` optional
                    - kwargs to pass to `ax.axvline()`
                    - the default is `None`
                        - will be set to `dict(color='C0', linestyle='--')`                        

            Raises
            ------

            Returns
            -------
                - `fig`
                    - 'Figure'
                    - the created matplotlib figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`

            Comments
            --------
        """

        cur_cmap = plt.rcParams["image.cmap"]

        #default parameters
        if y is None: y = 'C1'
        if featurenames is None: featurenames = [f'Feature {i}' for i in np.arange(X.shape[1])]

        if mus is None:
            mus = [None]*len(X)
        if sigmas is None:
            sigmas = [None]*len(X)
        if corrmat is None:
            corrmat = np.corrcoef(X.T)
        if cmap is None: cmap = cur_cmap
        if sctr_kwargs is None:
            sctr_kwargs = dict(s=1, alpha=0.5, zorder=2)
        if 's' not in sctr_kwargs.keys():       sctr_kwargs['s']        = 1
        if 'alpha' not in sctr_kwargs.keys():   sctr_kwargs['alpha']    = 0.5
        if 'zorder' not in sctr_kwargs.keys():  sctr_kwargs['zorder']   = 2
        if contour_kwargs is None:              contour_kwargs          = dict(cmap=cur_cmap)
        if 'cmap' not in contour_kwargs.keys(): contour_kwargs['cmap']  = cur_cmap
        if hist_kwargs is None:
            hist_kwargs = dict(density=True, alpha=0.5, zorder=2)
        if 'density' not in hist_kwargs.keys(): hist_kwargs['density']  = True
        if 'alpha' not in hist_kwargs.keys():   hist_kwargs['alpha']    = 0.5
        if 'zorder' not in hist_kwargs.keys():  hist_kwargs['zorder']   = 2
        if plot_kwargs is None:                     plot_kwargs                 = dict(color='C2')
        if 'color' not in plot_kwargs.keys():       plot_kwargs['color']        = 'C2'
        if axvline_kwargs is None:                  axvline_kwargs              = dict(color='C0', linestyle='--')
        if 'color' not in axvline_kwargs.keys():    axvline_kwargs['color']     = "C0"
        if 'linestyle' not in axvline_kwargs.keys():axvline_kwargs['linestyle'] = '--'
        
        if fig is None: fig = plt.figure()
        nrowscols = X.shape[1]


        idx = 0
        for idx1, (d1, l1, mu1, sigma1) in enumerate(zip(X.T, featurenames, mus, sigmas)):
            for idx2, (d2, l2, mu2, sigma2) in enumerate(zip(X.T, featurenames, mus, sigmas)):
                idx += 1

                if asstandardnormal and mu1 is not None and sigma1 is not None:
                    d1, mu1, sigma1, \
                    d2, mu2, sigma2, =\
                        self.__2standardnormal(
                            d1, mu1, sigma1,
                            d2, mu2, sigma2,
                        )

                #get x and y values (serve as bins as well)
                if isinstance(bins, int):
                        xvals = np.linspace(np.nanmin(d2), np.nanmax(d2), bins)
                        yvals = np.linspace(np.nanmin(d1), np.nanmax(d1), bins)
                else:
                    xvals = bins.copy()
                    yvals = bins.copy() 

                #plotting 2D distributions
                if idx1 > idx2:
                    
                    ax1 = self.__2d_distributions(
                        idx1=idx1, idx2=idx2, idx=idx,
                        d1=d1, mu1=mu1, sigma1=sigma1, l1=l1,
                        d2=d2, mu2=mu2, sigma2=sigma2, l2=l2,
                        corrmat=corrmat,
                        y=y,
                        cmap=cmap,
                        xvals=xvals, yvals=yvals,
                        fig=fig, nrowscols=nrowscols,
                        sctr_kwargs=sctr_kwargs,                        
                        axvline_kwargs=axvline_kwargs,
                    )

                #plotting 1d histograms
                elif idx1 == idx2:

                    axhist = self.__1d_distributions(
                        idx=idx,
                        d1=d1, mu1=mu1, sigma1=sigma1,
                        y=y,
                        cmap=cmap,
                        bins=xvals,
                        fig=fig, nrowscols=nrowscols,
                        hist_kwargs=hist_kwargs,
                        sctr_kwargs=sctr_kwargs,
                        plot_kwargs=plot_kwargs,
                        axvline_kwargs=axvline_kwargs,
                    )            

        #get axes
        axs = fig.axes

        return fig, axs

class LatentSpaceExplorer:
    """
        - class to plot generated samples from latent variables

        Attributes
        ----------
            - `plot_func`
                - `Callable`, optional
                - function to use for visualizing each individual sample
                - has to take at least 2 positional arguments
                    - `ax`
                        - `plt.Axes`
                        - axis to plot onto
                    - `X`
                        - `np.ndarray`
                        - dataseries to be plotted
                - has to take `**kwargs`
                - the default is `None`
                    - will be set to `lambda ax, x, kwargs: ax.plot(x, **kwargs)`
            - `subplots_kwargs`
                - `dict`, optional
                - kwargs to pass to `plt.subplots()`
                - the default is `None`
                    - will be initialized with `dict()`
            - `predict_kwargs`
                - `dict`, optional
                - kwargs to pass to `generator.predict()`
                    - generator is a parameter passed `self.plot()`
                - the default is `None`
                    - will be initialized with `dict()`
            - `plot_func_kwargs`
                - `dict`, optional
                - kwargs to pass to `plot_func`
                - the default is `None`
                    - will be initialized with `dict()`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `plot_dbe()`
            - `corner_plot()`
            - `generated_1d()`
            - `generated_2d()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `scipy`
            - `sklearn`
            - `typing`

        Comments
        --------
            - a figure shall be created before calling this method
            - if the plot is not showing you might have to call `plt.show()` after the execution of this method

    """

    def __init__(self,
        plot_func:Callable=None,
        subplots_kwargs:dict=None, predict_kwargs:dict=None, plot_func_kwargs:dict=None,
        verbose:int=0
        ) -> None:

        if plot_func is None:           self.plot_func              = lambda ax, x, **kwargs: ax.plot(x, **kwargs)
        else:                           self.plot_func              = plot_func
        if subplots_kwargs is None:     self.subplots_kwargs        = {}
        else:                           self.subplots_kwargs        = subplots_kwargs
        if predict_kwargs is None:      self.predict_kwargs         = {}
        else:                           self.predict_kwargs         = predict_kwargs
        if plot_func_kwargs is None:    self.plot_func_kwargs       = {}
        else:                           self.plot_func_kwargs       = plot_func_kwargs

        self.verbose = verbose

        return
    
    def __repr__(self) -> str:
        
        return (
            f'PlotLatentExamples(\n'
            f'    plot_func={repr(self.plot_func)},\n'
            f'    subplots_kwargs={repr(self.subplots_kwargs)}, predict_kwargs={repr(self.predict_kwargs)}, plot_func_kwargs={self.plot_func_kwargs},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def plot_dbe(self,
        X:np.ndarray, y:np.ndarray,
        res:int=100, k:int=1,
        ax:plt.Axes=None,
        contourf_kwargs:dict=None,
        ) -> None:
        """
            - method to plot estimated desicion-boundaries of data
            - uses voronoi diagrams to to do so
                - estimates the decision boundaries using KNN with k=1
                - Source: https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data
                    - last access: 17.05.2023
            
            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - 2d array containing the features of the data
                        - i.e. 2 features
                - `y`
                    - `np.ndarray`
                    - 1d array of shape `X.shape[0]`
                    - labels corresponding to `X`
                - `res`
                    - `int`, optional
                    - resolution of the estimated boundary
                    - the default is `100`
                - `k`
                    - `int`, optional
                    - number of neighbours to use in the KNN estimator
                    - the default is `1`
                - `ax`
                    - `plt.Axes`
                    - axes to add the density estimate to
                    - the default is `None`
                        - will call `plt.contourf()` instead of `ax.contourf()`
                - `contourf_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `.contourf()` function
                    - the default is `None`
                        - will be initialized with `{'alpha':0.5, 'zorder':-1}`

            Raises
            ------
                - `ValueError`
                    - if either `X` or `y` are not passed in coorect shapes

            Returns
            -------

            Comments
            --------

        """
        
        #initialize parameters
        if contourf_kwargs is None: contourf_kwargs = {'alpha':0.5, 'zorder':-1}

        y = y.flatten()

        #check shapes
        if X.shape[1] != 2:
            raise ValueError(f'"X" has to contain 2 features. I.e. it has to be of shape (n_samples,2) and not {X.shape}')
        if y.shape[0] != X.shape[0]:
            raise ValueError(f'"y" has to be a 1d version of "X" containing the labels corresponding to the samples. I.e. it has to be of shape (n_samples,) and not {y.shape}')

        #get background model
        background_model = KNeighborsClassifier(n_neighbors=k)
        background_model.fit(X, y)
        xx, yy = np.meshgrid(
            np.linspace(X[:,0].min(), X[:,0].max(), res),
            np.linspace(X[:,1].min(), X[:,1].max(), res),
        )

        voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape((res, res))

        #(add) plot
        if ax is not None:
            ax.contourf(xx, yy, voronoiBackground, **contourf_kwargs)
        else:
            plt.contourf(xx, yy, voronoiBackground, **contourf_kwargs)

        return

    def corner_plot(self,
        X:np.ndarray, y:Union[np.ndarray,str]=None, featurenames:np.ndarray=None,
        corner_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to generate a corner_plot of pairwise latent dimensions

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input dataset
                    - contains samples as rows and features as columns
                - `y`
                    - `np.ndarray`, `str`, optional
                    - labels corresponding to `X`
                    - if `np.ndarray`
                        - will be used as a colormap
                    - if `str`
                        - will be interpreted as that specific color
                    - the default is `None`
                        - will be set to `'C0'`
                - `featurenames`
                    - `np.ndarray`, optional
                    - names to give to the features present in `X`
                    - the deafault is `None`
                - `corner_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroLuSt.visualization.plots.CornerPlot.plot()`
                    - the default is `None`
                        - will initialize with `{}`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - the created matplotlib figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
            Comments
            --------
        """

        if y is None: y = 'C1'
        if corner_kwargs is None: corner_kwargs = {}

        CP = CornerPlot()
        fig, axs = CP.plot(
            X=X, y=y, featurenames=featurenames,
            **corner_kwargs,
        )

        return fig, axs

    def generated_1d(self,
        generator:Callable,
        z0:list,
        zi_f:Union[list,int],
        z0_idx:int=0,
        plot_func:str=None,
        subplots_kwargs:dict=None, predict_kwargs:dict=None, plot_func_kwargs:dict=None,
        verbose:int=None,
        ) -> Tuple[Figure,List[plt.Axes]]:
        """
            - method to actually generate a plot showing samples out of the latent space while varying 1 latent variable

            Parameters
            ----------
                - `generator`
                    - `Callable class`
                    - has to implement a predict method
                    - will be called to generate samples from the provided latent variables
                        - the latent variables are a list of the length `len(zi_f)+1`
                            - i.e. the `zi_f` fixed variables + the iterated `z0` latent variable
                - `z0`
                    - `list`
                    - values of one of the latent dimensions interpretable by generator
                    - will be iterated over and used to generate a grid
                    - has to be equally spaced
                - `zi_f`
                    - `list`, `int`
                    - fixed values of all latent dimensions which are not `z0`
                    - if a list gets passed
                        - has to contain as many entries as the list that get interpreted by `generator.predict - 1`
                        - each entry represents one latent dimensions value in order
                    - if an integer gets passed
                        - has to be of same value as the length of the list that get interpreted by generator.predict - 1
                        - will be initialized with a list of zeros length `zi_f`
                - `z0_idx`
                    - `int`, optional
                    - index of where in the list of latent dimensions `z0` is located
                    - has to differ from `z1_idx`
                    - the deafult is 0
                        - will set `z0` as the first element of the latent vector 
                - `plot_func`
                    - `Callable`, optional
                    - function to use for visualizing each individual sample
                    - has to take at least 2 positional arguments
                        - `ax`
                            - `plt.Axes`
                            - axis to plot onto
                        - `X`
                            - `np.ndarray`
                            - dataseries to be plotted
                    - has to take `**kwargs`
                    - overrides `self.plot_func`
                    - the default is `None`
                        - will fall back to `self.plot_func()`
                - `subplots_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `plt.subplots()`
                    - overwrites `self.subplot_kwargs`
                    - the default is `None`
                        - will default to `self.subplot_kwargs`
                - `predict_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `generator.predict()`
                        - generator is a parameter passed `self.plot()`
                    - overwrites `self.predict_kwargs`
                    - the default is `None`
                        - will default to `self.predict_kwargs`
                - `plot_func_kwargs`
                    - `dict`, optional
                    - kwargs to pass to the function passed to `plot_func`
                    - overwrites `self.plot_func_kwargs`
                    - the default is `None`
                        - will default to `self.plot_func_kwargs`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overwrites `self.verbose`
                    - the default is `None`
                        - will default to `self.verbose`
                    
            Raises
            ------
                - `UserWarning`
                    - if `z0` is not equally spaced

            Returns
            -------
                - `fig`
                    - `Figure`
                    - created figure
                - `axs`
                    - `list(plt.Axes)`
                    - axes corresponding to `fig`
                    - contains an axes to set labels and title as last entry

            Comments
            --------
                - if you want to show something in the background simply follow this structure:

                ```python
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_1d(...)
                >>> axs[-1].plot(...)
                >>> plt.show()
                ```
                
                - you can also use function that take plt.Axes as parameters. The following example plots some decision boundary estmiate in the background of the latent space examples:
                
                ```python
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_1d(...)
                >>> plot_dbe(..., ax=axs[-1])
                >>> plt.show()
                ```
        """
    
        #check shapes
        if np.any((np.diff(np.diff(z0))) > 1e-8):
             warnings.warn(f'"z0" has to be equally spaced!')
        
        #initialize properly
        z0         = np.array(z0)
        if isinstance(zi_f, int):
            zi_f   = np.zeros(zi_f)
        else:
            zi_f       = zi_f
        if plot_func is None:           plot_func           = self.plot_func
        if subplots_kwargs is None:     subplots_kwargs     = self.subplots_kwargs
        if predict_kwargs is None:      predict_kwargs      = self.predict_kwargs
        if plot_func_kwargs is None:    plot_func_kwargs    = self.plot_func_kwargs
        if verbose is None:             verbose             = self.verbose
        
        #get indices of fixed zi
        zi_f_idxs = [i for i in range(len(zi_f)+1) if i != z0_idx]



        #temporarily disable autolayout
        plt.rcParams['figure.autolayout'] = False


        fig, axs = plt.subplots(nrows=1, ncols=z0.shape[0], **subplots_kwargs)
        

        tit = ', '.join([f'z{fzi_idx}: {fzi}' for fzi_idx, fzi in zip(zi_f_idxs, zi_f)])

        #subplot for axis labels
        ax0 = fig.add_subplot(111, zorder=-1)
        ax0.set_title(f'Latent Space\n({tit})')
        ax0.set_xlabel(f'z[{z0_idx}]')
        ax0.set_yticks([])
        #set ticks and ticklabels correctly
        ax0.set_xlim(np.min(z0)-np.mean(np.diff(z0))/2, np.max(z0)+np.mean(np.diff(z0))/2)
        ax0.xaxis.set_major_locator(mticker.MaxNLocator(len(z0), prune='both'))        


        for col, z0i in enumerate(z0):

            #construct latent-vector - fixed_zi never change, variable_zi get iterated over
            z = np.zeros((np.shape(zi_f)[0]+1))
            z[z0_idx] = z0i
            z[zi_f_idxs] = zi_f
            z_sample = np.array([z])

            try:
                x_decoded = generator.predict(z_sample, **predict_kwargs)
            except Exception as e:
                raise ValueError(f'"generator" has to implement a "predict()" method that takes a list of len(zi_f)+1 parameters as input!')


            if x_decoded.shape[0] == 1: x_decoded = x_decoded.flatten()

            plot_func(axs[col], x_decoded, **plot_func_kwargs)

            #hide labels of latent samples
            # axs[col].set_title(z0i)
            axs[col].set_xlabel('')
            axs[col].set_xticks([])
            axs[col].set_yticks([])
            
            #set subplot background transparent (to show content of ax0)
            axs[col].patch.set_alpha(0.0)

        #get axes
        axs = fig.axes

        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.rcParams['figure.autolayout'] = True

        return fig, axs

    def generated_2d(self,
        generator:Callable,
        z0:list, z1:list,
        zi_f:Union[list,int],
        z0_idx:int=0, z1_idx:int=1,
        plot_func:Callable=None,
        subplots_kwargs:dict=None, predict_kwargs:dict=None, plot_func_kwargs:dict=None,
        verbose:int=None,
        ) -> Tuple[Figure, List[plt.Axes]]:
        """
            - method to actually generate a plot showing samples out of the latent space while varying 2 latent variables

            Parameters
            ----------
                - `generator`
                    - `Callable class`
                    - has to implement a `predict` method
                    - will be called to generate samples from the provided latent variables
                        - the latent variables are a list of the length `len(zi_f)+2`
                            - i.e. the `zi_f` fixed variables + the iterated `z0` and `z1` variables 
                - `z0`
                    - `list`
                    - values of one of the latent dimensions interpretable by generator
                    - will be iterated over and used to generate a grid
                    - has to be equally spaced
                - `z1`
                    - `list`
                    - values of the second latent dimension interpretable by generator
                    - will be iterated over and used to generate a grid
                    - has to be equally spaced
                - `zi_f`
                    - `list`, `int`
                    - fixed variables of all latent dimensions which are not `z0` and `z1`
                    - if a `list` gets passed
                        - has to contain as many entries as the list that get interpreted by `generator.predict - 2`
                        - each entry represents one latent dimensions value in order
                    - if an `integer` gets passed
                        - has to be of same value as the length of the list that get interpreted by generator.predict - 2
                        - will be initialized with a list of zeros length `zi_f`
                - `z0_idx`
                    - `int`, optional
                    - index of where in the list of latent dimensions `z0` is located
                    - has to differ from `z1_idx`
                    - the deafult is `0`
                        - will set `z0` as the first element of the latent vector 
                - `z1_idx`
                    - `int`, optional
                    - index of where in the list of latent dimensions `z1` is located
                    - has to differ from `z0_idx`
                    - the deafult is `1`
                        - will set `z1` as the second element of the latent vector 
                - `plot_func`
                    - `Callable`, optional
                    - function to use for visualizing each individual sample
                    - has to take at least 2 positional arguments
                        - `ax`
                            - `plt.Axes`
                            - axis to plot onto
                        - `X`
                            - `np.ndarray`
                            - dataseries to be plotted
                    - has to take `**kwargs`
                    - overrides `self.plot_func`
                    - the default is `None`
                        - will fall back to `self.plot_func()`
                - `subplots_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `plt.subplots()`
                    - overwrites `self.subplot_kwargs`
                    - the default is `None`
                        - will default to `self.subplot_kwargs`
                - `predict_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `generator.predict()`
                        - generator is a parameter passed `self.plot()`
                    - overwrites `self.predict_kwargs`
                    - the default is `None`
                        - will default to `self.predict_kwargs`
                - `plot_func_kwargs`
                    - `dict`, optional
                    - kwargs to pass to the function passed to `plot_func`
                    - overwrites `self.plot_func_kwargs`
                    - the default is `None`
                        - will default to `self.plot_func_kwargs`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overwrites `self.verbose`
                    - the default is `None`
                        - will default to `self.verbose`
                    
            Raises
            ------
                - `UserWarning`
                    - if `z0` or `z1` is not equally spaced
                - `ValueError`
                    - if `z0_idx` and `z1_idx` have the same value

            Returns
            -------
                - `fig`
                    - `Figure`
                    - created figure
                - `axs`
                    - `List[plt.Axes]`
                    - axes corresponding to `fig`
                    - contains an axis to set labels and title as last entry

            Comments
            --------
                - if you want to show something in the background simply follow this structure:
                
                ```python
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_2d(...)
                >>> axs[-1].plot(...)
                >>> plt.rcParams['figure.autolayout'] = False #to keep no whitespace between generated samples
                >>> plt.show()
                ```

                - you can also use function that take plt.Axes as parameters. The following example plots some decision boundary estmiate in the background of the latent space examples:
                
                ```python
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_2d(...)
                >>> plot_dbe(..., ax=axs[-1])
                >>> plt.rcParams['figure.autolayout'] = False #to keep no whitespace between generated samples
                >>> plt.show()
                ```

        """

        #check shapes
        if np.any((np.diff(np.diff(z0))) > 1e-8) or np.any(np.diff((np.diff(z1))) > 1e-8):
            warnings.warn(f'"z0" and "z1" have to be equally spaced!')
        if z0_idx == z1_idx:
            raise ValueError(
                f'"z0_idx" and "z1_idx" have to have different values!\n'
                f'    If you want to only vary one latent dimension use plot_1d()!'
            )

            
        #initialize properly
        z0         = np.array(z0)
        z1         = np.array(z1)
        if isinstance(zi_f, int):
            zi_f   = np.zeros(zi_f)
        else:
            zi_f       = zi_f
        if plot_func is None:           plot_func           = self.plot_func
        if subplots_kwargs is None:     subplots_kwargs     = self.subplots_kwargs
        if predict_kwargs is None:      predict_kwargs      = self.predict_kwargs
        if plot_func_kwargs is None:    plot_func_kwargs    = self.plot_func_kwargs
        if verbose is None:             verbose             = self.verbose
        

        #get indices of fixed zi
        zi_f_idxs = [i for i in range(len(zi_f)+2) if i not in [z0_idx, z1_idx]]



        #temporarily disable autolayout
        plt.rcParams['figure.autolayout'] = False


        fig, axs = plt.subplots(nrows=z1.shape[0], ncols=z0.shape[0], **subplots_kwargs)
        

        tit = ', '.join([f'z[{fzi_idx}]: {fzi}' for fzi_idx, fzi in zip(zi_f_idxs, zi_f)])

        z0_step = np.max(z0)-np.min(z0)
        z1_step = np.max(z1)-np.min(z1)

        #subplot for axis labels
        ax0 = fig.add_subplot(111, zorder=-1)
        ax0.set_title(f'Latent Space\n({tit})')
        ax0.set_xlabel(f'z[{z0_idx}]')
        ax0.set_ylabel(f'z[{z1_idx}]')

        #set ticks and ticklabels correctly
        ax0.set_ylim(np.min(z1)-np.mean(np.diff(z1))/2, np.max(z1)+np.mean(np.diff(z1))/2)
        ax0.set_xlim(np.min(z0)-np.mean(np.diff(z0))/2, np.max(z0)+np.mean(np.diff(z0))/2)
        ax0.yaxis.set_major_locator(mticker.MaxNLocator(len(z1), prune='both'))
        ax0.xaxis.set_major_locator(mticker.MaxNLocator(len(z0), prune='both'))

        for row, z1i in enumerate(z1[::-1]):
            for col, z0i in enumerate(z0):

                #construct latent-vector - fixed_zi never change, variable_zi get iterated over
                z = np.zeros((np.shape(zi_f)[0]+2))
                z[z0_idx] = z0i
                z[z1_idx] = z1i
                z[zi_f_idxs] = zi_f



                z_sample = np.array([z])

                try:
                    x_decoded = generator.predict(z_sample, **predict_kwargs)
                except Exception as e:
                    raise ValueError(f'"generator" has to implement a "predict()" method that takes a list of len(zi_f)+2 parameters as input!')


                if x_decoded.shape[0] == 1: x_decoded = x_decoded.flatten()

                plot_func(axs[row,col], x_decoded, **plot_func_kwargs)

                #hide labels of latent samples
                # axs[row, col].set_title(f'{z0i},{z1i}')
                axs[row, col].set_xlabel('')
                axs[row, col].set_ylabel('')
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])

                #set subplot background transparent (to show content of ax0)
                axs[row, col].patch.set_alpha(0.0)

        #get axes
        axs = fig.axes

        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.rcParams['figure.autolayout'] = True

        return fig, axs

class MultiConfusionMatrix:
    """
        - class to create a multi-model confusion matrix
        - inspired by the Weights&Biases multi-confusion-matrix plot 
            - https://wandb.ai/wandb/plots/reports/Confusion-Matrix-Usage-and-Examples--VmlldzozMDg1NTM (last access: 2023/07/20)

        Attributes
        ----------
            - `score_decimals`
                - `int`, optional
                - number of decimals to round `score` to when displaying
                - only relevant if `m_labels == 'score'`
                - the default is `2`
            - `text_colors` 
                - `str`, `tuple`, `list`, optional
                - colors to use for displaying
                    - model/bar labels in `plot_func='multi'`
                    - cell-values in `plot_func='single'`
                - if `str`
                    - will use that color for all bars/cells
                - if `tuple`
                    - has to be RGBA tuple
                    - will use that color for all bars/cells
                - if `list`
                    - for `plot_func='multi'`
                        - will use entry 0 for bar 0, entry 1 for bar 1 ect.
                    - for `plot_func='single'`
                        - length has to be equal to `confmat.size`
                        - will display colors from top-left to bottom right (in reading direction)                    
                - the default is `None`
                    - will autogenerate colors
            - `cmap`
                - `str`, `mcolors.Colormap`, optional
                - colormap to use for coloring the different models
                - the default is `None`
                    - will be set to `'nipy_spectral'`
            - `vmin`
                - `float`, optional
                - minimum value of the colormapping
                - used in scaling the colormap
                - argument of `astroLuSt.visualization.plotting.generate_colors()`
                - the default is `None`
            - `vmax`
                - `float`, optional
                - maximum value of the colormapping
                - used in scaling the colormap
                - argument of `astroLuSt.visualization.plotting.generate_colors()`
                - the default is `None`
            - `vcenter`
                - `float`, optional
                - center value of the colormapping
                - used in scaling the colormap
                - argument of `astroLuSt.visualization.plotting.generate_colors()`
                - the default is `None`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the deafault is `0`
            - `fig_kwargs`
                - `dict`, optional
                - kwargs to pass to `plt.figure()`
                - the default is `None`
                    - will be set to `dict(figsize=(9,9))`
            - `text_kwargs`
                - `dict`, optional
                - kwargs to pass to `ax.text()`
                - the default is `None`
                    - will be set to `dict()`                

        Methods
        -------
            - `__pad()`
            - `get_multi_confmat()`
            - `plot_bar()`
            - `plot_singlemodel()`
            - `plot_multimodel()`
            - `plot_result()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `sklearn`
            - `typing`
            - `warnings`
        
        Comments
        --------

    """

    def __init__(self,
        score_decimals:int=2,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None, vcenter:float=None,
        verbose:int=0,
        fig_kwargs:dict=None,
        text_kwargs:dict=None,
        ) -> None:

        self.score_decimals = score_decimals
        self.text_colors    = text_colors
        if cmap is None:        self.cmap       = 'nipy_spectral'
        else:                   self.cmap       = cmap
        self.vmin       = vmin
        self.vmax       = vmax
        self.vcenter    = vcenter
        self.verbose    = verbose
        if fig_kwargs is None:  self.fig_kwargs = dict(figsize=(9,9))
        else:                   self.fig_kwargs = fig_kwargs
        if text_kwargs is None: self.text_kwargs = dict()
        else:                   self.text_kwargs = text_kwargs
        
        return

    def __repr__(self):

        return (
            f'MultiConfusionMatrix(\n'
            f'    score_decimals={repr(self.score_decimals)},\n'
            f'    cmap={repr(self.cmap)}, vmin={self.vmin}, vmax={self.vmax}, vcenter={self.vcenter},\n'
            f'    verbose={repr(self.verbose)},\n'
            f'    fig_kwargs={repr(self.fig_kwargs)},\n'
            f')'
        )

    def __pad(self,
        y_true:Union[np.ndarray,list], y_pred:Union[np.ndarray,list],
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - private method to pad all arrays in `y_true` and `y_pred` to have the same length

            Parameters
            ----------
                - `y_true`
                    - `np.ndarray`, `list`, optional
                    - ground truth labels
                    - has to be 2d
                        - `shape = (nmodels,nsampels)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `y_pred`
                    - `np.ndarray`, optional
                    - model predictions
                    - has to be 2d
                        - `shape = (nmodels,nsamples)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`

            Raises
            ------

            Returns
            -------
                - `y_true_pad`
                    - `np.ndarray`, optional
                    - padded version of `y_true`
                - `y_pred_pad`
                    - `np.ndarray`, optional
                    - padded version of `y_pred`            

            Comments
            --------
        """
        
        maxlen = np.max(np.array([[len(yt),len(yp)] for yt, yp in zip(y_true, y_pred)]))

        y_true_pad = np.full((len(y_true), maxlen), np.nan, dtype=np.float64)
        y_pred_pad = np.full((len(y_pred), maxlen), np.nan, dtype=np.float64)
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            y_true_pad[idx, :yt.shape[0]] = yt
            y_pred_pad[idx, :yp.shape[0]] = yp
        
        return y_true_pad, y_pred_pad
    

    def get_multi_confmat(self,
        y_true:Union[np.ndarray,list], y_pred:Union[np.ndarray,list],
        sample_weight:np.ndarray=None,
        normalize:Literal['true','pred','all']=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to generate a multi-confusion matrix

            Parameters
            ----------
                - `y_true`
                    - `np.ndarray`, `list`, optional
                    - ground truth labels
                    - has to be 2d
                        - `shape = (nmodels,nsampels)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `y_pred`
                    - `np.ndarray`, `list`, optional
                    - model predictions
                    - has to be 2d
                        - `shape = (nmodels,nsamples)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `sample_weight`
                    - `np.ndarray`, optional
                    - sample weights to use in `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                - `normalize`
                    -  `Literal['true','pred','all']`, optional
                    - how to normalize the confusion matrix
                    - if `'true'` is passed
                        - normalize w.r.t. `y_true`
                    - if `'pred'` is passed
                        - normalize w.r.t. `y_pred`
                    - if `'all'` is passed
                        - normalize w.r.t. all confusion matrix cells
                    - will be passed to `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                        - no normalization
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the deafault is `None`
                        - will fall back to `self.verbose`

            Raises
            ------
                - `ValueError`
                    - if `y_true` and `y_pred` have a different lengths

            Returns
            -------
                - `multi_confmat`
                    - `np.ndarray`
                    - 3d of shape `(nmodels,nclasses,nclasses)`
                    - multi-confusion-matrix for the ensemble of models

            Comments
            --------
        """

        #default parameters
        if verbose is None: verbose = self.verbose

        #pad if 2d and necessary
        try:
            _ = y_true[0][0], y_pred[0][0]      #check if 2d
            y_true, y_pred = self.__pad(y_true=y_true, y_pred=y_pred)
        except:
            #make sure y_true and y_pred have the same length
            if len(y_true) != len(y_pred):
                raise ValueError(
                    f'If `y_true` and `y_pred` are 1d, they have to have the same lengths but have {len(y_true)} and {len(y_pred)}!'
                )
            else:
                pass

        #convert to numpy array (if lists got passed)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        #check if 1d array was passed
        if len(y_true.shape) < 2:
            y_true = y_true.reshape(1,-1)
            warnings.warn(message=(
                f'`y_true` has to be two dimensional but has shape {y_true.shape}!\n'
                f'    Called `y_true.reshape(1,-1)`. Therefore you might not get the expected result.'
            ))
        if len(y_pred.shape) < 2:
            y_pred = y_pred.reshape(1,-1)
            warnings.warn(message=(
                f'`y_pred` has to be two dimensional but has shape {y_pred.shape}!'
                f'    Called `y_pred.reshape(1,-1)`. Therefore you might not get the expected result.'
            ))

        #get unique labels
        uniques = np.unique([y_true,y_pred])
        uniques = uniques[np.isfinite(uniques)]

        #initialize multi-confusion matrix
        multi_confmat = np.zeros((len(y_true), len(uniques), len(uniques)))

        #get confusion matrix for each model
        for midx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            
            #remove non-finite values (padded values ect.)
            finite_bool = (np.isfinite(yt)&np.isfinite(yp))
            yt = yt[finite_bool]
            yp = yp[finite_bool]
            
            #get mapping of current unique labels to indices
            c_uniques = np.unique([yt,yp])
            idx_map = {c:np.where(uniques==c)[0][0] for c in c_uniques}
            
            #caluculate confusion matrix for current model (inverse definition to sklearn)
            cm = confusion_matrix(y_true=yt, y_pred=yp, normalize=normalize, sample_weight=sample_weight).T
            
            #include calculated confusion matrix in multi-confusion matrix
            for iidx, cui in enumerate(c_uniques):
                for jidx, cuj in enumerate(c_uniques):
                    multi_confmat[midx, idx_map[cui], idx_map[cuj]] = cm[iidx,jidx]
            
            #show resulting matrix
            if verbose > 3:
                CMD = ConfusionMatrixDisplay(multi_confmat[midx], display_labels=uniques)
                CMD.plot()
                plt.show()

        return multi_confmat

    def plot_bar(self,
        ax:plt.Axes,
        score:np.ndarray,
        m_labels:Union[list,Literal['score']]=None, score_decimals:int=None,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None, vcenter:float=None,
        text_kwargs:dict=None,
        ) -> None:
        """
            - method to create a bar-plot in one panel (`ax`)
                - i.e. confusion matrix cell of one class-combination

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - axes to plot onto
                - `score`
                    - `np.ndarray`
                    - scores of one class-combination for all models
                - `m_labels`
                    - `np.ndarray`, `Literal['score']`, optional
                    - labels to show for the different models (entries in `score`)
                    - if `'score'` is passed
                        - will show the respective model's (normalized) score                    
                    - the default is `None`
                        - no labels shown
                - `score_decimals`
                    - `int`, optional
                    - number of decimals to round each model's `score` to when displaying
                    - only relevant if `m_labels == 'score'`
                    - overrides `self.score_decimals` 
                    - the default is `None`
                        - will fall back to `self.score_decimals`
                - `text_colors` 
                    - `str`, `tuple`, `list`, optional
                    - colors to use for displaying model/bar labels
                    - if `str`
                        - will use that color for all bars
                    - if `tuple`
                        - has to be RGBA `tuple`
                        - will use that color for all bars
                    - if `list`
                        - will use entry 0 for bar 0, entry 1 for bar 1 ect.
                    - the default is `None`
                        - will autogenerate colors
                        - will use the the last color of `cmap` for the first half of the bars
                        - will use the the first color of `cmap` for the bottom half of the bars
                - `cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to use for coloring the different models
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `vmin`
                    - `float`, optional
                    - minimum value of the colormapping
                    - used in scaling the colormap
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - `float`, optional
                    - maximum value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `vcenter`
                    - `float`, optional
                    - center value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vcenter`
                    - the default is `None`
                        - will fall back to `self.vcenter`
                - `text_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.text()`
                    - overwrites `self.text_kwargs`
                    - the default is `None`
                        - will fall back to `self.text_kwargs`

            Raises
            ------
                - `TypeError`
                    - if `m_labels` is of the wrong type

            Returns
            -------

            Comments
            --------
        """

        #default parameters
        if m_labels == 'score': m_labels = np.round(score, score_decimals)
        elif isinstance(m_labels, (list, np.ndarray)): m_labels = m_labels
        elif m_labels is None:  m_labels = []
        else: raise TypeError('`m_labels` has to be either a list, np.ndarray, or `"score"`')
        if text_kwargs is None: text_kwargs = self.text_kwargs

        #generate colors for the bars and text (m_labels)
        colors = alvp.generate_colors(len(score)+1, vmin, vmax, vcenter, cmap=cmap)
        
        if text_colors is None:
            text_colors = colors.copy()
            text_colors[:len(text_colors)//2] = colors[-1]
            text_colors[len(text_colors)//2:] = colors[0]
        elif isinstance(text_colors, (str, tuple)):
            text_colors = [text_colors]*score.shape[0]

        
        #create barplor
        bars = ax.barh(
            y=np.arange(score.shape[0])[::-1], width=score,
            color=colors,
        )

        #add model labels if desired
        for idx, (b, mlab, tc) in enumerate(zip(bars, m_labels, text_colors)):
            ax.text(
                x=0.01*max(ax.get_xlim()), y=b.get_y()+b.get_height()/2,
                s=mlab,
                c=tc, va='center',
                **text_kwargs,
            )
        
        ax.grid(visible=True, axis='x')

        return

    def plot_singlemodel(self,
        confmat:np.ndarray,
        labels:np.ndarray=None, score_decimals:int=None,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None,
        fig_kwargs:dict=None,
        pcolormesh_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to produce classic confusion matrix
            - similar to `sklearn.metrics.ConfusionMatrixDisplay`
                - BUT axes defined inversely
                - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay (last access: 2023/07/20)

            Parameters
            ----------
                - `confmat`
                    - `np.ndarray`
                    - array containing confusion matrix
                    - has to be of shape `(nclasses, nclasses)`
                - `labels`
                    - `np.ndarray`, optional
                    - labels to display for different classes present in `y_true` and `y_pred`
                    - will assign labels in ascending orders for the values present in `y_true` and `y_pred`
                    - the default is `None`
                        - will generate labels using `np.arange(confmats.shape[-1])`
                - `score_decimals`
                    - `int`, optional
                    - number of decimals to round each model's `score` to when displaying
                    - only relevant if `m_labels == 'score'`
                    - overrides `self.score_decimals` 
                    - the default is `None`
                        - will fall back to `self.score_decimals`
                - `text_colors` 
                    - `str`, `tuple`, `list`, optional
                    - colors to use displaying text in each cell
                    - if `str`
                        - will use that color for all cells
                    - if `tuple`
                        - has to be RGBA tuple
                        - will use that color for all cells
                    - `list`
                        - length has to be equal to `confmat.size`
                        - will display colors from top-left to bottom right (in reading direction)                    
                    - overwrites `self.text_colors`
                    - the default is `None`
                        - will fall back to `self.text_colors`
                        - will autogenerate colors
                            - inverse to cmap
                - `cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to use for coloring the different models
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `vmin`
                    - `float`, optional
                    - minimum value of the colormapping
                    - used in scaling the colormap
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - `float`, optional
                    - maximum value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `fig_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `plt.figure()`
                    - overrides `self.fig_kwargs`
                    - the default is `None`
                        - will fall back to `self.fig_kwargs`
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

        if text_colors is None:         text_colors         = self.text_colors
        if cmap is None:                cmap                = self.cmap
        if score_decimals is None:      score_decimals      = self.score_decimals
        if fig_kwargs is None:          fig_kwargs          = self.fig_kwargs
        if pcolormesh_kwargs is None:   pcolormesh_kwargs   = dict()

        #generate colors for cell text
        colors = alvp.generate_colors(confmat.size, vmin=vmin, vmax=vmax, cmap=cmap)

        if text_colors is None:
            text_colors = np.empty_like(colors)
            idxs = np.argsort(confmat.flatten())[::-1]
            text_colors[idxs] = colors
            text_colors = text_colors.reshape(-1,confmat.shape[-1],4)
        elif isinstance(text_colors, str):
            text_colors = np.full(confmat.shape, text_colors)
        elif isinstance(text_colors, tuple):
            text_colors = np.full((*confmat.shape,4), text_colors)
        else:
            assert len(text_colors) == confmat.size, f'the length of `text_colors` has to be equal to `confmat.size`!'
            text_colors = np.array(text_colors)
            text_colors = text_colors.reshape(*confmat.shape,-1)


        #coordinates for plotting
        x = np.arange(confmat.shape[-1])
        
        #plot
        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(111)

        #plot confmat
        mesh = ax1.pcolormesh(x, x, confmat, cmap=cmap, vmin=vmin, vmax=vmax, **pcolormesh_kwargs)

        #add text
        for row in x:
            for col in x:
                c = text_colors[row, col]
                if isinstance(c[0], str):
                    c = c[0]
                else:
                    c = c
                ax1.text(
                    x=x[col], y=x[row],
                    s=np.round(confmat[row,col], score_decimals),
                    color=c, ha='center', va='center'
                ) 

        #labelling
        ax1.set_xticks(x, labels=labels[:x.shape[0]])
        ax1.set_yticks(x, labels=labels[:x.shape[0]])

        ax1.invert_yaxis()
        ax1.set_xlabel('True')
        ax1.set_ylabel('Predicted')

        axs = fig.axes

        return fig, axs

    def plot_multimodel(self,
        confmats:np.ndarray,
        labels:np.ndarray=None, m_labels:Union[np.ndarray,Literal['score']]=None, score_decimals:int=None,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None, vcenter:float=None,
        subplots_kwargs:dict=None,
        fig_kwargs:dict=None,
        text_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to produce the confusion-matrix plot containing results of multiple models

            Parameters
            ----------
                - `confmats`
                    - `np.ndarray`
                    - array containing confusion matrices for the models
                    - has to be of shape `(nmodels, nclasses, nclasses)`
                - `labels`
                    - `np.ndarray`, optional
                    - labels to display for different classes present in `y_true` and `y_pred`
                    - will assign labels in ascending orders for the values present in `y_true` and `y_pred`
                    - the default is `None`
                        - will generate labels using `np.arange(confmats.shape[-1])`
                - `m_labels`
                    - `np.ndarray`, `Literal['score']`, optional
                    - labels to show for the different models (axis 1 of `y_true` and `y_pred`)
                    - if `'score'` is passed
                        - will show the respective model's (normalized) score                    
                    - the default is `None`
                        - no labels shown
                - `score_decimals`
                    - `int`, optional
                    - number of decimals to round each model's `score` to when displaying
                    - only relevant if `m_labels == 'score'`
                    - overrides `self.score_decimals` 
                    - the default is `None`
                        - will fall back to `self.score_decimals`
                - `text_colors` 
                    - `str`, `tuple`, `list`, optional
                    - colors to use for displaying model/bar labels
                    - if `str`
                        - will use that color for all bars
                    - if `tuple`
                        - has to be RGBA tuple
                        - will use that color for all bars
                    - if `list`
                        - will use entry 0 for bar 0, entry 1 for bar 1 ect.
                    - overwrites `self.text_colors`
                    - the default is `None`
                        - will fall back to `self.text_colors`
                - `cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to use for coloring the different models
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `vmin`
                    - `float`, optional
                    - minimum value of the colormapping
                    - used in scaling the colormap
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - `float`, optional
                    - maximum value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `vcenter`
                    - `float`, optional
                    - center value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vcenter`
                    - the default is `None`
                        - will fall back to `self.vcenter`
                - `subplots_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `plt.subplots()`
                    - the default is `None`
                        - will be set to `dict(sharex='all', sharey='all')`
                - `fig_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `plt.figure()`
                    - overrides `self.fig_kwargs`
                    - the default is `None`
                        - will fall back to `self.fig_kwargs`
                - `text_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.text()`
                    - overwrites `self.text_kwargs`
                    - the default is `None`
                        - will fall back to `self.text_kwargs`
                        
            Raises
            ------
                - `ValueError`
                    - if the length of `labels` is to low
                        - i.e. less than the unique elements in the combined set of `y_true` and `y_pred`

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

        #default parameters
        if m_labels is None:        m_labels        = []
        if score_decimals is None:  score_decimals  = self.score_decimals
        if text_colors is None:     text_colors     = self.text_colors
        if cmap is None:            cmap            = self.cmap
        if subplots_kwargs is None: subplots_kwargs = dict(sharex='all', sharey='all')
        if fig_kwargs is None:      fig_kwargs      = self.fig_kwargs
        if text_kwargs is None:     text_kwargs     = self.text_kwargs

        nrowscols = confmats.shape[-1]
        if labels is None: labels                   = np.arange(nrowscols)
        #catch errors
        if len(labels) < nrowscols: raise ValueError(f'`labels` has to be at least of length equal to the number of unique classes in `y_true` and `y_pred` ({nrowscols}) but has length {len(labels)}!')


        #plotting
        fig, axs = plt.subplots(
            nrows=nrowscols, ncols=nrowscols,
            **subplots_kwargs,
            **fig_kwargs,
        )
        
        for row in range(nrowscols):
            for col in range(nrowscols):

                #plot barchart
                self.plot_bar(
                    ax=axs[row,col],
                    score=confmats[:,row,col], m_labels=m_labels, score_decimals=score_decimals,
                    text_colors=text_colors,
                    cmap=cmap, vmin=vmin, vmax=vmax, vcenter=vcenter,
                    text_kwargs=text_kwargs
                )

                #set axis labels
                if col == 0:            axs[row,col].set_ylabel(labels[row])
                if row == nrowscols-1:  axs[row,col].set_xlabel(labels[col])
                
                axs[row,col].set_yticklabels([])

        #figure labels
        plt.figtext(0.5, 0.0, 'True',      rotation=0 , fontsize='large')
        plt.figtext(0.0, 0.5, 'Predicted', rotation=90, fontsize='large')

        plt.tight_layout()

        axs = fig.axes

        return fig, axs

    def plot_result(self,
        y_true:Union[np.ndarray,list], y_pred:Union[np.ndarray,list],
        confmats:np.ndarray=None,
        labels:np.ndarray=None,
        sample_weight:np.ndarray=None,
        normalize:Literal['true','pred','all']=None,
        plot_func:Literal['multi', 'single', 'auto']='auto',
        verbose:int=None,
        plot_multimodel_kwargs:dict=None,
        plot_singlemodel_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to produce the plot of the multi-confusion-matrix

            Parameters
            ----------
                - `y_true`
                    - `np.ndarray`, `list`, optional
                    - ground truth labels
                    - has to be 2d
                        - `shape = (nmodels,nsampels)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `y_pred`
                    - `np.ndarray`, `list`, optional
                    - model predictions
                    - has to be 2d
                        - `shape = (nmodels,nsamples)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `confmats`
                    - `np.ndarray`, optional
                    - array containing confusion matrices for the models
                    - has to be of shape `(nmodels, nclasses, nclasses)`            
                    - if `y_true` and `y_pred` are also not None
                        - will use `y_true` and `y_pred` instead
                    - the default is `None`
                        - will use `y_true` and `y_pred` instead
                - `labels`
                    - `np.ndarray`, optional
                    - labels to display for different classes present in `y_true` and `y_pred`
                    - will assign labels in ascending orders for the values present in `y_true` and `y_pred`
                    - the default is `None`
                        - will autogenerate labels
                            - will use the unique values present in `y_true` and `y_pred` if both are not `None`
                            - will use `np.arange(confmats.shape[-1])` if `y_true` and `y_pred` are both `None`
                - `sample_weight`
                    - `np.ndarray`, optional
                    - sample weights to use in `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                - `normalize`
                    -  `Literal['true','pred','all']`, optional
                    - how to normalize the confusion matrix
                    - if `'true'` is passed
                        - normalize w.r.t. `y_true`
                    - if `'pred'` is passed
                        - normalize w.r.t. `y_pred`
                    - if `'all'` is passed
                        - normalize w.r.t. all confusion matrix cells
                    - will be passed to `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                        - no normalization
                - `plot_func`
                    - `Literal['auto','multi','single']`, optional
                    - method to use for deciding how to display the confusion matrix
                    - if `'auto'`
                        - will automatically decide
                    - if `'multi'`
                        - will use `self.plot_multimodel()` even if only one model passed
                    - if `single`
                        - will use `self.plot_singlemodel()`
                        - will plot the first entry of confmats even if multiple are passed
                            - i.e. `confmats[0]`
                    - the default is `auto`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the deafault is `None`
                        - will fall back to `self.verbose`
                - `plot_multimodel_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.plot_multimodel()`
                    - the default is `None`
                        - will be set to `dict()`
                - `plot_singlemodel_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.plot_singlemodel()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------
                - `ValueError`
                    - if `y_true`, `y_pred`, and `confmats` are all `None`
                    - if `confmats` is not a square matrix
                    - if a wrong value is passed as `plot_func`

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
        
        #default values
        if plot_multimodel_kwargs is None:  plot_multimodel_kwargs  = dict()
        if plot_singlemodel_kwargs is None: plot_singlemodel_kwargs = dict()

        #get confusion matrices for all models (Transpose because sklearn.metric.confusion_matrix is inversely defined to this method)
        #initialize labels
        if y_true is not None and y_pred is not None:
            
            #generate multi-confusion matrix
            confmats = self.get_multi_confmat(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, normalize=normalize, verbose=verbose)

            if labels is None: labels = np.unique([y_true, y_pred])
            
        elif y_true is None or y_pred is None and confmats is not None:
            if labels is None: labels = np.arange(confmats.shape[-1])
            confmats = confmats
        else:
            raise ValueError(f'Either `y_true` and `y_pred` have to be not `None` or `confmats` has to be not `None`, but all are `None`!')
        
        #reshape confmats if wrong shape has been passed
        if len(confmats.shape) != 3:
            confmats = confmats.reshape(1, *confmats.shape)
        
        #check if all shapes are correct
        if confmats.shape[1] != confmats.shape[2]: raise ValueError(f'Confusion matrices have to be square matrices but `confmats` has shape {confmats.shape}')

        #decide on plotting strategy
        if plot_func == 'auto':
            if confmats.shape[0] == 1:  plot_func = 'single'
            else:                       plot_func = 'multi'

        #create plots
        if plot_func == 'multi':
            fig, axs = self.plot_multimodel(
                confmats=confmats,
                labels=labels,
                **plot_multimodel_kwargs,
            )
        elif plot_func == 'single':
            fig, axs = self.plot_singlemodel(
                confmats[0],
                labels=labels,
                **plot_singlemodel_kwargs,
            )
        else:
            raise ValueError(f'`plot_func` has to be one of `["multi", "single", "auto"] but is {plot_func}')

        return fig, axs

class MultiHeadAttentionWeights:
    """
        - class to visualize attention weights of a multi-head attention (MHA) block

        Attributes
        ----------
            - `style`
                - `Literal['matrix','lines']`, optional
                -  display style to use
                - allowed choices
                    - `'matrix'`
                        - will display connection-weights as matrices
                    - `'lines'`
                        - will display connection-weights as lines connecting two features
                - the default is `None`
                    - will be set to `'matrix'`
            - `cmap`
                - `Union[str,mcolors.Colormap]`, optional
                - colormap to use for plotting
                - the default is `None`
                    - will be set to `'Blues'`
            - `cmap_norm`
                - `mcolors.Normalize`,optional
                - normalization to use for colormapping
                - the default is `'None'`
                    - will infer norm based on the passed data in `self.plot()`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
        
        Methods
        -------
            - `plot_attention_matrix()`
            - `plot_attention_lines()`
            - `plot()`
        
        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`
    
        Comments
        --------
            
    """

    def __init__(self,
        style:Literal['matrix','lines']=None,
        cmap:Union[str,mcolors.Colormap]=None, cmap_norm:mcolors.Normalize=None,
        verbose:int=0,
        ) -> None:

        if style is None:           self.style  = 'matrix'
        else:                       self.style  = style
        if cmap is None:            self.cmap   = 'Blues'
        elif isinstance(cmap, str): self.cmap   = plt.get_cmap(cmap)
        else:                       self.cmap   = cmap
        self.cmap_norm                          = cmap_norm

        self.verbose = verbose


        return

    def __repr__(self) -> str:

        return (
            f'{self.__class__.__name__}(\n'
            f'    style={repr(self.style)},\n'
            f'    cmap={repr(self.cmap)}, cmap_norm={repr(self.cmap_norm)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))


    def plot_attention_matrix(self,
        aw_h:np.ndarray,
        ax:plt.Axes,
        cmap:Union[str,mcolors.Colormap]=None, cmap_norm:mcolors.Normalize=None,
        pcolormesh_kwargs:dict=None
        ) -> None:
        """
            - method to plot the attention-weight-matrices of Multi-Head-Attention

            Parameters
            ----------
                - `aw_h`
                    - `np.ndarray`
                    - attention weights of one head
                    - has to have shape `(nfeatures,nfeatures)`
                - `ax`
                    - `plt.Axes`
                    - axis to plot into
                - `cmap`
                    - `Union[str,mcolors.Colormap]`, optional
                    - colormap to use for encoding the attention weights
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `cmap_norm`
                    - `mcolors.Normalize`, optional
                    - normalization to use for colormapping
                    - overrides `self.cmap_norm`
                    - the default is `'None'`
                        - will fall back to `self.cmap_norm`
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
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
        if cmap is None:                cmap                = self.cmap
        if isinstance(cmap, str):       cmap                = plt.get_cmap(cmap)
        if cmap_norm is None:           cmap_norm           = self.cmap_norm
        if pcolormesh_kwargs is None:   pcolormesh_kwargs   = dict()

        #get actual cmap; update cmap_norm if also self.cmap_norm is None
        if cmap_norm is None:       cmap_norm   = mcolors.Normalize(vmin=np.nanmin(aw_h), vmax=np.nanmax(aw_h))

        #plot matrices
        ax.pcolormesh(aw_h, cmap=cmap, norm=cmap_norm, **pcolormesh_kwargs)

        return
    
    def plot_attention_lines(self,
        aw_h:np.ndarray,
        ax:plt.Axes,
        cmap:Union[str,mcolors.Colormap]=None,
        cmap_norm:mcolors.Normalize=None,
        plot_kwargs:dict=None,
        ) -> None:
        """
            - method to plot the attention-weights of Multi-Head-Attention as lines

            Parameters
            ----------
                - `aw_h`
                    - `np.ndarray`
                    - attention weights of one head
                    - has to have shape `(nfeatures,nfeatures)`
                - `ax`
                    - `plt.Axes`
                    - axis to plot into
                - `cmap`
                    - `Union[str,mcolors.Colormap]`, optional
                    - colormap to use for encoding the attention weights
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `cmap_norm`
                    - `mcolors.Normalize`, optional
                    - normalization to use for colormapping
                    - overrides `self.cmap_norm`
                    - the default is `'None'`
                        - will fall back to `self.cmap_norm`
                - `plot_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.plot()`
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
        #default parameters
        if cmap is None:            cmap        = self.cmap
        if isinstance(cmap, str):   cmap        = plt.get_cmap(cmap)
        if cmap_norm is None:       cmap_norm   = self.cmap_norm
        if plot_kwargs is None:     plot_kwargs = dict()

        #get actual cmap; update cmap_norm if also self.cmap_norm is None
        if cmap_norm is None:       cmap_norm   = mcolors.Normalize(vmin=np.nanmin(aw_h), vmax=np.nanmax(aw_h))

        #plot lines
        for iidx in range(len(aw_h)):
            for jidx in range(len(aw_h[iidx])):
                ax.plot([iidx,jidx], c=cmap(cmap_norm(aw_h[iidx,jidx])), **plot_kwargs)


        #labelling
        ax.set_ylabel('Feature')

        return

    def plot(self,
        attention_weights:np.ndarray,
        featurenames:List[str]=None,
        style:Literal['matrix','lines']=None,
        cmap:Union[str,mcolors.Colormap]=None,
        cmap_norm:mcolors.Normalize=None,
        fig:Figure=None,
        plot_attention_matrix_kwargs:dict=None,
        plot_attention_lines_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the attention-weights of Multi-Head-Attention as lines

            Parameters
            ----------
                - `attention_weights`
                    - `np.ndarray`
                    - attention weights of all heads
                    - has to have shape `(nheads,nfeatures,nfeatures)`
                - `featurenames`
                    - `List[str]`
                    - names of the features stored in `attention_weights`
                    - has to have shape `(nfeatures)`
                    - the default is `None`
                        - no names displayed
                - `style`
                    - `Literal['matrix','lines']`, optional
                    -  display style to use
                    - allowed choices
                        - `'matrix'`
                            - will display connection-weights as matrices
                        - `'lines'`
                            - will display connection-weights as lines connecting two features
                    - overrides `self.style`
                    - the default is `None`
                        - will fall back to `self.style`
                - `cmap`
                    - `Union[str,mcolors.Colormap]`, optional
                    - colormap to use for encoding the attention weights
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `cmap_norm`
                    - `mcolors.Normalize`, optional
                    - normalization to use for colormapping
                    - overrides `self.cmap_norm`
                    - the default is `'None'`
                        - will fall back to `self.cmap_norm`
                        - if still `None`
                            - will be set to `mcolors.Normalize(vmin=np.nanmin(attention_weights), vmax=np.nanmax(attention_weights))`
                - `fig`
                    - `Figure`
                    - matplotlib figure to plot into
                    - the default is `None`
                        - will generate a new figure
                - `plot_attention_matrix_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.plot_attention_matrix()`
                    - the default is `None`
                        - will be set to `dict()`
                - `plot_attention_lines_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.plot_attention_lines()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - created figure
                -  `axs`
                    - `plt.Axes`
                    - axis corresponding to `fig`

            Comments
            --------
        """
       
        if style is None: style = self.style
        if cmap is None:            cmap        = self.cmap
        if isinstance(cmap, str):   cmap        = plt.get_cmap(cmap)
        if cmap_norm is None:       cmap_norm   = mcolors.Normalize(vmin=np.nanmin(attention_weights), vmax=np.nanmax(attention_weights))
        if fig is None: fig = plt.figure(figsize=(9,9))
        if plot_attention_matrix_kwargs is None:    plot_attention_matrix_kwargs    = dict()
        if plot_attention_lines_kwargs is None:     plot_attention_lines_kwargs     = dict()

        #determine grid-layout
        nsubplots = int(np.ceil(np.sqrt(attention_weights.shape[0])))

        #plotting
        for idx, aw_h in enumerate(attention_weights):
            
            ax = fig.add_subplot(nsubplots,nsubplots,idx+1)
            ax.set_title(f'Attention Head {idx+1}')
            
            #decide about style
            if style == 'matrix':
                self.plot_attention_matrix(aw_h, ax=ax, cmap=cmap, cmap_norm=cmap_norm, **plot_attention_matrix_kwargs)
                
                
                #labelling and formatting
                if featurenames is not None:
                    ax.set_xticks(np.arange(len(aw_h))+0.5, labels=featurenames)                
                    ax.set_yticks(np.arange(len(aw_h))+0.5, labels=featurenames)                
                ax.set_xlabel('Input Features')
                ax.set_ylabel('Output Features')                    
                ax.set_aspect('equal')
            elif style == 'lines':
                self.plot_attention_lines(aw_h, ax=ax, cmap=cmap, cmap_norm=cmap_norm, **plot_attention_lines_kwargs)
                
                #labelling and formatting
                if featurenames is not None:
                    ax.set_yticks(range(len(aw_h)), labels=featurenames)                
                ax.tick_params(labelleft=True, labelright=True) #label both sides          
                ax.set_xticks([0,1], labels=['Input', 'Output'])

        #add colorbar
        cax = fig.add_axes([1,0.08,0.03,0.87])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=cmap_norm), cax=cax)
        cbar.set_label('Attention Weights')


        axs = fig.axes

        return fig, axs

class ParallelCoordinates:
    """
        - class to create a Prallel-Coordinate plot
        - inspired by the Weights&Biases Parallel-Coordinates plot 
            - https://docs.wandb.ai/guides/app/features/panels/parallel-coordinates (last access: 15.05.2023)

        Attributes
        ----------
            - `nancolor`
                - `str`, `tuple`, optional
                - color to draw failed runs (evaluate to nan) in
                - if a tuple is passed it has to be a RGBA-tuple
                - the default is `'C2'`
            - `nanfrac`
                - `float`, optional
                - the fraction of the colormap to use for nan-values (i.e. failed runs)
                    - fraction of `256` (resolution of the colormap)
                - will also influence the number of bins/binsize used in the histogram of the last coordinate
                - a value between `0` and `1`
                - the default is `4/256`
            - `base_cmap`
                - `str`, `mcolors.Colormap`, optional
                - colormap to map the scores onto
                - some space will be allocated for `nan`, if nans shall be displayed as well
                - the default is `'plasma'`
            - `vmin`
                - `float`, optional
                - minimum value for the colormapping
                - for evenly spaced colors choose `0`
                - the default is `0`
            - `vmax`
                - `float`, optional
                - maximum value for the colormapping
                - for evenly spaced colors choose `1`
                - the default is `1`
            - `y_margin`
                - `float`, optional
                - how much space to add above and below the maxima of the coordinate axes
                    - i.e., padding of the coordinate axis
                - the default is `0.05`
                    - `5%` of the axis range
            - `xscale_dist`
                - Literal, `Callable`, optional
                - scaling to apply to the x-axis of the histogram/distribution of the last coordinate
                - allowed `Literals`
                    - `'symlog'`
                        - will use `self.symlog()` to calculate the scaling
                        - imitates `matplotlib`s symlog axis scaling
                    - `'linear'`
                        - applies linear scaling
                - if `Callable`
                    - has to take one argument (`x`, array to be scaled)
                    - has to return one parameter (`x_scaled`, scaled version of `x`)
                - the default is `None`
                    - will use `'linear'`
            - `sleep`
                - `float`, optional
                - time to sleep after finishing each job in plotting runs/models and coordinate-axes
                - the default is `0.0` (seconds)
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `make_new_cmap()`
            - `rescale2range()`
            - `symlog()`
            - `__deal_with_categorical()`
            - `__deal_with_inf()`
            - `__deal_with_nan()`
            - `__set_xscale_dist()`
            - `create_axes()`
            - `plot_line()`
            - `plot_score_distribution()`
            - `plot()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `re`
            - `time`
            - `typing`

        Comments
        --------

    """

    def __init__(self,
        nancolor:Union[str,tuple]='tab:grey', nanfrac:float=4/256,
        base_cmap:Union[str,mcolors.Colormap]='plasma', vmin:float=0, vmax:float=0,
        y_margin:float=0.05,
        xscale_dist:Union[Literal["symlog","linear"],Callable]=None,
        sleep:float=0,
        verbose:int=0,
        ) -> None:
        
        
        self.nancolor                                   = nancolor
        self.nanfrac                                    = nanfrac
        self.base_cmap                                  = base_cmap
        self.vmin                                       = vmin
        self.vmax                                       = vmax
        self.y_margin                                   = y_margin
        if xscale_dist is None:     self.xscale_dist    = 'linear'
        else: self.xscale_dist                          = xscale_dist
        self.sleep                                      = sleep
        self.verbose                                    = verbose
        
        return

    def __repr__(self) -> str:

        return (
            f'{self.__class__.__name__}(\n'
            f'    nancolor={repr(self.nancolor)}, nanfrac={repr(self.nanfrac)},\n'
            f'    base_cmap={repr(self.base_cmap)}, vmin={repr(self.vmin)}, vmin={repr(self.vmax)},\n'
            f'    sleep={repr(self.sleep)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def make_new_cmap(self,
        cmap:mcolors.Colormap,
        nancolor:Union[str,tuple]='tab:grey',
        nanfrac:float=4/256,
        ) -> mcolors.Colormap:
        """
            - method to generate a colormap allocating some space for `nan`-values
            - `nan`-values are represented by the lowest values

            Parameters
            ----------
                - `cmap`
                    - `mcolors.Colormap`
                    - template-colormap to use for the creation
                - `nancolor`
                    - `str`, `tuple`, optional
                    - color to draw values representing `nan` in
                    - if a tuple is passed it has to be a RGBA-tuple
                    - the default is `'tab:grey'`
                - `nanfrac`
                    - `float`, optional
                    - the fraction of the colormap to use for `nan`-values (i.e., failed runs)
                        - fraction of `256` (resolution of the colormap)
                    - will also influence the number of bins/binsize used in the histogram of the last coordinate
                    - a value between `0` and `1`
                    - the default is `4/256`

            Raises
            ------

            Returns
            -------
                - `cmap`
                    - `mcolors.Colormap`
                    - modified input colormap

            Comments
            --------
        """
        
        newcolors = cmap(np.linspace(0, 1, 256))
        c_nan = mcolors.to_rgba(nancolor)
        newcolors[:int(256*nanfrac)] = c_nan                       #tiny fraction of the colormap gets attributed to nan values
        cmap = mcolors.ListedColormap(newcolors)

        return cmap

    def rescale2range(self,
        X:np.ndarray,
        min_in:Union[float,np.ndarray]=None, max_in:Union[float,np.ndarray]=None,
        min_out:float=0, max_out:float=1,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to map an input array to a custom parameter range

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - array to be rescaled
                - `min_in`
                    - float, `np.ndarray`, optional
                    - reference value for the minimum of the input-range
                    - the default is `None`
                        - will be set to `np.nanmin(X)`
                - `max_in`
                    - `float`, `np.ndarray`, optional
                    - reference value for the maximum of the input-range
                    - the default is `None`
                        - will be set to `np.nanmax(X)`
                - `min_out`
                    - `float`, optional
                    - minimum value for the wished output range
                    - the default is `0`
                - `max_out`
                    - `float`, optional
                    - maximum value for the wished output range
                    - the default is `1`

            Raises
            ------

            Returns
            -------
                - `X_scaled`
                    - `np.ndarray`
                    - same shape as `X`
                    - scaled version of `X`

            Comments
            --------
        """
  
        #default parameters
        if min_in is None:  min_in  = np.nanmin(X)
        if max_in is None:  max_in  = np.nanmax(X)
        if verbose is None: verbose = self.verbose

        #rescale
        X_scaled = (X - min_in)/(max_in-min_in) * (max_out-min_out) + min_out


        return X_scaled

    def symlog(self,
        x:np.ndarray
        ) -> np.ndarray:
        """
            - method implementing a symmetric logarithm similar to `matplotlib` y-scale `'symlog'`

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - array to be mapped to the symmetric log function

            Raises
            ------

            Returns
            -------
                - `out`
                    - `np.ndarray`
                    - `x` after application of the symmetric log function

            Comments
            --------

        """

        out = np.sign(x) * np.log1p(np.abs(x))

        return out
    
 
    def __deal_with_categorical(self,
        X:np.ndarray,
        verbose:int=None,
        ) -> Tuple[np.ndarray, dict, list]:
        """
            - method to convert categorical columns (columns in `X` that are not convertible to `np.float64`) to a float representation
                - will assign one unique integer to every unique element in the corresponding columns

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - 2d array containing the data
                    - has shape `(nsamples,nfeatures)`
                    - every column will be checked and converted if necessary
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
                - `X_num`
                    - `np.ndarray`
                    - `X` but categorical columns are replaced by float representations
                - `mappings`
                    - `dict`
                    - mappings used to convert original categorical columns to numerical equivalents
                    - keys
                        - original classes
                    - values
                        - corresponding integers
                - `iscatbools`
                    - `list`
                    - has length like `X.shape[1]`
                    - contains booleans specifying whether a specific feature/column is categorical
            
            Columns
            -------
        """

        #default parameters
        if verbose is None: verbose = self.verbose

        X_num = X.T.copy()  #init numerical version of X
        mappings = []
        iscatbools = []
        for idx, xi in enumerate(X_num):
            #continuous
            try:
                xi.astype(np.float64)
                iscatbools.append(False)
                mappings.append(dict())
            #categorical
            except:
                #denote that columns is categorical
                iscatbools.append(True)

                #get uniques and int-encoding
                uniques, categorical = np.unique(xi, return_inverse=True)
                
                #convert to numerical
                categorical = categorical.astype(np.float64)
                
                #int equivalent of 'nan'
                nanidx = np.where(uniques=='nan')[0][0]
                
                #assign actual np.nan to 'nan' (to be colored correctly)
                categorical[(categorical==nanidx)] = np.nan
                
                #store mapping (for axis lables)
                mapping = {u:idx for idx, u in enumerate(uniques)}
                mapping.pop('nan')  #remove duplicate label for 'nan'
                mappings.append(mapping)

                #assign numerical version of categorical column to `X`
                X_num[idx] = categorical
        
        #make sure shapes are correct again
        X_num = X_num.T.astype(np.float64)

        return X_num, mappings, iscatbools

    def __deal_with_inf(self,
        X:np.ndarray,
        infmargin:float=0.05,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method that modifies `np.inf` and `-np.inf` to don't give issues when creating the plot
            - the current implementation replaces them with `np.nan`

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - data array in which to deal with `inf` and `-inf` values accordingly
                    - has shape `(nsamples,nfeatures)`
                - `infmargin`
                    - NOTE: not implemented yet
                    - `float`, optional
                    - margin to place between valid values and `+/- np.inf`
                    - the default is `0.05`
                        - 5% of data-range above maximum value for `+np.inf`
                        - 5% of data-range below minimum value for `-np.inf`
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
                - `X_inffilled`
                    - `np.ndarray`
                    - `X` where `+/-np.inf` have been dealt with accordingly

            Comment
            -------
                
        """

        #default parameters
        if verbose is None: verbose = self.verbose
        
        #TODO: deal with +/- inf separately (add extra ticks all the way at the top and bottom)
        X_inffilled = X.copy()
        X_inffilled[np.isinf(X)] = np.nan #fill inf with nan

        return X_inffilled

    def __deal_with_nan(self,
        X:np.ndarray,
        nanmargin:float=0.1,
        verbose:int=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method that deals with missing values (`np.nan`) to don't give issues when creating the plot
            - also makes sure that `np.nan` values are visualized accordingly to debug models and make inferences about the data

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - data array in which to deal with `np.inf` values accordingly
                    - has shape `(nsamples,nfeatures)`
                - `nanmargin`
                    - `float`, optional
                    - margin to place between valid values and `np.nan`-representation
                    - the default is `0.1`
                        - 10% of data-range below minimum value
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
                - `X_nanfilled`
                    - `np.ndarray`
                    - `X` where `np.nan` have been dealt with accordingly
                - `nanmask`
                    - `np.ndarray`
                    - boolean array representing entries where `X` contains `np.nan`
                    - used in plotting to ensure that lines containing any `np.nan` are using the correct colormapping

            Comment
            -------
                
        """
        #default parameters
        if verbose is None: verbose = self.verbose
        

        nanmask = np.isnan(X)            #check where nan were filled
        idxs = np.where(np.isnan(X))

        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)

        fill_values = mins - (maxs - mins)*nanmargin  #fill with `min - nanmargin`% of feature range (i.e., plot nan below everything else)

        X_nafilled = X.copy()
        X_nafilled[idxs] = np.take(fill_values, idxs[1])


        return X_nafilled, nanmask

    def __set_xscale_dist(self,
        x:np.ndarray,
        xscale_dist:Union[Literal['symlog', 'linear'],Callable]=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to apply scaling to the x-axis of the score-distribution

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - array to be scaled i.e.,
                        - actual data
                        - axis-ticks
                - `xscale_dist`
                    - `Literal["symlog","linear"]`, `Callable`, optional
                    - scaling to apply to the x-axis of the histogram/distribution of the last coordinate
                    - allowed Literals
                        - `'symlog'`
                            - will use `self.symlog()` to calculate the scaling
                            - imitates `matplotlib`s symlog axis scaling
                        - `'linear'`
                            - applies linear scaling
                    - if `Callable`
                        - has to take one argument (`x`, array to be scaled)
                        - has to return one parameter (`x_scaled`, scaled version of `x`)
                    - the default is `None`
                        - will use `'linear'`
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
                - `x_scaled`
                    - `np.ndarray`
                    - scaled version of `x`
            
            Comments
            --------
        
        """


        #default parameters
        if verbose is None: verbose = self.verbose
        
        #select correct scaling
        if xscale_dist == 'linear':
            x_scaled = x
        elif xscale_dist == 'symlog':
            x_scaled = self.symlog(x)
        elif isinstance(xscale_dist, Callable):
            x_scaled = xscale_dist(x)
        else:
            almf.printf(
                msg=(
                    f'Using `"linear"` since `"{xscale_dist}"` is not a valid argument. '
                    f'Allowed are `"symlog"`, `"linear"`, or a custom callabel taking one input and returning one output.'
                ),
                context=self.__set_xscale_dist.__name__,
                type='WARNING', level=0,
                verbose=verbose
            )

        return x_scaled
    
    def create_axes(self,
        ax:plt.Axes,
        coordnames:np.ndarray,
        mins:np.ndarray, maxs:np.ndarray,
        iscatbools:np.ndarray,
        mappings:List[dict],
        nanmask:np.ndarray, nanmins:np.ndarray,
        verbose:int=None,
        set_xticklabels_kwargs:dict=None,
        ):
        """
            - method to initialize the axes needed for the parallel-coordinate plot
            
            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - host axis to plot into
                - `coordnames`
                    - `np.ndarray`
                    - names of the individual coordinates
                    - will create one axis per entry of `coordnames`
                - `mins`
                    - `np.ndarray`
                    - same length as `coordnames`
                    - minimum y-limit for the different axis
                - `maxs`
                    - `np.ndarray`
                    - same length as `coordnames`
                    - maximum y-limits for the different axis
                - `iscatbools`
                    - `np.ndarray`
                    - contains booleans
                    - same length as `coordnames`
                    - specifies whether an individual axis contains a categorical feature or not
                    - will use `mappings` for tick-labelling instead of automatic `matplotlib` ticks
                - `nanmask`
                    - `np.ndarray`
                    - has dtype `bool`
                    - has shape `(nsamples,nfeature)`
                    - used to determine if an extra tick for missing values (`np.nan`) has to be allocated
                - `nanmins`
                    - `np.ndarray`
                    - same length as `coordnames`
                    - contains y-positions of ticks for missing values (`np.nan`)
                    - tick will get added if `nanmask` evaluates to true for any of the elements in the respective column
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `set_xticklabels_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.set_xticklabels()`
                        - i.e. names of the individual coordinates/axes
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `axs`
                    - `np.ndarray`
                    - contains
                        - host axis (`ax`) as zeroth element
                        - all created coordinate axis as following elements
                        - axis created for the score-distribution as last element

            Comments
            --------
        """

        if verbose is None:                 verbose                 = self.verbose
        if set_xticklabels_kwargs is None:  set_xticklabels_kwargs  = dict()


        n_subaxis = len(coordnames)

        ##add coordinate axes (make separate list to work work with any subplot)
        axs = [ax] + [ax.twinx() for idx in range(n_subaxis-1)]

        ##format all axis
        for idx, axi in enumerate(axs):
            axi.set_ylim(mins[idx], maxs[idx])
            axi.spines['top'].set_visible(False)
            axi.spines['bottom'].set_visible(False)
            if idx > 0:
                axi.spines['left'].set_visible(False)
                axi.yaxis.set_ticks_position('right')
                axi.spines['right'].set_position(('axes', idx / (n_subaxis)))
                axi.patch.set_alpha(0.5)
            if iscatbools[idx]:
                axi.set_yticks(
                    ticks=list(mappings[idx].values()),
                    labels=list(mappings[idx].keys()),
                )
            if np.any(nanmask, axis=0)[idx]:
                ylim = axi.get_ylim()   #store ylim to reset after new tick-assignment
                #add tick for nan
                axi.set_yticks(
                    ticks= list(axi.get_yticks())+[nanmins[idx]],
                    labels=list(axi.get_yticklabels())+['nan'],
                )
                #reassign ylim
                axi.set_ylim(ylim)

        axd = ax.twiny()
        axd.set_xlim(-(n_subaxis-1), None)
        axd.xaxis.set_ticks_position('bottom')
        axd.spines['top'].set_visible(False)
        axd.spines['bottom'].set_visible(False)
        axd.spines['right'].set_visible(False)
        axd.spines['left'].set_visible(False)
        axd.xaxis.set_label_position('bottom')
        axd.set_xlabel('Counts', loc='right')
        # axd.yaxis.set_label_position('right')
        # axd.set_ylabel('Score Distribution', loc='center')

        ##correct labelling
        ax.set_xlim(0, n_subaxis)
        ax.set_xticks(range(n_subaxis))
        ax.set_xticklabels(coordnames, **set_xticklabels_kwargs)
        ax.tick_params(axis='x', which='major', pad=7)
        ax.spines['right'].set_visible(False)
        ax.xaxis.tick_top()

        #make sure labels of last coordinate are not covered by distribution (axd)
        axs[-1].set_zorder(1)

        #append axd to axs
        axs += [axd]

        return axs

    def plot_line(self,
        line:np.ndarray,
        nabool:bool,
        ax:plt.Axes,
        linecolor:Tuple[str,tuple]='tab:blue', nancolor:Union[str,tuple]='tab:grey',
        sleep:float=0,
        verbose:int=None,
        pathpatch_kwargs:dict=None,
        ) -> None:
        """
            - method to add aline into the plot representing an individual run/model

            Parameters
            ----------
                - `line`
                    - `np.ndarray`
                    - has shape `(nfeatures)`
                    - line to add to the plot
                    - will be displayed as cubic bezier curve
                - `ax`
                    - `plt.Axes`
                    - axis to plot `line` into
                    - usually the host-axis of the plot
                - `linecolor`
                    - `str`, `tuple`, optional
                    - color to plot the line in
                    - if a tuple is passed, it has to be a RGBA-tuple
                    - the default is `'tab:blue'`
                - `nancolor`
                    - `str`, `tuple`, optional
                    - color to draw line in, if it contains missing values (`np.nan`)
                    - if a tuple is passed it has to be a RGBA-tuple
                    - the default is `'tab:grey'`
                - `sleep`
                    - `float`, optional
                    - time to sleep after finishing plotting `line`
                    - the default is `0.0` (seconds)
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`                
                - `pathpatch_kwargs`
                    - 'dict', optional
                    - kwargs to pass to `matplotlib.patches.PathPatch()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #default values
        if verbose is None:             verbose             = self.verbose
        if pathpatch_kwargs is None:    pathpatch_kwargs    = dict()

        #create cubic bezier for nice display
        verts = list(zip(
            [x for x in np.linspace(0, len(line) - 1, len(line) * 3 - 2, endpoint=True)],
            np.repeat(line, 3)[1:-1]
        ))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        
        #actually plot the line for current model
        ##runs with any missing values
        if nabool:
            patch = mpatches.PathPatch(path, facecolor='none', edgecolor=nancolor, **pathpatch_kwargs)
        ##completely valid runs
        else:
            patch = mpatches.PathPatch(path, facecolor='none', edgecolor=linecolor, **pathpatch_kwargs)
        
        #add line to plot        
        ax.add_patch(patch)


        time.sleep(sleep)

        return

    def plot_score_distribution(self,
        X:np.ndarray,
        ax:plt.Axes,
        nanfrac:float=None,
        xscale_dist:Union[Literal['symlog', 'linear'],Callable]=None,
        cmap:Union[mcolors.Colormap,str]='plasma', vmin:float=None, vmax:float=None,
        verbose:int=None,
        set_xticklabels_dist_kwargs:dict=None,
        ) -> None:
        """
            - method to add a distribution of the final coordinate (usually scores) to `ax`
            - the distribution will be colorcoded to match the values

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - has shape `(nsamples,nfeatures)`
                    - dataset to plot
                    - contains coordinate to plot as distribution as last row
                - `ax`
                    - `plt.Axes`
                    - axis to plot into
                    - usually the host axis
                - `nanfrac`
                    - `float`, optional
                    - the fraction of the colormap to use for missing values (`np.nan`)
                        - fraction of `256` (resolution of the colormap)
                    - will also influence the number of bins/binsize used in the histogram
                    - a value between `0` and `1`
                    - overrides `self.nanfrac`
                    - the default is `None`
                        - will fall back to `self.nanfrac`
                - `xscale_dist`
                    - `Literal["symlog","linear"]`, `Callable`, optional
                    - scaling to apply to the x-axis of the histogram/distribution of the last coordinate
                    - allowed Literals
                        - `'symlog'`
                            - will use `self.symlog()` to calculate the scaling
                            - imitates `matplotlib`s symlog axis scaling
                        - `'linear'`
                            - applies linear scaling
                    - if `Callable`
                        - has to take one argument (`x`, array to be scaled)
                        - has to return one parameter (`x_scaled`, scaled version of `x`)
                    - overrides `self.xscale_dist`
                    - the default is `None`
                        - will fall back to `self.xscale_dist`
                - `cmap`
                    - `mcolor.Colormap`, `str`, optional
                    - colormap to apply to the plot for encoding the score
                    - the default is `'plasma'`
                - `vmin`
                    - `float`, optional
                    - minimum value for the colormapping
                    - for evenly spaced colors choose `0`
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - `float`, optional
                    - maximum value for the colormapping
                    - for evenly spaced colors choose `1`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `set_xticklabels_dist_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.set_xticklabels()`
                        - i.e. ticklabels for the distribution counts
                    - the default is `None`
                        - will be set to `dict(rotation=45)`
                
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        #default parameter
        if nanfrac is None:                     nanfrac                     = self.nanfrac
        if vmin is None:                        vmin                        = self.vmin
        if vmax is None:                        vmax                        = self.vmax
        if verbose is None:                     verbose                     = self.verbose
        if xscale_dist is None:                 xscale_dist                 = self.xscale_dist
        if set_xticklabels_dist_kwargs is None: set_xticklabels_dist_kwargs = dict(rotation=45)

        if 'rotation' not in set_xticklabels_dist_kwargs.keys(): set_xticklabels_dist_kwargs['rotation'] = 45
        
        
        #adjust bins to colorbar
        bins = np.linspace(X[:,-1].min(), X[:,-1].max(), int(1//nanfrac))
        bins01 = self.rescale2range(bins, bins.min(), bins.max(), vmin, vmax, verbose=verbose)

        #get colors for bins
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        colors = cmap(bins01)
        
        #get histogram
        hist, bin_edges = np.histogram(X[:,-1], bins)
        
        #rescaling to specification
        hist = self.__set_xscale_dist(hist, xscale_dist=xscale_dist)

        #rescale to range(0,1) (to fit in axis)
        hist = hist/hist.max()

        #adapt xticks to match scaled display of data
        xticks = np.linspace(0,1,3)
        xticks = self.__set_xscale_dist(xticks, xscale_dist=xscale_dist)
        ax.set_xticks(xticks)
        ax.set_xticklabels(ax.get_xticks(), **set_xticklabels_dist_kwargs)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        #plot and colormap histogram
        binheight = nanfrac*(X.max()-X.min())    #rescale binheight to match axis limits
        ax.barh(bin_edges[:-1], hist, height=binheight, left=None, color=colors)
        
        return
 
    def plot(self,
        X:np.ndarray,
        coordnames:list=None,
        nancolor=None, nanfrac=None,
        base_cmap=None, vmin:float=None, vmax:float=None,
        y_margin:float=None,
        xscale_dist:Literal['symlog', 'linear']=None,
        ax:plt.Axes=None,
        sleep:float=None,
        verbose:int=None,
        set_xticklabels_kwargs:dict=None,
        pathpatch_kwargs:dict=None,
        set_xticklabels_dist_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - array contianin samples to show in the plot
                    - has shape `(nsamples,nfeatures)`
                    - last feature/coordinate will also be displayed as distribution
                - `coordnames`
                    - `np.ndarray`, optional
                    - names of the individual coordinates (columns in `X`)
                    - if not enough are passed, will set the missing names to default values
                    - the default is `None`
                        - will autogenerate names
                - `nancolor`
                    - `str`, `tuple`, optional
                    - color to draw sample/line in, if it contains missing values (`np.nan`)
                    - if a tuple is passed it has to be a RGBA-tuple
                    - the default is `'tab:grey'`
                - `nanfrac`
                    - `float`, optional
                    - the fraction of the colormap to use for missing values (`np.nan`)
                        - fraction of `256` (resolution of the colormap)
                    - will also influence the number of bins/binsize used in the histogram
                    - a value between `0` and `1`
                    - overrides `self.nanfrac`
                    - the default is `None`
                        - will fall back to `self.nanfrac`
                - `base_cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to map the last coordinate/feature in `X` onto
                    - some space will be allocated for `nan`, if nans shall be displayed as well
                    - overrides `self.base_cmap`
                    - the default is `None`
                        - will fall back to `self.base_cmap`
                - `vmin`
                    - `float`, optional
                    - minimum value for the colormapping
                    - for evenly spaced colors choose `0`
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - `float`, optional
                    - maximum value for the colormapping
                    - for evenly spaced colors choose `1`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `y_margin`
                    - `float`, optional
                    - how much space to add above and below the maxima of the coordinate axes
                        - i.e., padding of the coordinate axis
                    - overrides `self.y_margin`
                    - the default is `None`
                        - will fall back to `self.y_margin`
                - `xscale_dist`
                    - `Literal["symlog","linear"]`, `Callable`, optional
                    - scaling to apply to the x-axis of the histogram/distribution of the last coordinate
                    - allowed `Literals`
                        - `'symlog'`
                            - will use `self.symlog()` to calculate the scaling
                            - imitates `matplotlib`s symlog axis scaling
                        - `'linear'`
                            - applies linear scaling
                    - if `Callable`
                        - has to take one argument (`x`, array to be scaled)
                        - has to return one parameter (`x_scaled`, scaled version of `x`)
                    - overrides `self.xscale_dist`
                    - the default is `None`
                        - will fall back to `self.xscale_dist`
                - `ax`
                    - `plt.Axes`, optional
                    - axis to plot into
                    - the default is `None`
                        - will generate new figure and axis if not provided
                - `sleep`
                    - `float`, optional
                    - time to sleep after finishing plotting one line/sample
                    - overrides `self.sleep`
                    - the default is `None`
                        - will fall back to `self.sleep`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `set_xticklabels_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.set_xticklabels()` in `self.create_axes()`
                        - i.e. names of the individual coordinates/axes
                    - the default is `None`
                        - will be set to `dict()`                
                - `pathpatch_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `matplotlib.patches.PathPatch()` in `self.plot_line()`
                    - the default is `None`
                        - will be set to `dict()`                        
                - `set_xticklabels_dist_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.set_xticklabels()` in `self.plot_score_distribution()`
                        - i.e. ticklabels for the distribution counts
                    - the default is `None`
                        - will be set to `dict(rotation=45)`
                        
            Raises
            ------

            Returns
            ------
                - `fig`
                    - `Figure`
                    - created figure if no `ax` was passed
                    - original figure corresponding to host axis (`ax`) if `ax` was passed
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
            
            Comments
            --------
        
        """
        

        #default parameters
        if coordnames is None:                  coordnames                  = [f'Feature {i}' for i in range(X.shape[1])]
        if nancolor is None:                    nancolor                    = self.nancolor
        if nanfrac is None:                     nanfrac                     = self.nanfrac
        if base_cmap is None:                   base_cmap                   = self.base_cmap
        if vmin is None:                        vmin                        = self.vmin
        if vmax is None:                        vmax                        = self.vmax
        if y_margin is None:                    y_margin                    = self.y_margin
        if xscale_dist is None:                 xscale_dist                 = self.xscale_dist
        if sleep is None:                       sleep                       = self.sleep
        if verbose is None:                     verbose                     = self.verbose
        if set_xticklabels_kwargs is None:      set_xticklabels_kwargs      = dict(rotation=45)
        if pathpatch_kwargs is None:            pathpatch_kwargs            = dict()
        if set_xticklabels_dist_kwargs is None: set_xticklabels_dist_kwargs = dict()

        if 'rotation' not in set_xticklabels_dist_kwargs.keys(): set_xticklabels_dist_kwargs['rotation'] = 45


        #ensure correct shapes
        if len(coordnames) < X.shape[1]:
            almf.printf(
                msg=(
                    f'`len(coordnames)` has to be the same as `X.shape[1]`. '
                    f'Adding autogenerated labels to ensure correct length!'
                ),
                context=self.plot.__name__,
                type='WARNING',
                level=0,
                verbose=verbose
            )
            coordnames += [f'Feature {i+1}' for i in range(len(coordnames), X.shape[1])]

        #get colormap
        if isinstance(base_cmap, str): cmap = plt.get_cmap(base_cmap)
        else: cmap = base_cmap
        cmap = self.make_new_cmap(cmap=cmap, nancolor=nancolor, nanfrac=nanfrac)    #modified cmap

        #prepare data
        ##deal with categorical (str) columns
        X_plot, mappings, iscatbools = self.__deal_with_categorical(X, verbose=verbose)   
        ##deal with inf
        X_plot = self.__deal_with_inf(X_plot, infmargin=0.05, verbose=verbose)
        ##deal with nan
        X_plot, nanmask = self.__deal_with_nan(X_plot, nanmargin=0.1, verbose=verbose)

        #get ranges (for rescaling and limit definition)
        mins    = np.nanmin(X_plot, axis=0)
        maxs    = np.nanmax(X_plot, axis=0)
        nanmins = mins.copy()               #store minima for plotting nan
        mins    -= (maxs - mins)*y_margin
        maxs    += (maxs - mins)*y_margin
     
        ##scale to make features compatible
        X_plot = self.rescale2range(X_plot, mins, maxs, mins[0], maxs[0], verbose=verbose)


        #create axis, labels etc.
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        axs = self.create_axes(
            ax=ax,
            coordnames=coordnames,
            mins=mins, maxs=maxs,
            iscatbools=iscatbools,
            mappings=mappings,
            nanmask=nanmask,
            nanmins=nanmins,
            verbose=verbose,
            set_xticklabels_kwargs=set_xticklabels_kwargs,
        )

        ax  = axs[0]
        axd = axs[-1]

        #actual plotting
        ##plot lines
        norm_scores = self.rescale2range(X_plot[:,-1], X_plot[:,-1].min(), X_plot[:,-1].max(), vmin, vmax, verbose=verbose)   #normalize scores
        linecolors = cmap(norm_scores)  #get line colors from normalized scores
        for idx, line in enumerate(X_plot):
            self.plot_line(
                line,
                nabool=np.any(nanmask, axis=1)[idx],
                ax=ax,
                linecolor=linecolors[idx],
                nancolor=nancolor,
                verbose=verbose,
                pathpatch_kwargs=pathpatch_kwargs,
            )

        ##plot score distribution
        self.plot_score_distribution(
            X_plot,
            ax=axd,
            nanfrac=nanfrac,
            xscale_dist=xscale_dist,
            cmap=cmap, vmin=vmin, vmax=vmax,
            verbose=verbose,
            set_xticklabels_dist_kwargs=set_xticklabels_dist_kwargs,
        )

        axs = fig.axes

        return fig, axs

class VennDiagram:
    """
        - class to create a Venn diagram

        Attributes
        ----------
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `get_positions()`
            - `parse_query()`
            - `plot()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `re`
            - `typing`

        Comments
        --------

    """

    def __init__(self,
        verbose:int=0
        ) -> None:
        
        self.verbose = verbose

        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def get_positions(self,
        n:int,
        x0:np.ndarray=None,
        r:float=1,
        ) -> np.ndarray:
        """
            - method to obtain positions for the individual query keywords

            Parameters
            ----------
                - `n`
                    - `int`
                    - number of positions to generate
                    - will generate `n` equidistant points on a circle with
                        - origin `x0`
                        - radius `r`
                - `x0`
                    - `np.ndarray`, optional
                    - origin of the diagram
                    - the default is `None`
                    - will be set to `np.array([0,0])`
                - `r`
                    - `float`, optinal
                    - radius of the circle to position the queries on
                    - the default is `1`

            Raises
            ------

            Returns
            -------
                - `pos`
                    - `np.ndarray`
                    - generated positions in carthesian coordinates

            Comments
            --------

        """

        if x0 is None: x0 = np.array([0,0])

        if n > 1:
            phi = np.linspace(0,2*np.pi, n, endpoint=False)

            #array of base-circles (base_circles.shape[0] = x0.shape[0])
            base_circle = r*np.array([np.cos(phi),np.sin(phi)])

            pos = (x0.reshape(-1,1) + base_circle).T
        else:
            pos = x0.reshape(-1,1).T

        return pos
    
    def parse_query(self,
        query:str,
        array_name:str=None,
        idx_offset:int=3,
        axis:int=2,
        ) -> Tuple[str,int]:
        """
            - method to parse the `query` and substitue relevant parts for evaluation

            Parameters
            ----------
                - `query`
                    - `str`
                    - query to be parsed
                    - has to follow the following syntax
                        - any keyword is represented by `'@\d+'`
                            - where `'\d+'` can be substituted with any integer
                            - i.e., `'@1'`
                        - logical or is represented by `'|'`
                        - logical and is represented by `'&'`
                        - negation (not) is represented by `'<@\d+>'`
                            - where `'\d+'` can be substituted with any integer
                            - i.e., `'<@1>'` means `not '@1'`
                        - make sure to set parenthesis correctly!
                    - the query can then be evaluated by means of mathematical operations due to the substitutions made
                        - `'|'` --> `'+'`
                        - `'&'` --> `'*'`
                        - `'<@\d+>'` --> `'(1-@\d+)'`
                - `array_name`
                    - `str`, optional
                    - name of the array the query will be applied to
                    - the default is `None`
                        - will be set to `'query_array'`
                            - required for `self.plot()`
                - `idx_offset`
                    - `int`, optional
                    - offset of the actual index where the boolean mask is found
                    - the default is `3`
                            - required for `self.plot()`
                - `axis`
                    - `int`, optional
                    - axis that contains the individual masks
                    - the default is `2`
                            - required for `self.plot()`

            Raises
            ------

            Returns
            -------
                - `query`
                    - `str`
                    - parsed query with replacements
                - `n`
                    - `int`
                    - number of unique keywords in the query

            Comments
            --------
        """

        #default parameters
        if array_name is None:
            array_name = 'query_array'

        #get 

        #obtain number of unique keywords
        kwrds = np.unique(re.findall(r'@\d+', query))
        n = len(kwrds)

        #make substitutions
        query = re.sub(r'\|', '+', query)
        query = re.sub(r'\&', '*', query)
        query = re.sub(r'\~', '1-', query)

        #determine which axis to fill with ':,' and ',:'
        axisfill = ':,'*axis
        for idx, k in enumerate(kwrds):
            query = re.sub(k, f'{array_name}[{axisfill}{idx_offset+idx}]', query)

        return query, n

    def plot(self,
        query:str=None,
        n:int=None, x0:np.ndarray=None, r:float=1,
        res:int=250,
        labels:List[str]=None,
        fig:Figure=None,
        ax:plt.Axes=None,
        circle_cmap:Union[str,mcolors.Colormap]=None,
        show_cbar:bool=True,
        pcolormesh_kwargs:dict=None,
        circle_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to create the Venn diagram

            Parameters
            ----------
                - `query`
                    - `str`, optional
                    - query to be displayed in the diagram
                    - will be passed to `self.parse_query()`
                    - has to follow the following syntax
                        - any keyword is represented by `'@\d+'`
                            - where `'\d+'` can be substituted with any integer
                            - i.e., `'@1'`
                        - logical or is represented by `'|'`
                        - logical and is represented by `'&'`
                        - negation (not) is represented by `'<@\d+>'`
                            - where `'\d+'` can be substituted with any integer
                            - i.e., `'<@1>'` means `not '@1'`
                        - make sure to set parenthesis correctly!
                    - the default is `None`
                        - will only display the outlines of the circles in the diagram
                - `n`
                    - `int`, optional
                    - how many circles to show in the diagram
                    - if `None` or smaller than the number of unique keywords in `query`
                        - will use the number of unique keywords in `query` to determine the number of circles
                    - the default is `None`
                        - infer from `query`
                - `x0`
                    - `np.ndarray`, optional
                    - origin of the diagram
                    - 1d array containing
                        - x coordinate of the origin
                        - y coordinate of the origin
                    - the default is `None`
                        - will be set to `np.array([0,0])`
                - `r`
                    - `float`, optional
                    - radius of the diagram
                        - i.e. how far the centers of the individual circles are away from `x0`
                    - the default is 1
                - `res`
                    - `int`, optional
                    - resolution of the colormap in the background
                    - the default is `250`
                - `labels`
                    - `list`, optional
                    - list of labels to assign to the individual queries
                    - will be shown in the legend
                    - if the less labels than unique keywords (circles) have been passed
                        - will generate artificial labels of the form `'[\d+]'`
                    - the default is `None`
                        - will index all keywords starting from 1
                - `fig`
                    - `Figure`
                    - figure to plot the diagram into
                    - the default is `None`
                        - will generate a new figure
                - `ax`
                    - `plt.Axes`
                    - axis to plot the diagram into
                    - the default is `None`
                        - will create a new axis via `fig.add_subplot(111)`
                - `circle_cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to use for plotting the circle outlines
                    - the default is `None`
                        - will be set to `'tab10'`
                - `show_cbar`
                    - `bool`, optional
                    - whether to plot a colorbar or not
                    - the default is `True`
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict(cmap='binary')`
                - `circle_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `plt.Circle()`
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
                    - axes correspoding to `fig`

            Comments
            --------

        """

        #default parameters
        if query is not None:
            query, n_q = self.parse_query(query, idx_offset=3, array_name=None, axis=2)
        else:
            query = 'query_array[:,:,2]'  #no query
            n_q = 1
        if n is None or n < n_q:
            n = n_q
        if circle_cmap is None:                         circle_cmap                 = 'tab10'
        if pcolormesh_kwargs is None:                   pcolormesh_kwargs           = dict(cmap='binary')
        elif 'cmap' not in pcolormesh_kwargs.keys():    pcolormesh_kwargs['cmap']   = 'binary'
        if circle_kwargs is None:                       circle_kwargs               = dict()
        if labels is None:                              labels = range(1,n+1)
        else:                                           labels = np.append(labels, [f'[{i}]' for i in range(1,(n+1)-len(labels))])

        almf.printf(
            msg=f'Parsed `query`: {query}',
            type=f'INFO',
            context=f'{self.__class__.__name__}.{self.plot.__name__}',
            verbose=self.verbose
        )

        #radius of circles
        r_circ = r*np.sqrt(2)   #a little larger than `r` such that they overlap in the center

        #get positions relative to x0
        pos = self.get_positions(n=n, x0=x0, r=r)

        #x and y values for pcolormesh (venn-colormap)
        xy = np.linspace(-(r+r_circ), r+r_circ, res)
        xx, yy = np.meshgrid(xy, xy)
        xx = np.expand_dims(xx, -1)
        yy = np.expand_dims(yy, -1)
        
        #initialize masking for the total query (venn-colormap)
        vennmask = np.zeros_like(xx)
        vennmask[:] = np.nan

        #initialize query array (contains xcoords, ycoords, venn-mask, masks for individual query parts)
        query_array = np.concatenate(
            (xx, yy, vennmask), axis=-1
        )

        #add masks for individual keywords/circles (query parts)
        for idx, p in enumerate(pos):
            b = (((xx-p[0])**2+(yy-p[1])**2) <= r_circ**2) #essentially (x-x0)**2 + (y-y0)**2 <= r**2
            query_array = np.append(query_array, b, axis=-1)
            

        #apply query
        query_array[:,:,2] = eval(query)

        #plot diagram
        #create figure if neither fig nor ax are provided
        if fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        else:
            fig = ax.axes.get_figure()
        
        ##background mask
        mesh = ax.pcolormesh(
            query_array[:,:,0], query_array[:,:,1], query_array[:,:,2],
            **pcolormesh_kwargs,
        )
        
        ##actual circles (outlines)
        colors = alvp.generate_colors(len(pos), cmap=circle_cmap)
        for idx, (p, c, l) in enumerate(zip(pos, colors, labels)):
            circle = plt.Circle(
                p.flatten(), r_circ,
                color=c, fill=False,
                label=l,
                **circle_kwargs,
            )
            ax.add_artist(circle)

        ##add colorbar
        if show_cbar:
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Query Result')
            if 'vmax' in pcolormesh_kwargs.keys():
                cmax = pcolormesh_kwargs['vmax']
            else:
                cmax = query_array[:,:,2].max().astype(int)
            if 'vmin' in pcolormesh_kwargs.keys():
                cmin = pcolormesh_kwargs['vmin']
            else:
                cmin = query_array[:,:,2].min().astype(int)
            cbar.ax.set_yticks(range(cmin, cmax+1))

        #hide labels
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #add legend(
        ax.legend()

        ax.set_aspect('equal')

        axs = fig.axes
        
        return fig, axs
    

#%%functions
def plot_confusion_matrix(
    X:np.ndarray, y:np.ndarray,
    xlab:str="Class", ylab:str="Class",
    cbarlabel:str=None, cmap:Union[str,mcolors.Colormap]="viridis",
    annotate:bool=True, annotationcolor:Union[str,tuple]="w",
    textfontsize:float=None,
    ax:plt.Axes=None,
    fontsize:float=16,
    ) -> Tuple[Figure,plt.Axes]:
    """
        - function to plot a confusion matrix

        Parameters
        ----------
            - `X`
                - `np.ndarray`
                    - 2d array
                    - value assigned to each pair of classes
            - `y`
                - `np.ndarray`
                    - 1d array
                - classes the values got calculated for
            - `xlab`
                - `str`, optional
                - x-axis label
                - the default is `"Class"`
            - `ylab`
                - `str`, optional
                - y-axis label
                - the default is `"Class"`
            - `cbarlab`
                - `str`, optional
                - colorbar label
                - the default is `None`
            - `cmap`
                - `str`, `mcolors.Colormap` optional
                - matplotlib colormap to use
                - the default is `"viridis"`
            - `annotate`
                - bool, optional
                - whether to depict the heatmap values in the individiual cells
                - the default is `True`
            - `annotationcolor`
                - `str`, optional
                - color of the heatmapvalues, when depicted in the cells
                - the default is `"w"`
            - `annotationfontsize`
                - `float`, optional
                - fontsize of the heatmapvalues, when depicted in the cells
                - the default is `None`
                    - will result in the same value as `fontsize`
            - `fontsize`
                - `float`, optional
                - fontsize of labels
                - the default is `16`
        
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

        Dependencies
        ------------
            - `numpy`
            - `matplotlib`

        Comments
        --------

    """

    #default paramaeters
    if textfontsize is None: textfontsize = fontsize


    #generate tick positions
    tick_positions = np.arange(0, X.shape[0], 1)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    #plot
    im = ax.imshow(X, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)

    #annotating
    if annotate:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                text = ax.text(j, i, f"{X[i, j]:.1f}",
                            ha="center", va="center", color=annotationcolor, fontsize=textfontsize)

    #set locators
    ax.xaxis.set_major_locator(plt.MaxNLocator(X.shape[0]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(X.shape[0]))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(y)
    ax.set_yticklabels(y)
    
    #fontsize
    ax.tick_params("both", labelsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    #labelling
    cbar.set_label(cbarlabel, fontsize=fontsize)
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)

    axs = fig.axes

    return fig, axs

def plot_predictioneval(
    y_true:np.ndarray, y_pred:np.ndarray,
    fig_kwargs:dict=None,
    sctr_kwargs:dict=None,
    plot_kwargs:dict=None,
    ) -> Tuple[Figure,plt.Axes]:
    """
        - function to produce a plot of the true and predicted lables vs the model prediction

        Parameters
        ----------
            - `y_true`
                - `np.ndarray`
                - ground-truth labels
                - will be plotted on the x-axis
            - `y_pred`
                - `np.ndarray`
                - labels predicted by the model
                - will be plotted on the y-axis
            - `fig_kwargs`
                - `dict`, optional
                - kwargs to pass to `plt.figure()`
                - the default is `None`
                    - will be initialized with `{}`
            - `sctr_kwargs`
                - `dict`, optional
                - kwargs to pass to `ax.scatter()`
                - the default is `None`
                    - will be initialized with `{'color':'tab:blue', 's':1}`
            - `plot_kwargs`
                - `dict`, optional
                - kwargs to pass to `ax.plot()`
                - the default is `None`
                    - will be initialized with `{'color':'tab:orange'}`
            
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
            
            Dependencies
            ------------
                - `matplotlib`
                - `numpy`
            
            Comments
            --------

    """
    
    #initialize parameters
    if fig_kwargs is None: fig_kwargs = {}
    if sctr_kwargs is None: sctr_kwargs = {'color':'tab:blue', 's':1}
    if plot_kwargs is None: plot_kwargs = {'color':'tab:orange'}


    x_ideal = np.linspace(np.nanmin(y_true),np.nanmax(y_true),3)

    fig = plt.figure(**fig_kwargs)
    ax1 = fig.add_subplot(111)

    ax1.scatter(y_true, y_pred, **sctr_kwargs, label='Samples')
    ax1.plot(x_ideal, x_ideal,  **plot_kwargs, label=r'$y_\mathrm{True}=y_\mathrm{Pred}$')

    ax1.set_xlabel(r'$y_\mathrm{True}$')
    ax1.set_ylabel(r'$y_\mathrm{Pred}$')

    ax1.legend()

    axs = fig.axes

    return fig, axs

