
#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy as np
from typing import Union, Tuple, Callable, List, Literal

from astroLuSt.visualization.plotting import generate_colors


#%%classes
class Pad2Size:
    """
        - class to pad a list of arrays with different lengths to a list of arrays with the same length

        Attributes
        ----------
            - `size`
                - `int`, optional
                - target size of the padded arrays
                - ever entry in `X` will have that length
                - the default is `None`
                    - will pad to the length of the longest entry in `X`
            - `subsampling_mode`
                - `Literal['first','last','random']`, optional
                - mode to use for subsampling
                    - in case an entry in `X` is longer than `size`
                - allowed options are
                    - `'first'`
                        - will use the first `size` elements of entries in `X` that exceed `size`
                    - `'last'`
                        - will use the last `size` elements of entries in `X` that exceed `size`
                    - `'random'`
                        - will use a random subsample of `size` elements of entries in `X` that exceed `size`
                - the default is `'first'`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Infered Attributes
        ------------------
            - `size_fitted`
                - `int`
                - `size` after running `self.fit()`

        Methods
        -------
            - `fit()`
            - `transform()`
            - `fit_transform()`
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
        size:int=None,
        subsampling_mode:Literal['first','last','random']='first',
        verbose:int=0         
        ) -> None:

        self.size               = size
        self.subsampling_mode   = subsampling_mode
        self.verbose            = verbose

        #infered attributes
        self.size_fitted        = size

        return
    
    def __repr__(self) -> str:
        return (
            f'Pad2Size(\n'
            f'    size={repr(self.size)},\n'
            f'    subsampling_mode={repr(self.subsampling_mode)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def fit(self,
        X:List[np.ndarray], y:np.ndarray=None,
        ) -> None:
        """
            - method to fit the transformer
            
            Parameters
            ----------
                - `X`
                    - `List[np.ndarray]`
                    - list containing arrays of differnt lengths to be padded
                - `y`
                    - `np.ndarray`, optional
                    - labels corresponding to `X`
                    - not used, only for consistency
                    - the default is `None`
            
            Raises
            ------

            Returns
            -------

            Comments
            --------

        """
        #obtain padding length
        if self.size is None:
            self.size_fitted = max([len(x) for x in X])
        else:
            self.size_fitted = self.size
        
        return
    
    def transform(self,
        X:List[np.ndarray], y:np.ndarray=None,
        size:int=None,
        subsampling_mode:Literal['first','last','random']=None,
        pad_kwargs:dict=None,
        ) -> np.ndarray:
        """
            - method to transform the input

            Parameters
            ----------
                - `X`
                    - `List[np.ndarray]`
                    - list containing arrays of differnt lengths to be padded
                - `y`
                    - `np.ndarray`, optional
                    - labels corresponding to `X`
                    - not used, only for consistency
                    - the default is `None`
                - `size`
                    - `int`, optional
                    - target size of the padded arrays
                    - ever entry in `X` will have that length
                    - overrides `self.size`
                    - the default is `None`
                        - will fall back to `self.size`
                - `subsampling_mode`
                    - `Literal['first','last','random']`, optional
                    - mode to use for subsampling
                        - in case an entry in `X` is longer than `size`
                    - allowed options are
                        - `'first'`
                            - will use the first `size` elements of entries in `X` that exceed `size`
                        - `'last'`
                            - will use the last `size` elements of entries in `X` that exceed `size`
                        - `'random'`
                            - will use a random subsample of `size` elements of entries in `X` that exceed `size`
                    - overrides `self.subsampling_mode`
                    - the default is `None`
                        - will fall back to `self.subsampling_mode`
                - `pad_kwargs`
                    - `dict`, optional
                    - additional kwargs to pass to `np.pad()`
                    - the default is `None`
                        - will be set to `dict(constant_values=(np.nan))`
                            - i.e. padding values are `np.nan`
            Raises
            ------

            Returns
            -------
                - `X_pad`
                    - `np.ndarray`
                    - padded version of `X`
                    - has shape `(len(X),size)`

            Comments
            --------     
        """

        #default parameters
        if size is None:                size                = self.size_fitted
        if subsampling_mode is None:    subsampling_mode    = self.subsampling_mode
        if pad_kwargs is None:          pad_kwargs          = dict(constant_values=(np.nan))

        #init output
        X_pad = np.empty((len(X),size))

        #transform `X`
        for idx, x in enumerate(X):
            #execute padding
            if x.shape[0] < size:
                to_add = size-x.shape[0]
                pad_width = (0,to_add)
                X_pad[idx] = np.pad(x, pad_width=pad_width, **pad_kwargs)
            #execute subsampling        
            elif x.shape[0] > size:
                #set indices according to mode
                if subsampling_mode == 'first': idxs = slice(size)
                elif subsampling_mode == 'last':idxs = slice(-size)
                elif subsampling_mode == 'random':
                    idxs = np.random.choice(np.arange(x.shape[0]), size=size, replace=(x.shape[0]<size))
                    idxs = np.sort(idxs)
                #first `size` entries
                X_pad[idx] = x[idxs]
            #no modification
            else:
                X_pad[idx] = x

        return X_pad

    def fit_transform(self,
        X:List[np.ndarray], y:np.ndarray=None,
        fit_kwargs:dict=None,
        transform_kwargs:dict=None,
        ) -> np.ndarray:
        """
            - method to fit the transformer and transform the input
        
            Parameters
            ----------
                - `X`
                    - `List[np.ndarray]`
                    - list containing arrays of differnt lengths to be padded
                - `y`
                    - `np.ndarray`, optional
                    - labels corresponding to `X`
                    - not used, only for consistency
                    - the default is `None`
                - `fit_kwargs`
                    - `dict`, optional
                    - additional kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`
                - `transform_kwargs`
                    - `dict`, optional
                    - additional kwargs to pass to `self.transform()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `X_pad`
                    - `np.ndarray`
                    - padded version of `X`
                    - has shape `(len(X),size)`

            Comments
            --------

        """
        if fit_kwargs is None:      fit_kwargs       = dict()
        if transform_kwargs is None:transform_kwargs = dict()

        self.fit(X, y, **fit_kwargs)
        X_pad = self.transform(X, y, **transform_kwargs)
        
        return X_pad
    
    def plot_result(self,
        X_pad:np.ndarray,
        X:List[np.ndarray]=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the transformed result

            Parameters
            ----------
                - `X_pad`
                    - `np.ndarray`
                    - padded version of the input `X`
                - `X`
                    - `List[np.ndarray]`, optional
                    - list containing arrays of differnt lengths to be padded
                    - original input before transformation
                    - the default is `None`
                        - will be ignored

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    -  created figure
                - `axs`
                    - `plt.Axes`
                    - axis corresponding to `fig`

            Comments
            --------
        
        """
        #default values
        if X is None: X = [np.nan]*X_pad.shape[0]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        colors = generate_colors(len(X))
        for x, xp, c in zip(X, X_pad, colors):
            ax1.plot(x,  c=c,   ls='-', lw=5)
            ax1.plot(xp, c='w', ls='-',lw=3)
            ax1.plot(xp, c=c, ls='--', lw=2)
        ax1.plot(np.nan, color='tab:blue', ls='-',  lw=5, label='Original')
        ax1.plot(np.nan, color='tab:blue', ls='--', lw=2, label='Padded')
        ax1.legend()
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        axs = fig.axes

        return fig, axs

class PeriodicExpansion:
    """
        - class to expand periodic timeseries on either side (min or max)
            - takes all datapoints up to a reference x-value
            - appends them to the original according to specification
        - follows structure of sklearn transformers
        
        Attributes
        ----------
            - `x_ref_min`
                - `float`, optional
                - reference x-value for appending to minimum side
                    - will be used in order to determine which phases to consider for appending
                - used if `minmax` contains `min`
                - the default is `0`
            - `x_ref_max`
                - `float`, optional
                - reference x-value for appending to maximum side
                    - will be used in order to determine which phases to consider for appending
                - used if `minmax` contains `max`
                - the default is `0`
            - `minmax`
                - `str`, optional
                - specify where to extend the dataseries
                - if `'min'` is contained in `minmax`
                    - will expand on the minimum side
                    - will consider all phases from `x_ref_min` to the maximum phase
                - if `'max'` is contained in `minmax`
                    - will expand on the maximum side
                    - will consider all phases from the minimum phase up to `x_ref_max`
                - the default is `None`
                    - will be set to `'minmax'`

        Methods
        -------
            - `fit()`
            - `transform()`
            - `fit_transform()`
            - `plot_result()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`

        Comments
        --------

    """

    def __init__(self,
        x_ref_min:float=0, x_ref_max:float=0,
        minmax:str=None,
        verbose:int=0,
        ) -> None:
        
        self.x_ref_min  = x_ref_min
        self.x_ref_max  = x_ref_max
        if minmax is None:  self.minmax = 'minmax'
        else:               self.minmax = minmax
        
        self.verbose    = verbose

        return

    def __repr__(self) -> str:
        
        return (
            f'PeriodicExpansion(\n'
            f'    x_ref_min={repr(self.x_ref_min)}, x_ref_max={repr(self.x_ref_max)},\n'
            f'    minmax={repr(self.minmax)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def fit(self,
        X:np.ndarray, y:np.ndarray=None,
        x_ref_min:float=None, x_ref_max:float=None,
        minmax:str=None,
        ) -> None:
        """
            - method to fit the transformer
            
            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - contains dataseries (y-values) to be transformed
                    - has to be of shape `(nsamples,nfeatures)`
                - `y`
                    - `np.ndarray`, optional
                    - 1D array
                    - contains x-values for all dataseries/features contained in `X`
                    - the default is `None`
                        - will generate x-values between 0 and 1
                        - i.e. `y = np.linspace(0,1,X.shape[1])` will be called
                - `x_ref_min`
                    - `float`, optional
                    - reference x-value for appending to minimum side
                        - will be used in order to determine which phases to consider for appending
                    - used if `minmax` contains `min`
                    - overrides `self.x_ref_min`
                    - the default is `None`
                        - will fall back to `self.x_ref_min`
                - `x_ref_max`
                    - `float`, optional
                    - reference x-value for appending to maximum side
                        - will be used in order to determine which phases to consider for appending
                    - used if `minmax` contains `max`
                    - overrides `self.x_ref_max`
                    - the default is `None`
                        - will fall back to `self.x_ref_max`
                - `minmax`
                    - `str`, optional
                    - specify where to extend the dataseries
                    - if `'min'` is contained in `minmax`
                        - will expand on the minimum side
                        - will consider all phases from `x_ref_min` to the maximum phase
                    - if `'max'` is contained in `minmax`
                        - will expand on the maximum side
                        - will consider all phases from the minimum phase up to `x_ref_max`
                    - overrides `self.minmax`
                    - the default is `None`
                        - will fall back to `self.minmax`
                        
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if x_ref_min is None:   x_ref_min   = self.x_ref_min
        if x_ref_max is None:   x_ref_max   = self.x_ref_max
        if minmax is None:      minmax      = self.minmax
        if y is None:           y           = np.linspace(0,1,X.shape[1])

        #internalize input arrays
        self.X      = X
        self.y      = y

        #initialize output arrays
        self.X_expanded = X.copy()
        self.y_expanded = y.copy()

        if 'min' in minmax:
            y_bool = (x_ref_min < y)
            
            X_append = X[:,y_bool]
            y_append = np.nanmin(y) - (np.nanmax(y) - y[y_bool])
            self.X_expanded = np.append(self.X_expanded, X_append, axis=1)
            self.y_expanded = np.append(self.y_expanded, y_append)

        if 'max' in minmax:
            y_bool = (y < x_ref_max)

            X_append = X[:,y_bool]
            y_append = np.nanmax(y) + (y[y_bool] - np.nanmin(y))
            self.X_expanded = np.append(self.X_expanded, X_append, axis=1)
            self.y_expanded = np.append(self.y_expanded, y_append)

        return
    
    def transform(self,
        X:np.ndarray=None, y:np.ndarray=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to transform the input

            Parameters
            ----------
                - `X`
                    - `np.ndarray`, optional
                    - not needed in this method
                    - contains dataseries to be transformed
                    - the default is `None`
                - `y`
                    - `np.ndarray`, optional
                    - not needed in this method
                    - contains x-values for all dataseries/features contained in `X`
                    - the default is `None`
                        
            Raises
            ------

            Returns
            -------
                - `X_expanded`
                    - `np.ndarray`
                    - the transformed version of `X`
                    - i.e. `X` with datapoints appended according to specification
                - `y_expanded`
                    - `np.ndarray`
                    - the transformed version of `y`
                    - i.e. `y` with datapoints appended according to specification

            Comments
            --------            
        """


        return self.X_expanded, self.y_expanded
    
    def fit_transform(self,
        X:np.ndarray, y:np.ndarray=None,
        fit_kwargs:dict=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer and transform the input

            Parameters
            ----------
                - `X`
                    - `np.ndarray`, optional
                    - not needed in this method
                    - contains dataseries to be transformed
                    - the default is `None`
                - `y`
                    - `np.ndarray`, optional
                    - 1D array
                    - contains x-values for all dataseries/features contained in `X`
                    - the default is `None`
                        - will generate x-values between 0 and 1
                        - i.e. `y = np.linspace(0,1,X.shape[1])` will be called
                - `fit_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`
                        
            Raises
            ------

            Returns
            -------
                - `X_expanded`
                    - `np.ndarray`
                    - the transformed version of `X`
                    - i.e. `X` with datapoints appended according to specification
                - `y_expanded`
                    - `np.ndarray`
                    - the transformed version of `y`
                    - i.e. `y` with datapoints appended according to specification

            Comments
            --------
        
        """

        if fit_kwargs is None: fit_kwargs = {}

        self.fit(X, y, **fit_kwargs)

        return self.X_expanded, self.y_expanded
    
    def plot_result(self,
        cmap:Union[str,mcolors.Colormap]='nipy_spectral',
        sctr_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the result after successful transformation

            Parameters
            ----------
                - `cmap`
                    - `str`, `mcolors.Colormap`, optional
                    - colormap to use for plotting different samples in `X`
                    - the default is `nipy_spectral`
                - `sctr_kwargs`
                    - `dict`, optinonal
                    - kwargs to pass to `ax.scatter`
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
                    - axis corresponding to `fig`
            
            Comments
            --------
        """

        if sctr_kwargs is None: sctr_kwargs = {}

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        colors = generate_colors(classes=self.X_expanded.shape[0], cmap=cmap)
        for x, xe, c in zip(self.X, self.X_expanded, colors):
            exp_bool = ~np.isin(self.y_expanded, self.y)
            ax1.scatter(self.y,                    x,            facecolor='none', ec=c,      **sctr_kwargs)
            ax1.scatter(self.y_expanded[exp_bool], xe[exp_bool], facecolor=c,      ec='none', **sctr_kwargs)
            
        ax1.scatter(np.nan, np.nan, facecolor='none',     ec='tab:blue', label='Original',        **sctr_kwargs)
        ax1.scatter(np.nan, np.nan, facecolor='tab:blue', ec='none',     label='Newly Generated', **sctr_kwargs)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        ax1.legend()

        axs = fig.axes

        return fig, axs

class PeriodicShift:
    """
        - class to shift a dataseries considering periodic boundary conditions

        Attributes
        ----------
            - `shift`
                - `float`
                - magnitude of the shift to apply to the data
                - when set to `0` will only enforce periodic boundaries
            - `borders`
                - `Tuple[float]`
                - upper and lower boundary of the periodic interval
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
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------

    """
    def __init__(self,
        shift:float,
        borders:Tuple[float],
        verbose:int=0,
        ) -> None:

        self.shift = shift
        self.borders = borders
        self.verbose = verbose

        return
    
    def __repr__(self) -> str:

        return (
            f'{self.__class__.__name__}(\n'
            f'    shift={repr(self.shift)},\n'
            f'    borders={repr(self.borders)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def fit(self,
        x:np.ndarray=None,
        shift:float=None,
        borders:Tuple[float]=None,
        verbose:int=None,
        *args,
        ) -> None:
        """
            - method to fit the transformer
            - will overwrite `self.shift` and `self.borders` with new values (if provided)

            Parameters
            ----------
                - `x`
                    - `np.ndarray`, optional
                    - input array to be shifted
                    - not needed
                    - the default is `None`
                - `shift`
                    - `float`, optional
                    - magnitude of the shift to apply to the data
                    - when set to `0` will only enforce periodic boundaries
                    - updates `self.shift` to new value
                    - the default is `None`
                        - no keep original value
                - `borders`
                    - `Tuple[float]`, optional
                    - upper and lower boundary of the periodic interval
                    - updates `self.borders` to new value
                    - the default is `None`
                        - no keep original value
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

            Comments
            --------
        """

        if shift is not None: self.shift = shift
        if borders is not None: self.borders = borders
        
        return
    
    def transform(self,
        x:np.ndarray,
        verbose:int=None,
        *args,
        ) -> np.ndarray:
        """
            - method to transform the input

            Parameters
            ----------
                - `x`
                    - `np.ndarray`, optional
                    - input array to be shifted
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
                - `x_shifted`
                    - `np.ndarray`
                    - shifted version of `x`

            Comments
            --------
        """
        #shift array
        lower_bound = np.min(self.borders)
        upper_bound = np.max(self.borders)

        
        #apply shift
        x_shifted = x+self.shift
        
        #resproject into interval
        out_of_lower_bounds = (x_shifted < lower_bound)
        out_of_upper_bounds = (x_shifted > upper_bound)

        lower_deltas = lower_bound-x_shifted[out_of_lower_bounds]
        x_shifted[out_of_lower_bounds] = upper_bound - lower_deltas    
        upper_deltas = x_shifted[out_of_upper_bounds]-upper_bound
        x_shifted[out_of_upper_bounds] =lower_bound + upper_deltas
        

        return x_shifted

    def fit_transform(self,
        x:np.ndarray,
        shift:float=None,
        borders:Tuple[float]=None,
        verbose:int=None,
        *args,
        ) -> np.ndarray:
        """
            - method to fit the transformer and transform the input

            Parameters
            ----------
                - `x`
                    - `np.ndarray`, optional
                    - input array to be shifted
                - `shift`
                    - `float`, optional
                    - magnitude of the shift to apply to the data
                    - when set to `0` will only enforce periodic boundaries
                    - updates `self.shift` to new value
                    - the default is `None`
                        - no keep original value
                - `borders`
                    - `Tuple[float]`, optional
                    - upper and lower boundary of the periodic interval
                    - updates `self.borders` to new value
                    - the default is `None`
                        - no keep original value
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
                - `x_shifted`
                    - `np.ndarray`
                    - shifted version of `x`

            Comments
            --------
        """
        self.fit(x, shift=shift, borders=borders, *args)
        x_shifted = self.transform(x, *args)

        return x_shifted
    
    def plot_result(self,
        x:np.ndarray, y:np.ndarray=None,
        ax:plt.Axes=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visualize the current result when applying the transformer

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - array to be transformed
                - `y`
                    - `np.ndarray`, optional
                    - y-values corresponding to `x`
                    - the default is `None`
                        - will be set to `np.linspace(0,1,x.shape[0])`
                - `ax`
                    - `plt.Axes`, optional
                    - axes to plot into
                    - the default is `None`
                        - will generate new `fig` and `axs`
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - the created figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`

            Comments
            --------

        """
        
        #default parmeters
        if y is None: y = np.linspace(0,1,x.shape[0])
        
        x_shifted = self.transform(x, verbose=0)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.plot(   x,        y, label="Original")
        ax.scatter(x_shifted,y, label="Shifted")
        ax.axvline(self.borders[0], color="tab:grey", ls="--", label="Boundaries")
        ax.axvline(self.borders[1], color="tab:grey", ls="--", label="_Boundaries")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        axs = fig.axes

        return fig, axs

class Periodize:
    """
        - class to generate a periodic signal out of an existing one

        Attributes
        ----------
            - `repetitions`
                - `float`, optional
                - number of periodic repetitions of the signal
                - can also be a fraction
                - will be infered from `size` if not provided at all
                - the default is `None`
            - `size`
                - `int`, optional
                - desired output size of the signal after periodizing
                - will be infered from `repetitions` if not provided at all
                - the default is `None`
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
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------

    """
    def __init__(self,
        repetitions:float=None,
        size:int=None,
        verbose:int=0,
        ) -> None:
        
        self.repetitions    = repetitions
        self.size           = size
        self.verbose        = verbose

        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    repetitions={repr(self.repetitions)},\n'
            f'    size={repr(self.size)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def fit(self,
        x:np.ndarray, y:np.ndarray,
        repetitions:float=None,
        size:int=None,
        verbose:int=None,
        ) -> None:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the input series to be periodized
                - `y`
                    - `np.ndarray`
                    - y-values of the input series to be periodized
                - `repetitions`
                    - `float`, optional
                    - number of periodic repetitions of the signal
                    - can also be a fraction
                    - will be infered from `size` if not provided at all
                    - the default is `None`
                        - will fall back to `self.repetitions`
                - `size`
                    - `int`, optional
                    - desired output size of the signal after periodizing
                    - will be infered from `repetitions` if not provided at all
                    - the default is `None`
                        - will fall back to `self.size`
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

            Comments
            --------
        """
        #default params
        if repetitions is None: repetitions = self.repetitions
        if size is None: size = self.size
        
        #check feasibility
        assert (repetitions is not None) or (size is not None), "At least one of `repetitions` and `size` has to be `not None`."

        #compute new metrics
        if size is None:        size = int(y.shape[0]*repetitions)
        if repetitions is None: repetitions = size/y.shape[0]

        #update parameters
        self.size = size
        self.repetitions = repetitions

        return
    
    def transform(self,
        x:np.ndarray, y:np.ndarray,
        verbose:int=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the input series to be periodized
                - `y`
                    - `np.ndarray`
                    - y-values of the input series to be periodized
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
                - `x_periodized`
                    - `np.ndarray`
                    - periodized version of `x`
                - `y_periodized`
                    - `np.ndarray`
                    - periodized version of `y`

            Comments
            --------
        """
        #init periodized arrays for sample
        y_periodized = np.zeros(self.size)
        x_periodized = np.zeros(self.size)
        
        #add complete repetitions
        for r in range(int(self.size//len(y))):
            start_idx = r*y.shape[0]
            end_idx = (r+1)*y.shape[0]
            end_offset = r*(np.nanmax(x)-np.nanmin(x))
            x_periodized[start_idx:end_idx] += (x+end_offset) #offset xi to get correct xvalues
            y_periodized[start_idx:end_idx] += y
        #add fractional repetitions
        if self.size < len(y):
            end_idx = 0
            end_offset = 0
        else:
            end_offset = (r+1)*(np.nanmax(x)-np.nanmin(x))
        x_periodized[end_idx:] = (x[:self.size%len(y)]+end_offset)
        y_periodized[end_idx:] = y[:self.size%len(y)]

        return x_periodized, y_periodized
    
    def fit_transform(self,
        x:np.ndarray, y:np.ndarray,
        repetitions:float=None,
        size:int=None,
        verbose:int=None,        
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer and transform the input

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the input series to be periodized
                - `y`
                    - `np.ndarray`
                    - y-values of the input series to be periodized
                - `repetitions`
                    - `float`, optional
                    - number of periodic repetitions of the signal
                    - can also be a fraction
                    - will be infered from `size` if not provided at all
                    - the default is `None`
                        - will fall back to `self.repetitions`
                - `size`
                    - `int`, optional
                    - desired output size of the signal after periodizing
                    - will be infered from `repetitions` if not provided at all
                    - the default is `None`
                        - will fall back to `self.size`
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
                - `x_periodized`
                    - `np.ndarray`
                    - periodized version of `x`
                - `y_periodized`
                    - `np.ndarray`
                    - periodized version of `y`

            Comments
            --------
        """

        self.fit(x, y, repetitions=repetitions, size=size)
        x_periodized, y_periodized = self.transform(x, y)

        return x_periodized, y_periodized
    
    def plot_result(self,
        x:np.ndarray, y:np.ndarray,
        ax:plt.Axes=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visualize the current result when applying the transformer

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - array to be transformed
                - `y`
                    - `np.ndarray`
                    - y-values corresponding to `x`
                - `ax`
                    - `plt.Axes`, optional
                    - axes to plot into
                    - the default is `None`
                        - will generate new `fig` and `axs`
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - the created figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`

            Comments
            --------

        """

        x_per, y_per = self.transform(x, y)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        ax.scatter(x, y, label='Input Signal')
        ax.plot(x_per, y_per, c='w', lw=3)
        ax.plot(x_per, y_per, label='Periodized Signal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        
        axs = fig.axes

        return fig, axs

class Resample:
    """
        - class to resample some dataseries based on linear interpolation

        Attributes
        ----------
            - `size`
                - `int`
                - desired output size of the signal after resampling
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
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------

    """
    def __init__(self,
        size:int,
        verbose:int=0,
        ) -> None:
        
        self.size           = size
        self.verbose        = verbose

        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    size={repr(self.size)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def fit(self,
        x:np.ndarray=None, y:np.ndarray=None,
        size:int=None,
        verbose:int=None,
        ) -> None:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the input series to be resampled
                - `y`
                    - `np.ndarray`
                    - y-values of the input series to be resampled
                - `size`
                    - `int`, optional
                    - desired output size of the signal after resampling
                    - the default is `None`
                        - will fall back to `self.size`
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

            Comments
            --------
        """        
        if size is not None: self.size = size

        return

    def transform(self,
        x:np.ndarray, y:np.ndarray,
        sort_before:bool=True,
        verbose:int=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer and transform the input

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the input series to be resampled
                - `y`
                    - `np.ndarray`
                    - y-values of the input series to be resampled
                - `size`
                    - `int`, optional
                    - desired output size of the signal after resampling
                    - the default is `None`
                        - will fall back to `self.size`
                - `sort_before`
                    - `bool`, optional
                    - whether to sort the input based on `x` before transforming
                    - the default is `True`
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
                - `x_res`
                    - `np.ndarray`
                    - resampled version of `x`
                - `y_res`
                    - `np.ndarray`
                    - resampled version of `y`

            Comments
            --------
        """
        if sort_before:
            idxs = np.argsort(x)
            x_, y_ = x[idxs], y[idxs]

        x_res = np.linspace(np.nanmin(x), np.nanmax(x), self.size)

        y_res =  np.interp(x_res, x_, y_)

        return x_res, y_res

    def fit_transform(self,
        x:np.ndarray, y:np.ndarray,
        size:int=None,
        verbose:int=None,
        transform_kwargs:dict=None
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer and transform the input

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the input series to be resampled
                - `y`
                    - `np.ndarray`
                    - y-values of the input series to be resampled
                - `size`
                    - `int`, optional
                    - desired output size of the signal after resampling
                    - the default is `None`
                        - will fall back to `self.size`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `transform_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.transform()`
                    - the default is `None`
                        - will be set to `dict()`
            
            Raises
            ------

            Returns
            -------
                - `x_res`
                    - `np.ndarray`
                    - resampled version of `x`
                - `y_res`
                    - `np.ndarray`
                    - resampled version of `y`

            Comments
            --------
        """
        if transform_kwargs is None: transform_kwargs = dict()

        self.fit(x, y, size)
        x_res, y_res = self.transform(x, y, **transform_kwargs)


        return x_res, y_res
    
    def plot_result(self,
        x:np.ndarray, y:np.ndarray,
        ax:plt.Axes=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visualize the current result when applying the transformer

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - array to be transformed
                - `y`
                    - `np.ndarray`
                    - y-values corresponding to `x`
                - `ax`
                    - `plt.Axes`, optional
                    - axes to plot into
                    - the default is `None`
                        - will generate new `fig` and `axs`
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - the created figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`

            Comments
            --------

        """        
        x_res, y_res = self.transform(x, y)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.scatter(x, y, label="Input")
        ax.scatter(x_res, y_res, label="Resampled")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        axs = fig.axes        

        return fig, axs
    
#%%functions
def fold(
    time:np.ndarray,
    period:float, tref:float=None,
    verbose=0,
    ) -> Tuple[np.ndarray,np.ndarray]:
    """
        - takes an array of times
            - folds it onto a specified period into phase space
            - returns folded array of phases (in interval 0 to 1) and periods (0 to `period`)

        Parameters
        ----------
            - `time`
                - `np.ndarray`
                - times to be folded with the specified period
            - `period` 
                - `float`
                - period to fold the times onto
            - `tref`
                - `float`, optional
                - reference time to consider when folding the lightcurve
                - the default is `None`
                    - will take `np.nanmin(time)` as reference
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Raises
        ------

        Returns
        -------
            - `phases_folded`
                - `np.ndarray`
                - phases corresponding to the given `time` folded onto `period`
            - `periods_folded`
                - `np.ndarray`
                - `phases_folded` in time domain

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------
    """

    #convert to phases
    phases = time2phase(time=time, period=period, tref=tref, verbose=verbose)
    
    #fold phases by getting the remainder of the division by the ones-value 
    #this equals getting the decimal numbers of that specific value
    #+1 because else a division by 0 would occur
    #floor always rounds down a value to the ones (returns everything before decimal point)
    phases_folded = np.mod(phases,1)

    periods_folded = phases_folded * period

    return phases_folded, periods_folded

def phase2time(
    phase:Union[np.ndarray,float],
    period:Union[np.ndarray,float],
    tref:Union[np.ndarray,float]=0,
    verbose:int=0,
    ) -> Union[np.ndarray,float]:
    """
        - converts a given array of phases into its respective time equivalent

        Parameters
        ----------
            - `phases`
                - `np.ndarray`, `float`
                - the phases to convert to times
            - `period`
                - `np.ndarray`, `float`
                - the given period(s) the phase describes
            - `tref`
                - `np.ndarray`, `float`, optional
                - reference time
                    - i.e. offset from `time==0`
                - the default is `0`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Raises
        ------

        Returns
        -------
            - `time`
                - `np.array`, float
                - the resulting time array, when the phases are converted 

        Dependencies
        ------------
            - `typing`

        Comments
        --------
            - operates with phases in the interval `[0,1]`
    """

    time = phase*period + tref
    
    return time

def time2phase(
    time:np.ndarray,
    period:float, tref:float=None,
    verbose=0,        
    ) -> Tuple[np.ndarray,np.ndarray]:
    """
        - takes an array of times
            - convert it into phase space by means of a specified period
            - returns array of phases (in interval 0 to 1)

        Parameters
        ----------
            - `time`
                - `np.ndarray`
                - times to be folded with the specified period
            - `period` 
                - `float`
                - period to fold the times onto
            - `tref`
                - `float`, optional
                - reference time to consider when folding the lightcurve
                - the default is `None`
                    - will take `np.nanmin(time)` as reference
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Raises
        ------

        Returns
        -------
            - `phase`
                - `np.ndarray`
                - phases corresponding to the given `time` w.r.t. `period`

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------
    """
    
    if tref is None:
        tref = np.nanmin(time)

    delta_t = time-tref
    phase = delta_t/period
    
    return phase
