
#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy as np
from typing import Union, Tuple, Callable

from astroLuSt.visualization.plotting import generate_colors
from astroLuSt.monitoring import formatting as almof


#%%classes
class PeriodicExpansion:
    """
        - class to expand periodic timeseries on either side (min or max)
            - takes all datapoints up to a reference x-value
            - appends them to the original according to specification
        - follows structure of sklearn transformers
        
        Attributes
        ----------
            - `x_ref_min`
                - float, optional
                - reference x-value for appending to minimum side
                    - will be used in order to determine which phases to consider for appending
                - used if `minmax` contains `min`
                - the default is 0
            - `x_ref_max`
                - float, optional
                - reference x-value for appending to maximum side
                    - will be used in order to determine which phases to consider for appending
                - used if `minmax` contains `max`
                - the default is 0
            - `minmax`
                - str, optional
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
            - matplotlib
            - numpy

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
                    - np.ndarray
                    - contains dataseries to be transformed
                - `y`
                    - np.ndarray, optional
                    - 1D array
                    - contains x-values for all dataseries/features contained in `X`
                    - the default is `None`
                        - will generate x-values between 0 and 1
                        - i.e. `y = np.linspace(0,1,X.shape[1])` will be called
                - `x_ref_min`
                    - float, optional
                    - reference x-value for appending to minimum side
                        - will be used in order to determine which phases to consider for appending
                    - used if `minmax` contains `min`
                    - overrides `self.x_ref_min`
                    - the default is `None`
                        - will fall back to `self.x_ref_min`
                - `x_ref_max`
                    - float, optional
                    - reference x-value for appending to maximum side
                        - will be used in order to determine which phases to consider for appending
                    - used if `minmax` contains `max`
                    - overrides `self.x_ref_max`
                    - the default is `None`
                        - will fall back to `self.x_ref_max`
                - `minmax`
                    - str, optional
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
                    - np.ndarray, optional
                    - not needed in this method
                    - contains dataseries to be transformed
                    - the default is `None`
                - `y`
                    - np.ndarray, optional
                    - not needed in this method
                    - contains x-values for all dataseries/features contained in `X`
                    - the default is `None`
                        
            Raises
            ------

            Returns
            -------
                - `X_expanded`
                    - np.ndarray
                    - the transformed version of `X`
                    - i.e. `X` with datapoints appended according to specification
                - `y_expanded`
                    - np.ndarray
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
                    - np.ndarray, optional
                    - not needed in this method
                    - contains dataseries to be transformed
                    - the default is `None`
                - `y`
                    - np.ndarray, optional
                    - not needed in this method
                    - contains x-values for all dataseries/features contained in `X`
                    - the default is `None`
                - `fit_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `{}`
                        
            Raises
            ------

            Returns
            -------
                - `X_expanded`
                    - np.ndarray
                    - the transformed version of `X`
                    - i.e. `X` with datapoints appended according to specification
                - `y_expanded`
                    - np.ndarray
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
                    - str, mcolors.Colormap, optional
                    - colormap to use for plotting different samples in `X`
                    - the default is `nipy_spectral`
                - `sctr_kwargs`
                    - dict, optinonal
                    - kwargs to pass to `ax.scatter`
                    - the default is `None`
                        - willbe set to `{}`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - matplotlib Figure
                    - created figure
                - `axs`
                    - plt.Axes
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
    

#%%functions

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
                - np.ndarray, float
                - the phases to convert to times
            - `period`
                - np.ndarray, float
                - the given period(s) the phase describes
            - `tref`
                - np.ndarray, float, optional
                - reference time
                    - i.e. offset from `time==0`
                - the default is 0
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------

        Returns
        -------
            - `time`
                - np.array, float
                - the resulting time array, when the phases are converted 

        Dependencies
        ------------
            - typing

        Comments
        --------
            - operates with phases in the interval [0,1]
    """
    import warnings

    warnings.warn('This function is deprecated. Use the new version in `astroLuSt.physics.timeseries`!')

    time = phase*period + tref
    
    return time

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
                - np.ndarray
                - times to be folded with the specified period
            - `period` 
                - float
                - period to fold the times onto
            - `tref`
                - float, optional
                - reference time to consider when folding the lightcurve
                - the default is `None`
                    - will take `np.nanmin(time)` as reference
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------

        Returns
        -------
            - `phases_folded`
                - np.ndarray
                - phases corresponding to the given `time` folded onto `period`
            - `periods_folded`
                - np.ndarray
                - `phases_folded` in time domain

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------
    """

    if tref is None:
        tref = np.nanmin(time)

    delta_t = time-tref
    phases = delta_t/period
    
    #fold phases by getting the remainder of the division by the ones-value 
    #this equals getting the decimal numbers of that specific value
    #+1 because else a division by 0 would occur
    #floor always rounds down a value to the ones (returns everything before decimal point)
    phases_folded = np.mod(phases,1)

    periods_folded = phases_folded * period

    return phases_folded, periods_folded

def resample(
    x:np.ndarray, y:np.ndarray,
    ndatapoints:int=50,
    sort_before:bool=True,
    verbose:int=0
    ) -> Tuple[np.ndarray,np.ndarray,Figure,plt.Axes]:
    """
        - function to resample a dataseries `y(x)` to nfeatures new datapoints via interpolation

        Parameters
        ----------
            - `x`
                - np.ndarray
                - independent input variable x
            - `y`
                - np.ndarray
                - dependent variable (`y(x)`)
            - `ndatapoints`
                - int, optional
                - number of datapoints of the resampled dataseries
                - the default is 50
            - `sort_before`
                - bool, optional
                - whether to sort the input arrays `x` and `y` with regards to `x` before resampling
                - the default is `True`
            - `verbose`
                - int optional
                - verbosity level
                - the default is 0
            
        Raises
        ------

        Returns
        -------
            - `interp_x`
                - np.ndarray
                - resamples array of `x`
            - `interp_y`
                - np.ndarray
                - resamples array of `y`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing
        
        Comments
        --------

    """

    #try converting to numpy (otherwise bin generation might fail)
    if not isinstance(x, np.ndarray):
        try:
            x = x.to_numpy()
        except:
            raise TypeError(f'"x" hat to be of type np.ndarray and not {type(x)}')
    if not isinstance(y, np.ndarray):
        try:
            y = y.to_numpy()
        except:
            raise TypeError(f'"y" hat to be of type np.ndarray and not {type(y)}')
            

    if sort_before:
        idxs = np.argsort(x)
        x_, y_ = x[idxs], y[idxs]

    interp_x = np.linspace(0, 1, ndatapoints)

    interp_y =  np.interp(interp_x, x_, y_)

    if verbose > 1:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y, label="Input")
        ax1.scatter(interp_x, interp_y, label="Resampled")

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        plt.tight_layout()
        plt.show()

        axs = fig.axes
    else:
        fig = None
        axs = None
    

    return interp_x, interp_y, fig, axs

def periodic_shift(
    input_array:np.ndarray,
    shift:float, borders:Union[list,np.ndarray],
    testplot:bool=False,
    verbose:int=0
    ) -> np.ndarray:
    """
        - function to shift an array considering periodic boundaries

        Parameters
        ----------
            - `input_array`
                - np.ndarray
                - array to be shifted along an interval with periodic boundaries
            - `shift`
                - float
                - magnitude of the shift to apply to the array
            - `borders`
                - list, np.ndarray
                - upper and lower boundary of the periodic interval
            - `testplot`
                - bool, optional
                - wether to show a testplot
                - the default is `False`
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------
            - `TypeError`
                - if the provided parameters are of the wrong type

        Returns
        -------
            - `shifted`
                - np.ndarray
                - array shifted by shift along the periodic interval in borders

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing

        Comments
        --------

    """        
    
    ################################
    #check if all types are correct#
    ################################
    
    if type(input_array) != np.ndarray:
        raise TypeError("input_array has to be of type np.ndarray! If you want to shift a scalar, simply convert it to an array and acess outputarray[0]")
    if (type(borders) != np.ndarray) and (type(borders) != list):
        raise TypeError("borders has to be of type np.array or list!")
    if (type(testplot) != bool):
        raise TypeError("testplot has to be of type bool")



    #############
    #shift array#
    #############
    lower_bound = np.min(borders)
    upper_bound = np.max(borders)

    
    #apply shift
    shifted = input_array+shift
    
    #resproject into interval
    out_of_lower_bounds = (shifted < lower_bound)
    out_of_upper_bounds = (shifted > upper_bound)

    lower_deltas = lower_bound-shifted[out_of_lower_bounds]
    shifted[out_of_lower_bounds] = upper_bound - lower_deltas    
    upper_deltas = shifted[out_of_upper_bounds]-upper_bound
    shifted[out_of_upper_bounds] =lower_bound + upper_deltas
    

    if verbose > 0:
        print("input_array: %a"%input_array)
        print("shifted_array: %a"%shifted)
        print("shift: %g"%shift)
        print("boundaries: %g, %g"%(lower_bound, upper_bound))
    
    if testplot:
        y_test = np.ones_like(shifted)
        
        fig = plt.figure()
        plt.suptitle("Testplot to visualize shift")
        # plt.plot(shifted[out_of_lower_bounds], y_test[out_of_lower_bounds]-1, color="b", marker=".", alpha=1, linestyle="", zorder=4)
        # plt.plot(shifted[out_of_upper_bounds], y_test[out_of_upper_bounds]+1, color="r", marker=".", alpha=1, linestyle="", zorder=4)
        plt.plot(shifted, y_test, color="r", marker="x", alpha=1, linestyle="", zorder=4, label="shifted array")
        plt.plot(input_array, y_test, color="k", marker=".", alpha=1, linestyle="", zorder=4, label="original array")
        plt.vlines([lower_bound, upper_bound], ymin=y_test.min()-1, ymax=y_test.max()+1, color="g", linestyle="--", label="boundaries")
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        plt.show()

    return shifted

def periodize(
    x:list, y:list,
    repetitions:Union[np.ndarray,float]=2, outshapes:Union[np.ndarray,int]=None,
    testplot:bool=False,
    verbose:int=0,
    ) -> Tuple[list,list]:
    """
        - function to create a periodic signal out of the `x` and `y` values of a time-series given in phase space
            - i.e., one repetition is provided

        Parameters
        ----------
            - `x`
                - list
                - has to be at least 2d
                - also heterogeneous 2d arrays are allowed
                - has to have same shapes as `y` in all axis
                - x-values to be periodized
            - `y`
                - list
                - has to be at least 2d
                - also heterogeneous 2d arrays are allowed
                - has to have same shapes as `x` in all axis
                - y-values to be periodized
            - `repetitions`
                - np.ndarray, float, optional
                - if np.ndarray
                    - has to have same length as `x` and `y`
                - number of times the signal (`y(x)`) shall be repeated
                - if `repetitions < 1` gets passed, the signal will be cut off at that point
                - the default is 2
            - `outshapes`
                - np.ndarray, int, optional
                - if np.ndarray
                    - has to have same length as `x` and `y`
                - desired output shape after periodizing
                - overrides `repetitions`
                - the default is `None`
                    - will be calculated from `repetitions`
            - `testplot`
                - bool, optional
                - whether to show a test-plot
                - the default is `False`
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------

        Returns
        -------
            - `x_periodized`
                - np.ndarray
                - same shape as `y`
                - times of the periodized signal
            - `y_periodized`
                - np.ndarray
                - same shape as `y`
                - y-values of the periodized signal

        Dependencies
        ------------
            - numpy
            - matplotlib
            - typing

        Comments
        --------
    
    """

    #make sure all shapes are correct
    try:
        y[0][0]
        x[0][0]
    except Exception as e:
        raise ValueError('`x` and `y` have to be 2d lists!')
    if isinstance(repetitions, (int,float)):    repetitions = np.array([repetitions]*len(y))
    if isinstance(outshapes, int):              outshapes   = np.array([outshapes]*len(y))

    #inintialize all relevant params depending on what has been provided
    if outshapes is None:   outshapes   = [int(yi.shape[0]*r) for yi, r in zip(y, repetitions)]
    else:                   repetitions = [os/yi.shape[0] for yi, os in zip(y, outshapes)]

    almof.printf(
        msg=f'Using the following: {outshapes=}, {repetitions=}.',
        context='periodize()',
        type='INFO',
        verbose=verbose,
    )

    #init output lists
    x_periodized = []
    y_periodized = []
    for xi, yi, outshape in zip(x, y, outshapes):

        #init periodized arrays for sample
        yi_periodized = np.zeros(outshape)
        xi_periodized = np.zeros(outshape)
        
        #add complete repetitions
        for r in range(int(outshape//len(yi))):
            start_idx = r*yi.shape[0]
            end_idx = (r+1)*yi.shape[0]
            end_offset = r*(np.nanmax(xi)-np.nanmin(xi))
            xi_periodized[start_idx:end_idx] += (xi+end_offset) #offset xi to get correct xvalues
            yi_periodized[start_idx:end_idx] += yi
        #add fractional repetitions
        if outshape < len(yi):
            end_idx = 0
            end_offset = 0
        else:
            end_offset = (r+1)*(np.nanmax(xi)-np.nanmin(xi))
        xi_periodized[end_idx:] = (xi[:outshape%len(yi)]+end_offset)
        yi_periodized[end_idx:] = yi[:outshape%len(yi)]

        #append to output
        y_periodized.append(yi_periodized)
        x_periodized.append(xi_periodized)
        
    if testplot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for idx, (xp, yp, xi, yi) in enumerate(zip(x_periodized, y_periodized, x, y)):
            lab_per = 'Periodized Signal'*(idx==0)
            lab_in  = 'Input Signal'*(idx==0)
            ax1.scatter(xi, yi, label=lab_in)
            ax1.plot(xp, yp, c='w', lw=3)
            ax1.plot(xp, yp, label=lab_per)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return x_periodized, y_periodized
