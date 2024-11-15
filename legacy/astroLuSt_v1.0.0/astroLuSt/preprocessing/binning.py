

#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from typing import Union, Tuple, Callable


#%%definitions
class Binning:
    """
        - class to execute data-binning on a given input series
        - essentially calculates a mean representative curve
        - the scatter of the original data and thus certainty of the representation is captured by the standard deviation of `y` in an interval


        Attributes
        ----------
            - `nintervals`
                - `float`, `int` optional
                - number of intervals/bins to generate
                - if a value between `0` and `1` is passed
                    - will be interpreted as fraction of input dataseries length
                - if a value greater than `1` is passed
                    - will be converted to integer
                    - will be interpreted as the number of intervals to use
                - the default is `0.5`
            - `npoints_per_interval`
                - `float`, `int`, optional
                - generate intervals/bins automatically such that each bin contains `npoints_per_interval` datapoints
                    - the last interval will contain all datapoints until the end of the dataseries
                - if between `0` and `1`
                    - will be interpreted as a fraction of the input dataseries length
                        - i.e. each bin contains `npoints_per_interval`*100% datapoints`
                - if greater `1`
                    - will be converted to `int`
                    - will be interpreted as actual number of datapoints
                - if set will overwrite `nintervals`
                - the default is `None`
                    - will use `nintervals` to generate the bins
            - `xmin`
                - `float`, optional
                - the minimum value to consider for the interval/bin creation
                - the default is `None`
                    - will use the minimum of the input-series `x`-values
            - `xmax`
                - `float`, optional
                - the maximum value to consider for the interval/bin creation
                - the default is `None`
                    - will use the maximum of the input-series `x`-values
            - `ddof`
                - `int`, optional
                - Delta Degrees of Freedom used in `np.nanstd()`
                - the default is `0`
            - `meanfunc_x`
                - `Callable`, optional
                - function to use to calculate the mean of each interval in `x`
                - the function shall take one argument
                    - the input dataseries `x`-values
                - the function shall return a single floating point value
                - the default is `None`
                    - will use `np.nanmean`
            - `meanfunc_y`
                - `Callable`, optional
                - function to use to calculate the mean of each interval in `y`
                - the function shall take one argument
                    - the input dataseries `y`-values
                - the function shall return a single floating point value
                - the default is `None`
                    - will use `np.nanmean`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
        
        Infered Attributes
        ------------------
            - `bins`
                - `np.ndarray`
                - array containing the boundaries of the intervals/bins used for binning the curve
            - `n_per_bin`
                - `np.ndarray`
                - contains the number of samples contained within each bin
            - `x`
                - `np.ndarray`
                - x-values of the input data series
            - `x_binned`
                - `np.ndarray`
                - binned values for input `x`
                - has shape `(1, nintervals)`
            - `y`
                - `np.ndarray`
                - y-values of the input data series
            - `y_binned`
                - `np.ndarray`
                - binned values for input `y`
                - has shape `(1, nintervals)`
            - `y_std`
                - `np.ndarray`
                - standard deviation of `y` for each interval
                - characterizes the scattering of the input curve
                - has shape `(1, nintervals)`

        Methods
        -------
            - `generate_bins()`
            - `bin_curve()`
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
        nintervals:Union[float,int]=0.2, npoints_per_interval:Union[float,int]=None,
        xmin:float=None, xmax:float=None,
        ddof:int=0,
        meanfunc_x:Callable=None, meanfunc_y:Callable=None,
        verbose:int=0,     
        ) -> None:
    
        self.nintervals = nintervals
        self.npoints_per_interval= npoints_per_interval
        self.xmin= xmin
        self.xmax= xmax
        self.ddof= ddof
        if meanfunc_x is None:
            self.meanfunc_x = np.nanmean
        else:
            self.meanfunc_x = meanfunc_x
        if meanfunc_y is None:
            self.meanfunc_y = np.nanmean
        else:
            self.meanfunc_y = meanfunc_y
        self.verbose= verbose

        pass

    def __repr__(self) -> str:

        return (
            f'{self.__class__.__name__}(\n'
            f'    nintervals={repr(self.nintervals)}, npoints_per_interval={repr(self.npoints_per_interval)},\n'
            f'    xmin={repr(self.xmin)}, xmax={repr(self.xmax)},\n'
            f'    ddof={repr(self.ddof)},\n'
            f'    meanfunc_x={repr(self.meanfunc_x)}, meanfunc_y={repr(self.meanfunc_y)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def generate_bins(self,
        x:np.ndarray, y:np.ndarray,
        nintervals:Union[float,int]=None, npoints_per_interval:Union[float,int]=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to generate the requested bins

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values w.r.t. which the binning shall be executed
                - `y`
                    - `np.ndarray`
                    - y-values to be binned
                - `nintervals`
                    - `float`, `int` optional
                    - nuber of intervals/bins to generate
                    - if a value between `0` and `1` is passed
                        - will be interpreted as fraction of input dataseries length
                    - if a value greater than `1` is passed
                        - will be converted to integer
                        - will be interpreted as the number of intervals to use
                    - overrides `self.nintervals`
                    - the default is `None`
                        - will fall back to `self.nintervals`
                - `npoints_per_interval`
                    - `int`, `int`, optional
                    - generate intervals/bins automatically such that each bin contains `npoints_per_interval` datapoints
                        - the last interval will contain all datapoints until the end of the dataseries
                    - if between `0` and `1`
                        - will be interpreted as a fraction of the input dataseries length
                            - i.e. each bin contains `npoints_per_interval`*100% datapoints
                    - if greater `1`
                        - will be converted to `int`
                        - will be interpreted as actual number of datapoints
                    - if set will overwrite `nintervals`
                    - overrides `nintervals`
                    - overrides `self.npoints_per_interval`
                    - the default is `None`
                        - will fall back to `self.npoints_per_interval`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overwrites `self.verbose` if set
                    - the default is `None`           
            
            Raises
            ------

            Returns
            -------
                - `self.bins`
                    - `np.ndarray`
                    - boundaries of the generated bins
            
            Comments
            --------
        """

        #set/overwrite internal attributes
        if npoints_per_interval is not None:
            nintervals = None
        elif npoints_per_interval is None and nintervals is None:
            nintervals = self.nintervals
            npoints_per_interval = self.npoints_per_interval
        elif npoints_per_interval is None and nintervals is not None:
            npoints_per_interval = None
        elif npoints_per_interval is not None and nintervals is None:
            nintervals = None

        if verbose is None: verbose = self.verbose

        #dynamically calculate bins if npoints_per_interval is specified 
        if npoints_per_interval is not None:
            if 0 < npoints_per_interval and npoints_per_interval < 1:
                #calculate `npoints_per_interval` as fraction of the shape of x and y 
                npoints_per_interval = int(self.npoints_per_interval*x.shape[0])
            elif npoints_per_interval >= 1:
                npoints_per_interval = int(npoints_per_interval)
            else:
                raise ValueError("`npoints_per_interval` has to be greater than 0!")

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
            
            sortidx = np.argsort(x)
            x_ = np.array(x[sortidx])
            y_ = np.array(y[sortidx])
            chunck_idxs = np.arange(0, x_.shape[0], npoints_per_interval)
            
            bins = x_[chunck_idxs]
            bins = np.append(bins, np.nanmax(x_)+1E-4)

        #interpret nintervals
        else:
            if 0 < nintervals and nintervals < 1:
                #calculate nintervals as fraction of the shape of x and y 
                nintervals = int(self.nintervals*x.shape[0])
            elif nintervals >= 1:
                nintervals = int(nintervals)
            else:
                raise ValueError("`nintervals` has to be greater than 0!")


            if self.xmin is None: self.xmin = np.nanmin(x)
            if self.xmax is None: self.xmax = np.nanmax(x)

            bins = np.linspace(self.xmin, self.xmax, nintervals+1)
            bins[-1] += 1E-4

        #assign as attribute
        self.bins = bins


        if self.verbose > 0:
            print(f"INFO(Binning): Generated {len(self.bins)-1} bins")

        return self.bins
    
    def fit(self,
        x:np.ndarray, y:np.ndarray,
        bins:np.ndarray=None,
        ddof:int=None,
        meanfunc_x:Callable=None,
        meanfunc_y:Callable=None,
        verbose:int=None,
        generate_bins_kwargs:dict={},
        ) -> None:
        """
            - method to execute the binning of `y` w.r.t. `x`
            - similar to scikit-learn scalers

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values w.r.t. which the binning shall be executed
                - `y`
                    - np.ndarray
                    - y-values to be binned
                - `bins`
                    - `np.ndarray`, optional
                    - array containing the boundaries of the intervals/bins to use for binning the curve
                    - will overwrite the autogeneration-process
                    - the default is `None`
                - `ddof`
                    - `int`, optional
                    - Delta Degrees of Freedom used in `np.nanstd()`
                    - overwrites `self.ddof` if set
                    - the default is `None`
                        - will fall back to `self.ddof`
                - `meanfunc_x`
                    - `Callable`, optional
                    - function to use to calculate the mean of each interval in `x`
                    - the function shall take one argument
                        - the input dataseries `x`-values
                    - the function shall return a single floating point value
                    - will overwrite `self.meanfunc_x` if passed
                    - the default is `None`
                        - will use `self.meanfunc_x`
                - `meanfunc_y`
                    - `Callable`, optional
                    - function to use to calculate the mean of each interval in `y`
                    - the function shall take one argument
                        - the input dataseries `y`-values
                    - the function shall return a single floating point value
                    - will overwrite `self.meanfunc_y` if passed
                    - the default is `None`
                        - will use `self.meanfunc_y`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overwrites `self.verbosity` if set
                    - the default is `None`
                - `generate_bins_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.generate_bins()`
                    
            Raises
            ------

            Returns
            -------
           
        """

        self.x = x
        self.y = y

        #set/overwrite class attributes
        if ddof is None: ddof = self.ddof
        if verbose is None: verbose = self.verbose
        
        if bins is None:
            bins = self.generate_bins(self.x, self.y, verbose=verbose, **generate_bins_kwargs)
        else:
            self.bins = bins
        if meanfunc_x is None: meanfunc_x = self.meanfunc_x
        if meanfunc_y is None: meanfunc_y = self.meanfunc_y

        #init result arrays
        self.x_binned  = np.array([])
        self.y_binned  = np.array([])
        self.y_std     = np.array([])
        self.n_per_bin = np.array([])    #number of samples per bin

        for b1, b2 in zip(bins[:-1], bins[1:]):

            iv_bool = (b1 <= self.x)&(self.x < b2)

            #adopt transforms
            self.x_binned  = np.append(self.x_binned,  meanfunc_x(self.x[iv_bool]))
            self.y_binned  = np.append(self.y_binned,  meanfunc_y(self.y[iv_bool]))
            self.y_std     = np.append(self.y_std,     np.nanstd(self.y[iv_bool], ddof=ddof))
            self.n_per_bin = np.append(self.n_per_bin, np.count_nonzero(iv_bool))

        return 

    def transform(self,
        x:np.ndarray=None, y:np.ndarray=None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            - method to transform the input-dataseries
            - similar to scikit-learn scalers

            Parameters
            ----------
                - `x`
                    - `np.ndarray`, optional
                    - x-values w.r.t. which the binning shall be executed
                    - only here for consistency, will not be considered in the method
                    - the default is `None`
                - `y`
                    - `np.ndarray`
                    - y-values to be binned  
                    - only here for consistency, will not be considered in the method
                    - the default is `None`
            Raises
            ------

            Returns
            -------
                - `x_binned`
                    - `np.ndarray`
                    - binned values for input `x`
                    - has shape `(1, nintervals)`
                - `y_binned`
                    - `np.ndarray`
                    - binned values for input `y`
                    - has shape `(1, nintervals)`
                - `y_std`
                    - `np.ndarray`
                    - standard deviation of `y` for each interval
                    - characterizes the scattering of the input curve
                    - has shape `(1, nintervals)`
            
            Comments
            --------
        """

        x_binned = self.x_binned
        y_binned = self.y_binned
        y_std = self.y_std

        return x_binned, y_binned, y_std
    
    def fit_transform(self,
        x:np.ndarray, y:np.ndarray,
        fit_kwargs:dict={},
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            - method to fit the transformer and transform the data in one go
            - similar to scikit-learn scalers

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values w.r.t. which the binning shall be executed
                - `y`
                    - `np.ndarray`
                    - y-values to be binned            
                - `fit_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.fit()`

            Raises
            ------

            Returns
            -------
                - `x_binned`
                    - `np.ndarray`
                    - binned values for input `x`
                    - has shape `(1, nintervals)`
                - `y_binned`
                    - `np.ndarray`
                    - binned values for input `y`
                    - has shape `(1, nintervals)`
                - `y_std`
                    - `np.ndarray`
                    - standard deviation of `y` for each interval
                    - characterizes the scattering of the input curve
                    - has shape `(1, nintervals)`

             Comments
             --------            
        """

        self.fit(
            x, y,
            **fit_kwargs,
        )
        x_binned, y_binned, y_std = self.transform()

        return  x_binned, y_binned, y_std

    def plot_result(self,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the result of the data binning

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - created figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------
        """
    
        verbose = self.verbose

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.x, self.y, label="Input", zorder=1, color="C1", alpha=0.7)
        ax1.errorbar(self.x_binned, self.y_binned, yerr=self.y_std, linestyle="", marker="o", label="Binned", zorder=2, color="C0", alpha=1)

        if verbose > 2:
            ax1.vlines(self.bins, ymin=np.nanmin(self.y), ymax=np.nanmax(self.y), color='C3', zorder=3, label='Bin Boundaries')

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        fig.tight_layout()

        axs = fig.axes

        return fig, axs