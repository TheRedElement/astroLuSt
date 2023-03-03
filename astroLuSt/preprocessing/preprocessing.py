
#%%imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed


import warnings

#%%definitions

def fold(
    time:np.ndarray,
    period:float, tref:float=None,
    verbose=0) -> np.ndarray:
    """
        - takes an array of times
            - folds it onto a specified period into phase space
            - returns folded array of phases (in interval 0 to 1)

        Parameters
        ----------
            - time
                - np.array
                - times to be folded with the specified period
            - period 
                - float
                - period to fold the times onto
            - tref
                - float, optional
                - reference time to consider when folding the lightcurve
                - the default is None
                    - will take min(time) as reference
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------

        Returns
        -------
            - phases_folded
                - np.array
                - phases corresponding to the given time folded onto the period
            - periods_folded
                - np.array
                - phases_folded in time domain

        Dependencies
        ------------
            - numpy

        Comments
        --------
    """

    if tref is None:
        tref = time.min()

    delta_t = time-tref
    phases = delta_t/period
    
    #fold phases by getting the remainder of the division by the ones-value 
    #this equals getting the decimal numbers of that specific value
    #+1 because else a division by 0 would occur
    #floor always rounds down a value to the ones (returns everything before decimal point)
    phases_folded = phases-np.floor(phases)

    periods_folded = phases_folded * period

    return phases_folded, periods_folded

def resample(
    x:np.ndarray, y:np.ndarray,
    ndatapoints:int=50,
    verbose:int=0
    ) -> tuple:
    """
        - function to resample a dataseries y(x) to nfeatures new datapoints via interpolation

        Parameters
        ----------
            - x
                - np.ndarray
                - independent input variable x
            - y
                - np.ndarray
                - dependent variable (y(x))
            - ndatapoints
                - int, optional
                - number of datapoints of the resampled dataseries
                - the default is 50
            - verbose
                -  int optional
                - verbosity level
                - the default is 0
            
        Raises
        ------

        Returns
        -------
            - interp_x
                - np.ndarray
                - resamples array of x
            - interp_y
                - np.ndarray
                - resamples array of y

        Dependencies
        ------------
            - numpy
            - matplotlib
        
        Comments
        --------

    """
    interp_x = np.linspace(0, 1, ndatapoints)

    interp_y =  np.interp(interp_x, x, y)

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


class Binning:
    """
        - class to execute data-binning on a given input series
        - essentially calculates a mean representative curve
        - the scatter of the original data and thus certainty of the representation is captured by the standard deviation of 'y' in an interval


        Attributes
        ----------
            - nintervals
                - int, optional
                - nuber of intervals/bins to generate
                - the default is 100
            - npoints_per_interval
                - int, optional
                - generate intervals/bins automatically such that each bin contains 'npoints_per_interval' datapoints
                    - the last interval will contain all datapoints until the end of the dataseries
                - if set will overwrite nintervals
                - the default is None
                    - will use 'nintervals' to generate the bins
            - xmin
                - float, optional
                - the minimum value to consider for the interval/bin creation
                - the default is None
                    - will use the minimum of the input-series x-values
            - xmax
                - float, optional
                - the maximum value to consider for the interval/bin creation
                - the default is None
                    - will use the maximum of the input-series x-values
            - ddof
                - int, optional
                - Delta Degrees of Freedom used in np.nanstd()
                - the default is 0
            - verbose
                - int, optional
                - verbosity level
        
        Derived Attributes
        ------------------
            - generated_bins
                - np.ndarray
                - boundaries of the generated intervals/bin

        Methods
        -------
            - generate_bins()
            - bin_curve()
            - plot_result()

        Dependencies
        ------------
            - matplotlib
            - numpy

        Comments
        --------
    """

    def __init__(self,
        nintervals:int=100, npoints_per_interval:int=None,
        xmin:float=None, xmax:float=None,
        ddof:int=0,
        verbose:int=0,     
        ):
    
        self.nintervals = nintervals
        self.npoints_per_interval= npoints_per_interval
        self.xmin= xmin
        self.xmax= xmax
        self.ddof= ddof
        self.verbose= verbose

        pass

    def __repr__(self):

        return (
            f'Binning(\n'
            f'    nintervals={self.nintervals}, npoints_per_interval={self.npoints_per_interval},\n'
            f'    xmin={self.xmin}, xmax={self.xmax},\n'
            f'    ddof={self.ddof},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )

    def generate_bins(self,
        x:np.ndarray, y:np.ndarray,
        nintervals:int=None, npoints_per_interval:int=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to generate the requested bins

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values w.r.t. which the binning shall be executed
                - y
                    - np.ndarray
                    - y-values to be binned
                - nintervals
                    - int, optional
                    - nuber of intervals/bins to generate
                    - overwrites self.nintervals
                    - the default is None
                        - uses self.nintervals
                - npoints_per_interval
                    - int, optional
                    - generate intervals/bins automatically such that each bin contains 'npoints_per_interval' datapoints
                        - the last interval will contain all datapoints until the end of the dataseries
                    - overwrites self.npoints_per_interval
                    - if set will overwrite nintervals
                    - the default is None
                        - will use 'nintervals' to generate the bins
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose if set
                    - the default is None                
            
            Raises
            ------

            Returns
            -------
                - self.bins
                    - np.ndarray
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
            sortidx = np.argsort(x)
            x_ = x[sortidx]
            y_ = y[sortidx]
            chunck_idxs = np.arange(0, x_.shape[0], npoints_per_interval)
            
            bins = x_[chunck_idxs]
            bins = np.append(bins, np.nanmax(x_)+1E-4)

        #interpret nintervals
        else:
            if 0 < nintervals and nintervals <= 1:
                #calculate nintervals as fraction of the shape of x and y 
                nintervals = int(self.nintervals*x.shape[0])
            elif nintervals > 1:
                nintervals = int(nintervals)
            else:
                raise ValueError("'nintervals' has to greater than 0!")


            if self.xmin is None: self.xmin = np.nanmin(x)
            if self.xmax is None: self.xmax = np.nanmax(x)

            bins = np.linspace(self.xmin, self.xmax, nintervals+1)
            bins[-1] += 1E-4

        #assign as attribute
        self.bins = bins


        if self.verbose > 0:
            print(f"INFO: Generated {len(self.bins)} bins")

        return self.bins
    
    def bin_curve(self,
        x:np.ndarray, y:np.ndarray,
        bins:np.ndarray=None,
        ddof:int=None,
        verbose:int=None,
        **generate_bins_kwargs:dict,
        ):
        """
            - method to execute the binning of y w.r.t. x

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values w.r.t. which the binning shall be executed
                - y
                    - np.ndarray
                    - y-values to be binned
                - bins
                    - np.ndarray
                    - array containing the boundaries of the intervals/bins to use for binning the curve
                    - will overwrite the autogeneration-process
                - ddof
                    - int, optional
                    - Delta Degrees of Freedom used in np.nanstd()
                    - overwrites self.ddof if set
                    - the default is None
                - verbose
                    - int, optional
                    - overwrites self.verbosity if set
                    - verbosity level
                - **generate_bins_kwargs
                    - kwargs of generate_bins()
                    
            Raises
            ------

            Returns
            -------
                - x_binned
                    - np.ndarray
                    - binned values for input 'x'
                    - has shape (1, nintervals)
                - y_binned
                    - np.ndarray
                    - binned values for input 'y'
                    - has shape (1, nintervals)
                - y_std
                    - np.ndarray
                    - standard deviation of 'y' for each interval
                    - characterizes the scattering of the input curve
                    - has shape (1, nintervals)            
        """

        #set/overwrite class attributes
        if ddof is None: ddof = self.ddof
        if verbose is None: verbose = self.verbose
        
        if bins is None:
            bins = self.generate_bins(x, y, verbose=verbose, **generate_bins_kwargs)
        else:
            self.bins = bins

        #init result arrays
        x_binned = np.array([])
        y_binned = np.array([])
        y_std = np.array([])
        self.n_per_bin = np.array([])    #number of samples per bin

        for b1, b2 in zip(bins[:-1], bins[1:]):

            iv_bool = (b1 <= x)&(x < b2)

            x_binned       = np.append(x_binned,       np.nanmean(x[iv_bool]))
            y_binned       = np.append(y_binned,       np.nanmean(y[iv_bool]))
            y_std          = np.append(y_std,          np.nanstd(y[iv_bool], ddof=ddof))
            self.n_per_bin = np.append(self.n_per_bin, np.count_nonzero(iv_bool))

        return x_binned, y_binned, y_std

    def plot_result(self,
        x, y,
        x_binned, y_binned, y_std,
        ):
        """
            - function to plot the result of the binning in phase

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values w.r.t. which the binning shall be executed
                - y
                    - np.ndarray
                    - y-values to be binned
                - x_binned
                    - np.ndarray
                    - binned values for input 'x'
                    - has shape (1, nintervals)
                    - output of bin_curve()
                - y_binned
                    - np.ndarray
                    - binned values for input 'y'
                    - has shape (1, nintervals)
                    - output of bin_curve()
                - y_std
                    - np.ndarray
                    - standard deviation of 'y' for each interval
                    - characterizes the scattering of the input curve
                    - has shape (1, nintervals)                        
                    - output of bin_curve()

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure|None
                    - figure created if verbosity level specified accordingly
                - axs
                    - matplotlib axes|None
                    - axes corresponding to 'fig'              

            Comments
            --------
        """
    
        verbose = self.verbose

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y, label="Input", zorder=1, color="tab:blue", alpha=0.7)
        ax1.errorbar(x_binned, y_binned, yerr=y_std, linestyle="", marker=".", label="Binned", zorder=2, color="tab:orange", alpha=1)

        if verbose > 2:
            ax1.vlines(self.bins, ymin=np.nanmin(y), ymax=np.nanmax(y), color='tab:grey', zorder=3, label='Bin Boundaries')

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        fig.tight_layout()
        plt.show()

        axs = fig.axes

        return fig, axs

class SigmaClipping:
    """
        - class to execute sigma-clipping on x and y
        - creates a mask retaining only values that lie outside an interval of +/- sigma*std_y around a mean curve

        Attributes
        ----------
            - x
                - np.ndarray
                - x-values of the dataseries to clip
                - same shape as y
            - y
                - np.ndarray
                - y-values of the dataseries to clip
                - same shape as x
            - mean_x
                - np.ndarray, optional
                - x-values of a representative mean curve
                - does not have to have the same shape as x
                - same shape as mean_y and std_y
                - if 'None' will infer a mean curve via data-binning
                - the default is 'None'
            - mean_y
                - np.ndarray, optional
                - y-values of a representative mean curve
                - does not have to have the same shape as y
                - same shape as mean_x and std_y
                - if 'None' will infer a mean curve via data-binning
                - the default is 'None'
            - std_y
                - np.ndarray|Nonw, optional
                - standard deviation/errorbars of the representative mean curve in y-direction
                - does not have to have the same shape as y
                - same shape as mean_x and mean_y
                - if 'None' will infer a mean curve via data-binning
                - the default is 'None'
            - sigma_bottom
                - float, optional
                - multiplier for the bottom boundary
                - i.e. bottom boundary = mean_y - sigma_bottom*std_y
                - the default is 2
                    - i.e. 2*sigma
            - sigma_top
                - float, optional
                - multiplier for the top boundary
                - i.e. top boundary = mean_y + sigma_top*std_y
                - the default is 2
                    - i.e. 2*sigma
            - verbose
                - int, optional
                - verbosity level
            - **binning_kwargs
                - kwargs for the Binning class
                - used to generate mean curves if none are provided

        Derived Attributes
        ------------------
            - clip_mask
                        - np.ndarray
                        - mask for the retained values
                        - 1 for every value that got retained
                        - 0 for every value that got cut     
            - lower_bound
                - np.ndarray
                - traces out the lower bound to be considered for the sigma-clipping
            - upper_bound
                - np.ndarray
                - traces out the upper bound to be considered for the sigma-clipping
            - sort_array
                - np.ndarray
                - indices to sort self.x in ascending order
                - only needed for plotting y_mean_interp, upper_bound, lower_bound
            - y_mean_interp
                - np.array
                - traces out the interpolated mean representative curve (resulting from binning)
            - y_std_interp
                - np.array
                - traces out the interpolated standard deviation of the mean representative curve

        Methods
        -------
            - get_mean_curve()
            - clip_curve
            - plot_result()

        Dependencies
        ------------
            - matplotlib
            - numpy

        Comments
        --------
    """


    def __init__(self,
        x:np.ndarray, y:np.ndarray,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,                 
        sigma_bottom:float=2, sigma_top:float=2,
        verbose:int=0,
        **binning_kwargs:dict,
        ) -> None:

        self.x = x
        self.y = y

        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_y  = std_y
        
        self.sigma_bottom = sigma_bottom
        self.sigma_top = sigma_top

        if binning_kwargs == {}:
            self.binning_kwargs = {'nintervals':0.1}
        else:
            self.binning_kwargs = binning_kwargs


        self.verbose = verbose

        pass
    
    def __repr__(self) -> str:

        return (
        f'SigmaClipping(\n'
        f'    x={self.x}, y={self.y},\n'
        f'    mean_x:={self.mean_x}, mean_y={self.mean_y}, std_y={self.std_y},\n'
        f'    sigma_bottom={self.sigma_bottom}, sigma_top={self.sigma_top},\n'
        f'    verbose={self.verbose},\n'
        f'    **{self.binning_kwargs},\n'
        f')'
        )
    
    def get_mean_curve(self,
        verbose:int=None,
        ) -> None:
        """
            - method to adopt the mean curves if provided and generate some if not

            Parameters
            ----------
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        if verbose is None:
            verbose = self.verbose

        #calculate mean curve if insufficient information is provided
        if self.mean_x is None or self.mean_y is None or self.std_y is None:
            
            if verbose > 0:
                print(
                    f"INFO: Calculating mean-curve because one of 'mean_x', 'mean_y', std_y' is None!"
                )
            
            binning = Binning(
                verbose=verbose-1,
                **self.binning_kwargs
            )

            self.mean_x, self.mean_y, self.std_y = binning.bin_curve(self.x, self.y)
        else:
            assert (self.mean_x.shape == self.mean_y.shape) and (self.mean_y.shape == self.std_y.shape), f"shapes of 'mean_x', 'mean_y' and 'std_y' have to be equal but are {self.mean_x.shape}, {self.mean_y.shape}, {self.std_y.shape}"
        
        return 

    def clip_curve(self,
        sigma_bottom:float=None, sigma_top:float=None,
        verbose:int=None,
        ):
        """
            - method to actually execute sigma-clipping on x and y
            - creates a mask retaining only values that lie outside an interval of +/- sigma*std_y around a mean curve

            Parameters
            ----------
                - sigma_bottom
                    - float, optional
                    - multiplier for the bottom boundary
                    - i.e. bottom boundary = mean_y - sigma_bottom*std_y
                    - if set will completely overwrite the existing attribute self.sigma_bottom
                        - i.e. self.sigma_bottom will be set as sigma_bottom
                    - the default is 2
                        - i.e. 2*sigma
                - sigma_top
                    - float, optional
                    - multiplier for the top boundary
                    - i.e. top boundary = mean_y + sigma_top*std_y
                    - if set will completely overwrite the existing attribute self.sigma_top
                        - i.e. self.sigma_top will be set as sigma_top
                    - the default is 2
                        - i.e. 2*sigma
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None                        

            Raises
            ------
        
            Returns
            -------
                - clip_mask
                    - np.ndarray
                    - mask for the retained values
                    - 1 for every value that got retained
                    - 0 for every value that got cut            

            Dependencies
            ------------

            Comments
            --------
        """

        #overwrite original attributes if specified
        if sigma_bottom is None:
            sigma_bottom = self.sigma_bottom
        else:
            self.sigma_bottom = sigma_bottom
        if sigma_top is None:
            sigma_top = self.sigma_top
        else:
            self.sigma_top = sigma_top


        #catching errors
        assert self.x.shape == self.y.shape, f"shapes of 'x' and 'y' have to be equal but are {self.x.shape}, {self.y.shape}"

        self.get_mean_curve(verbose=verbose)

        #sorting-array
        self.sort_array = np.argsort(self.x)

        #get mean curve including error
        self.y_mean_interp = np.interp(self.x, self.mean_x, self.mean_y)
        self.y_std_interp  = np.interp(self.x, self.mean_x, self.std_y)

        #mask of what to retain
        self.lower_bound = self.y_mean_interp-sigma_bottom*self.y_std_interp 
        self.upper_bound = self.y_mean_interp+sigma_top*self.y_std_interp
        clip_mask = (self.lower_bound<self.y)&(self.y<self.upper_bound)

        self.clip_mask = clip_mask

        return clip_mask
    
    def plot_result(self):
        """
            - method to create a plot visualizing the sigma-clipping result

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure|None
                    - figure created if verbosity level specified accordingly
                - axs
                    - matplotlib axes|None
                    - axes corresponding to 'fig'

            Comments
            --------

        """
        ret_color = "tab:blue"
        cut_color = "tab:grey"
        used_bins_color = "tab:orange"
        mean_curve_color = "tab:green"
        ulb_color="k"

        ulb_lab = r"$\bar{y}~\{+%g,-%g\}\sigma$"%(self.sigma_top, self.sigma_bottom)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.x[~self.clip_mask], self.y[~self.clip_mask],             color=cut_color,                                 alpha=0.7, zorder=1, label="Clipped")
        ax1.scatter(self.x[self.clip_mask],  self.y[self.clip_mask],              color=ret_color,                                 alpha=1.0, zorder=2, label="Retained")
        ax1.errorbar(self.mean_x,            self.mean_y, yerr=self.std_y,        color=used_bins_color, linestyle="", marker=".",            zorder=3, label="Used Bins")
        ax1.plot(self.x[self.sort_array],    self.y_mean_interp[self.sort_array], color=mean_curve_color,                                     zorder=4, label="Mean Curve")
        ax1.plot(self.x[self.sort_array],    self.upper_bound[self.sort_array],   color=ulb_color,       linestyle="--",                      zorder=5, label=ulb_lab)
        ax1.plot(self.x[self.sort_array],    self.lower_bound[self.sort_array],   color=ulb_color,       linestyle="--",                      zorder=5) #,label=ulb_lab)

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax1.legend()
        plt.show()

        axs = fig.axes

        return fig, axs

class PDM:
    """
        - Class to execute a Phase Dispersion Minimization on a given dataseries.
        - implementation according to Stellingwerf 1978
            - https://ui.adsabs.harvard.edu/abs/1978ApJ...224..953S/abstract

        Attributes
        ----------
            - period_start
                - float, optional
                - the period to consider as starting point for the analysis
                - the default is 1
            - period_stop
                - float, optional
                - the period to consider as stopping point for the analysis
                - the default is 100
            - nperiods
                - int, optional
                - how many trial periods to consider during the analysis
                - the default is 100
            - trial_periods
                - np.ndarray, optional
                - if passed will use the values in that array and ignore
                    - period_start
                    - period_stop
                    - nperiods
            - nintervals
                - float, optional
                - number of intervals to use when executing data-binning (to estimate the mean curve)
                - if between 0 and 1
                    - interpreted as a fraction of the shapes of 'x' and 'y'
                - if > 1
                    - interpreted as the actual number of bins to use
                    - if a float is passed, will be rounded to the nearest integer
                - the default is 100
            - n_retries
                - int, optional
                - number of integers the initially found period shall be divided with
                    - will divide initial period by the values contained in range(1, n_retries+1)
                - goal is to mitigate period-multiplicities
                - the first retry is just a higher resolution of the initially found best period
                - the default is 1
            - nperiods_retry
                - int, optional
                - number of periods to resolve the region around the retry-periods with
                - i.e. number of periods to resolve the 'retry_range' with
                - the default is 20
            - retry_range
                - float, optional
                - size of the interval around the retry-period
                - i.e. a value of 0.1 will scan the interval (retry-period*(1-0.1/2), retry-period*(1+0.1/2))
                - the retry range will be resolved with 'nperiods_retry' periods
                - the default is 0.1
            - tolerance_expression
                - str, optional
                - expression to define the tolerance (w.r.t. the best period) up to which theta a new period is considered an improvement
                - will get evaluated to define the tolerance
                    - i.e. eval(f'{best_theta}{tolerance_expression}*{tolerance_decay}**(retry-1)') will be called
                - the default is '*1.1'
                    - results in tolerance = best_period*1.1
            - tolerance_decay
                - float, optional
                - factor specifiying the decay of the tolerance as the amount of retries increases
                - i.e. tolerance_expression will be multiplied by 'tolerance_decay**retry'
                    - retry specifies the n-th retry
                - the default is
                    - 1
                    - i.e. no decay
            - breakloop
                - bool, optional
                - whether to quit retrying once no improvement of the variance is achieved anymore
                - the default is True
            - variance_mode
                - str, optional
                - how to estimate the curve variance
                - options
                    - 'interval'
                        - will use the variance of each bin of the binned, phased dataseries
                    - 'interp'
                        - will use the variance of a hose around the binned, phased curve evaluated at each input point
                        - will smaller variation in the periodogram
            - sort_output_by
                - str, optional
                - w.r.t. which parameter tosort the output arrays
                - options are
                    - 'periods'
                        - sorts thetas and variances w.r.t. trial periods
                    - 'variances'
                        - sorts periods and thetas w.r.t. variances
                    - 'thetas'
                        - sorts periods and variances w.r.t. thetas
            - normalize
                - bool, optional
                - whether to normalize the calculated variances
                - the default is False
            - n_jobs
                - int, optional
                - number of jobs to use in the joblib.Parallel() function
                - the default is -1
                    - will use all available workers
            - verbose
                - int, optional
                - verbosity level
                - the default is 0
        
        Infered Attributes
        ------------------
            - theta_tolerance
                - float
                - tolerance value up to which a period is considered an improvement over previously found best periods
            - best_theta
                - float
                - theta statistics corresponding to the best period
            - trial_periods
                - np.ndarray
                - all tested periods
            - thetas
                - np.ndarray
                - theta statistics corresponding to 'trial_periods'
            - var_norms
                - np.ndarray
                - normalized variances corresponding to 'trial_periods'
            - best_period
                - float
                - predicted best period
                    - period of minimum dispersion
            - errestimate
                - float
                - error estimation of 'best_period'
            - best_theta
                - float
                - theta statistics corresponding to 'best_period'
            - best_var
                - float
                - normalized variance corresponding to 'best_period'
            - best_fold_x
                - np.ndarray
                - folded array of the input (x) onto 'best_period'
            - best_fold_y
                - np.ndarray
                - y values corresponding to 'best_fold_x'

        Methods
        -------
            - generate_period_grid()
            - plot_result()
            - test_one_p()
            - fit()
            - predict()
            - fit_predict()

        Dependencies
        ------------
            - numpy
            - matplotlib
            - joblib

        Comments
        --------

        """ 

    def __init__(self,
        #initial period determination
        period_start:float=0.1, period_stop:float=None, nperiods:int=100,
        trial_periods:np.ndarray=None,
        nintervals:float=100,
        npoints_per_interval:int=None,
        #refining found period
        n_retries:int=1,
        nperiods_retry:int=20,
        retry_range:float=0.1,
        tolerance_expression:str='*1.1',
        tolerance_decay:float=1,
        breakloop:bool=True,
        #how to calculate the variance
        variance_mode:str='interval',
        #output
        sort_output_by:str='periods',
        normalize:bool=False,
        #computation and plotting
        n_jobs:int=-1,
        verbose:int=0                 
        ) -> None:

        
        self.period_start = period_start
        self.period_stop  = period_stop 
        self.nperiods     = nperiods
        self.trial_periods = trial_periods

        self.nintervals = nintervals
        self.npoints_per_interval = npoints_per_interval
        self.n_retries = n_retries
        self.nperiods_retry = nperiods_retry
        self.retry_range = retry_range
        self.tolerance_expression = tolerance_expression
        self.tolerance_decay = tolerance_decay
        self.breakloop = breakloop
        self.variance_mode = variance_mode
        self.sort_output_by = sort_output_by
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.verbose = verbose


        
        pass

    def __repr__(self):

        return (
            f'PDM(\n'
            f'    period_start={self.period_start},\n'
            f'    period_stop={self.period_stop},\n'
            f'    nperiods={self.nperiods},\n'
            f'    trial_periods={self.trial_periods},\n'
            f'    nintervals={self.nintervals},\n'
            f'    n_retries={self.n_retries},\n'
            f'    nperiods_retry={self.nperiods_retry},\n'
            f'    retry_range={self.retry_range},\n'
            f'    tolerance_expression={self.tolerance_expression},\n'
            f'    tolerance_decay={self.tolerance_decay},\n'
            f'    breakloop={self.breakloop},\n'
            f'    variance_mode={self.variance_mode},\n'
            f'    sort_output_by={self.sort_output_by},\n'
            f'    normalize={self.normalize},\n'
            f'    n_jobs={self.n_jobs},\n'
            f'    verbose={self.verbose},\n'
            f')'       
        )

    def generate_period_grid(self,
        x:np.ndarray,
        period_start:float=0.1, period_stop:float=None, nperiods:float=100,
        ):
        """
            - method to automatically generate a period grid w.r.t. x

            Parameters
            ----------
                - x
                    - np.ndarray
                    - input array
                    - x-values of the data-series
                - period_start
                    - float, optional
                    - the period to consider as starting point for the analysis
                    - the default is 1
                - period_stop
                    - float, optional
                    - the period to consider as stopping point for the analysis
                    - the default is 100
                - nperiods
                    - int, optional
                    - how many trial periods to consider during the analysis
                    - the default is 100
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if period_stop is None:
            period_stop = np.nanmax(x)-np.nanmin(x)

        trial_periods = np.linspace(period_start, period_stop, nperiods)

        return trial_periods

    def plot_result(self):
        """
            - method to plot the result of the pdm
            - will produce a plot with 2 panels
                - top panel contains the periodogram
                - bottom panel contains the input-dataseries folded onto the best period
        
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title("PDM-result", fontsize=18)
        ax1.scatter(self.trial_periods, self.thetas, color="tab:blue", s=1, marker=".", zorder=1)
        ax1.axvline(self.best_period, color="tab:orange", linestyle="-", label=r"$P_{\mathrm{PDM}} =$" + f"{self.best_period:.3f}", zorder=2)
        ax1.fill_between([np.nanmin(self.trial_periods), np.nanmax(self.trial_periods)], y1=[self.best_theta]*2, y2=[max(self.theta_tolerance, self.best_theta)]*2, color='tab:grey', alpha=0.2, label='Tolerated as improvement')
        ax1.axhline(self.best_theta, color="tab:orange", linestyle="-", zorder=2)
        ax1.tick_params("both")
        ax1.set_xlabel("Period")
        ax1.set_ylabel(r"$\theta$")
        ax1.legend()
        ax2 = fig.add_subplot(212)
        ax2.set_title("Resulting lightcurve")
        ax2.plot(self.best_fold_x, self.best_fold_y, color="tab:blue", marker=".", linestyle="", label="Folded Input-Dataseries")
        ax2.tick_params("both")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.legend()

        plt.tight_layout()
                
        plt.show()

        axs = fig.axes

        return fig, axs

    def get_theta_for_p(self,
            x:np.ndarray, y:np.ndarray, p:float,
        ):
        """
            - function to get the theta-statistics for one particular period (p) w.r.t. x and y

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - y
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - p
                    - float
                    - test period to calculate the theta-statistics for
            
            Raises
            ------

            Returns
            -------
                - theta
                    - float
                    - theta-statistics for 'p' w.r.t. x and y
                - var_norm
                    - float
                    - normalized variance w.r.t. the mean input curve
                    - calculated via estimating the variance in bins
                    - normalized w.r.t. the number of datapoint per bin


            Comments
            --------
                - Calculation of the variance
                    - bin the curve
                    - calculate the variance as a measure of scatter within each bin
                    - (interpolate the variances to create a variance curve)
                    - (evaluate interpolated result ar each x value of the timeseries)
                    - normalize variance by weighting it w.r.t to the number of datapoints per bin
                - Will determine the best period in 2 steps
                    - find an initial period
                    - try mutiplicities of the initial period and return the period of lowest variance as the best period        
        
        """
        #fold curve on test period
        folded, _ = fold(x, p)

        #calculate mean standard deviation
        mean_x, mean_y, std_y, bins, n_per_bin, fig, axs = \
            bin_curve(
                folded, y,
                nintervals=self.nintervals,
                ddof=1,
                verbose=0
            )
        
        #get variance for each point
        var_y_interp  = np.interp(x, mean_x, std_y**2)
        
        #total variance of mean curve
        if self.variance_mode == 'interp':
            #if interpolated no bins present (1 datapoint per bin) => normalization by dividing through size
            var_norm = np.nansum(var_y_interp)/var_y_interp.size
        elif self.variance_mode == 'interval':
            #normalize w.r.t. number of samples per bin (weight with n_per_bin, divide by total nuber of samples)
            var_norm = np.nansum((n_per_bin-1)*std_y**2)/np.nansum(n_per_bin - 1)
        else:
            raise ValueError('Unrecognized argument for "variance_mode". Currently supported are "interp", "interval"!')

        #calculate theta statistics
        theta = var_norm/self.tot_var

        return theta, var_norm

    def fit(self,
        x:np.ndarray, y:np.ndarray,
        tolerance_expression:str=None,
        tolerance_decay:float=None,
        breakloop:bool=None,
        n_jobs:int=None,
        verbose:int=None
        ):
        """
            - method to fit the pdm-estimator
            - will execute the calculation and assign results as attributes
            - similar to fit-method in scikit-learn

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - y
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - tolerance_expression
                    - str, optional
                    - expression to define the tolerance (w.r.t. the best period) up to which theta a new period is considered an improvement
                    - will get evaluated to define the tolerance
                        - i.e. eval(f'{best_theta}{tolerance_expression}*{tolerance_decay}**(retry-1)') will be called
                    - will overwrite the tolerance_expression attribute
                    - the default is '*1.1'
                        - results in tolerance = best_period*1.1
                - tolerance_decay
                    - float, optional
                    - factor specifiying the decay of the tolerance as the amount of retries increases
                    - will overwrite the tolerance_decay attribute
                    - i.e. tolerance_expression will be multiplied by 'tolerance_decay**retry'
                        - retry specifies the n-th retry
                    - the default is
                        - 1
                        - i.e. no decay
                - breakloop
                    - bool, optional
                    - will overwrite the breakloop attribute
                    - whether to quit retrying once no improvement of the variance is achieved anymore
                    - the default is True
                - n_jobs
                    - int, optional
                    - will overwrite the n_jobs attribute
                    - number of jobs to use in the joblib.Parallel() function
                    - the default is -1
                        - will use all available workers
                - verbose
                    - int, optional
                    - will overwrite the verbose attribute
                    - verbosity level
                    - the default is 0
                                        
            Raises
            ------

            Returns
            -------
                - best_period
                    - float
                    - the period yielding the lowest variance in the whole curve
                - best_var
                    - float
                    - the lowest variance calculated
                    - variance corresponding to 'best_period'
                - periods_sorted
                    - np.ndarray
                    - the periods sorted after the the variance they yielded in the curve
                - vars_sorted
                    - np.ndarray
                    - the variances sorted from low to high
                    - corresponding to periods_sorted
                - best_fold
                    - np.ndarray
                    - the resulting phases of the times folded with best_period
                - errestimate
                    - float
                    - an estiamte of the uncertainty of the result
                    - estimated to be 2* the maximum distance between two trial periods
                        - because the best period is certain to lie within the trial interval but where exactly is not sure


        """

        if tolerance_expression is None: tolerance_expression = self.tolerance_expression
        if tolerance_decay is None:      tolerance_decay      = self.tolerance_decay
        if breakloop is None:            breakloop            = self.breakloop
        if n_jobs is None:               n_jobs               = self.n_jobs
        if verbose is None:              verbose              = self.verbose

        if self.trial_periods is None:
            self.trial_periods = self.generate_period_grid(x, self.period_start, self.period_stop, self.nperiods)

        #calculate total variance of curve
        self.tot_var = np.nanvar(y)

        #calculate variance of folded curve for all trial periods
        res = Parallel(n_jobs=self.n_jobs, verbose=verbose)(delayed(self.get_theta_for_p)(
            x, y, p,
            ) for p in self.trial_periods)

        self.thetas    = np.array(res)[:,0]
        self.var_norms = np.array(res)[:,1]

        #retry to figure out if a period multiple was detected
        best_p = self.trial_periods[np.nanargmin(self.thetas)]  #current best period
        best_theta = np.nanmin(self.thetas)                     #theta statistics of the current best period

        # theta_tolerance = eval(f'{best_theta}{tolerance_expression}')   #tolerance for improvement
        for retry in range(1,self.n_retries+1):
            if verbose > 3:
                print(f'INFO: Retrying with best_period/{retry}')

            #trial periods for retry period
            retry_trial_periods = np.linspace((best_p/retry)*(1-self.retry_range/2), (best_p/retry)*(1+self.retry_range/2), self.nperiods_retry)
            retry_trial_periods = retry_trial_periods[(self.period_start<retry_trial_periods)&(retry_trial_periods<self.period_stop)]
            if len(retry_trial_periods) > 0:
                #calculate theta statistics for retry trial priods
                retry_res = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(self.get_theta_for_p)(
                x, y, p,
                ) for p in retry_trial_periods)

                retry_thetas    = np.array(retry_res)[:,0]
                retry_var_norms = np.array(retry_res)[:,1]

                #best theta in retry_range
                retry_best_theta = np.nanmin(retry_thetas)

                #update if improvement is made
                theta_tolerance = eval(f'{best_theta}{tolerance_expression}*{tolerance_decay**(retry-1)}')   #tolerance for improvement
                self.theta_tolerance = 0    #init theta tolerance
                if retry_best_theta < theta_tolerance:
                    self.theta_tolerance = theta_tolerance
                    #update current best theta
                    best_theta = retry_best_theta
                #break if no improvement is made
                else:
                    if self.breakloop:
                        if verbose > 1:
                            print(f'INFO: Broke loop after retry #{retry} because best theta retry ({retry_best_theta:.3f}) > current best theta ({theta_tolerance:.3f})')
                        break
                

                #append results to global results
                self.trial_periods = np.append(self.trial_periods, retry_trial_periods)
                self.var_norms     = np.append(self.var_norms,     retry_var_norms)
                self.thetas        = np.append(self.thetas,        retry_thetas)
            else:
                pass

            #calculated desired parameters
            if   self.sort_output_by   == 'periods':   sortidx = np.argsort(self.trial_periods)
            elif self.sort_output_by   == 'thetas':    sortidx = np.argsort(self.thetas)
            elif self.sort_output_by   == 'variances': sortidx = np.argsort(self.var_norms)
            else: raise ValueError('Unrecognized argument for "sort_output_by". Currently supported are "periods", "thetas", "variances"!')

            self.best_theta     = best_theta
            self.trial_periods  = self.trial_periods[sortidx]
            self.thetas         = self.thetas[sortidx]
            self.var_norms      = self.var_norms[sortidx]
            self.best_period    = self.trial_periods[(self.thetas == self.best_theta)][0]
            self.best_var       = self.var_norms[(self.thetas == self.best_theta)][0]
            self.best_fold_x, _ = fold(x, self.best_period)
            self.best_fold_y    = y

            self.errestimate = np.nanmax(2*np.diff(np.sort(self.trial_periods)))  #error estimate as 2*maximum difference between periods

        return

    def predict(self):
        """
            - method to predict with the fitted pdm-estimator
            - will return relevant results
            - similar to predict-method in scikit-learn

            Returns
            -------
                - best_period
                    - float
                    - best period estimate
                - errestimate
                    - float
                    - error estimation of the best period
                - best_theta
                    - float
                    - theta-statistics of best period
        """
        return self.best_period, self.errestimate, self.best_theta
    
    def fit_predict(self,
        x:np.ndarray, y:np.ndarray,
        **kwargs
        ):
        """
            - method to fit classifier and predict the results

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - y
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - kwargs
                    - keyword arguments of fit()

            Returns
            -------
                - best_period
                    - float
                    - best period estimate
                - errestimate
                    - float
                    - error estimation of the best period
                - best_theta
                    - float
                    - theta-statistics of best period                            
        """
        
        self.fit(x, y, **kwargs)
        best_period, errestimate, best_theta = self.predict()

        return best_period, errestimate, best_theta

def periodic_shift(input_array:np.array, shift:float, borders:list, testplot:bool=False, verbose:int=0):
    #TODO: include timer
    """
        - function to shift an array considering periodic boundaries

        Parameters
        ----------
            - input_array
                - np.array
                - array to be shifted along an interval with periodic boundaries
            - shift
                - float/int
                - magnizude of the shift to apply to the array
            - borders
                - list/np.array
                - upper and lower boundary of the periodic interval
            - testplot
                - bool, optional
                - wether to show a testplot
                - the default is False
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------
            - TypeError
                - if the provided parameters are of the wrong type

        Returns
        -------
            - shifted
                - np.array
                - array shifted by shift along the periodic interval in borders

        Dependencies
        ------------
            - numpy
            - matplotlib

        Comments
        --------
    """        
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #time execution



    ################################
    #check if all types are correct#
    ################################
    
    if type(input_array) != np.ndarray:
        raise TypeError("input_array has to be of type np.ndarray! If you want to shift a scalar, simply convert it to an array and acess outputarray[0]")
    if (type(borders) != np.ndarray) and (type(borders) != list):
        raise TypeError("borders has to be of type np.array or list!")
    if (type(timeit) != bool):
        raise TypeError("timeit has to be of type bool")
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

    #time execution
    if timeit:
        task.end_task()
    
    return shifted


#%%testing

# lc = pd.read_csv('../../data/eleanor_extracted/slurm/tic65992.csv', sep=';')
# lc = pd.read_csv('../../data/eleanor_extracted/slurm/tic1533189.csv', sep=';')
# lc = pd.read_csv('../../data/eleanor_extracted/slurm/tic57466.csv', sep=';')

# x = lc['time']
# y = lc['corr_flux']
# for s in np.unique(lc['sector']):
#     y[lc['sector']==s] /= np.nanmedian(y[lc['sector']==s])

p = 0.5
x = np.linspace(0,20,1000)
x += np.random.normal(size=x.shape)*0.05
y = np.sin(x*2*np.pi/0.5)  + np.random.normal(size=x.shape)*0.05

resample_idxs = np.random.choice(np.arange(0, x.shape[0], 1), size=257, replace=False)

x = x[resample_idxs]
y = y[resample_idxs]

pdm = PDM(
    period_start=0.1, period_stop=1.4, nperiods=100,
    # trial_periods=np.array([0.5, 1, 0.333]),
    nintervals=30,
    n_retries=5,
    tolerance_expression='*1.01',
    tolerance_decay=0.99,
    nperiods_retry=50,
    breakloop=False,
    n_jobs=1,
    verbose=3
)

# print(pdm)

pdm.fit_predict(x, y)
# # pdm.fit(x, y)

# fig, axs = pdm.plot_result()

#%%
sigclip = SigmaClipping(
    pdm.best_fold_x, pdm.best_fold_y,
    sigma_top=5, sigma_bottom=5,
    # **{'nintervals':0.1}
)
sigclip.clip_curve(sigma_top=1)
sigclip.plot_result()

# bins = np.linspace(0,0.4,6)
binning = Binning(nintervals=10, npoints_per_interval=50, verbose=3)
# print(binning)
# binning.generate_bins(pdm.best_fold_x, pdm.best_fold_y)
# x_binned, y_binned, y_std = binning.bin_curve(pdm.best_fold_x, pdm.best_fold_y, **{'nintervals':None, 'npoints_per_interval':None})
# binning.plot_result(pdm.best_fold_x, pdm.best_fold_y, x_binned, y_binned, y_std)

for a in dir(binning):
    print(a)
