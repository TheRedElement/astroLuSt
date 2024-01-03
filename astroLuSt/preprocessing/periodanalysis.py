
#%%imports
from astropy.timeseries import LombScargle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Union, Tuple, Callable
import warnings

from astroLuSt.preprocessing.binning import Binning
from astroLuSt.preprocessing.dataseries_manipulation import fold


#%%definitions
class PDM:
    """
        - Class to execute a Phase Dispersion Minimization on a given dataseries.
        - implementation according to Stellingwerf 1978
            - https://ui.adsabs.harvard.edu/abs/1978ApJ...224..953S/abstract

        Attributes
        ----------
            - `period_start`
                - float, optional
                - the period to consider as starting point for the analysis
                - the default is `None`
                    - will try to consider timeseries and estimate nyquist frequency from that
                    - if that fails defaults to 1
            - `period_stop`
                - float, optional
                - the period to consider as stopping point for the analysis
                - if `n_retries` is > 0 will be increased by `n_retries*nperiods_retry`
                - the default is `None`
                    - will try to consider the length of the timeseries to analyze (i.e. maximum determinable period = length of dataseries)
                    - if that fails will be set to 100
            - `nperiods`
                - int, optional
                - how many trial periods to consider during the analysis
                - the default is 100
            - `n_nyq`
                - float, optional
                - nyquist factor
                - the average nyquist frequency corresponding to `x` will be multiplied by this value to get the minimum period
                - the default is `None`
                    - will default to 1
            - `n0`
                - int, optional
                - oversampling factor
                - i.e. number of datapoints to use on each peak in the periodogram
                - the default is `None`
                    - will default to 5
            - `trial_periods`
                - np.ndarray, optional
                - if passed will use the values in that array and ignore
                    - `period_start`
                    - `period_stop`
                    - `nperiods`
            - `n_retries`
                - int, optional
                - number of integers the initially found period shall be divided with
                    - will divide initial period by the values contained in `range(1, n_retries+1)`
                - goal is to mitigate period-multiplicities
                - the first retry is just a higher resolution of the initially found best period
                - the default is 1
            - `nperiods_retry`
                - int, optional
                - number of periods to resolve the region around the retry-periods with
                - i.e. number of periods to resolve the `retry_range` with
                - the default is 20
            - `retry_range`
                - float, optional
                - size of the interval around the retry-period
                - i.e. a value of 0.1 will scan the interval `(retry-period*(1-0.1/2), retry-period*(1+0.1/2))`
                - the retry range will be resolved with `nperiods_retry` periods
                - the default is 0.1
            - `tolerance_expression`
                - str, optional
                - expression to define the tolerance (w.r.t. the best period) up to which `theta` a new period is considered an improvement
                - will get evaluated to define the tolerance
                    - i.e. `eval(f'{best_theta}{tolerance_expression}*{tolerance_decay}**(retry-1)')` will be called
                - the default is `'*1.1'`
                    - results in `tolerance = best_period*1.1`
            - `tolerance_decay`
                - float, optional
                - factor specifiying the decay of the tolerance as the amount of retries increases
                - i.e. tolerance_expression will be multiplied by `tolerance_decay**retry`
                    - `retry` specifies the n-th retry
                - the default is
                    - 1
                    - i.e. no decay
            - `breakloop`
                - bool, optional
                - whether to quit retrying once no improvement of the variance is achieved anymore
                - the default is `True`
            - `variance_mode`
                - str, optional
                - how to estimate the curve variance
                - options
                    - `'interval'`
                        - will use the variance of each bin of the binned, phased dataseries
                    - `'interp'`
                        - will use the variance of a hose around the binned, phased curve evaluated at each input point
                        - will smaller variation in the periodogram
            - `sort_output_by`
                - str, optional
                - w.r.t. which parameter to sort the output arrays
                - options are
                    - `'periods'`
                        - sorts thetas and variances w.r.t. trial periods
                    - `'variances'`
                        - sorts periods and thetas w.r.t. variances
                    - `'thetas'`
                        - sorts periods and variances w.r.t. thetas
            - `normalize`
                - bool, optional
                - whether to normalize the calculated variances
                - the default is `False`
            - `n_jobs`
                - int, optional
                - number of jobs to use in the `joblib.Parallel()` function
                - the default is -1
                    - will use all available workers
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0
            - `binning_kwargs`
                - dict, optional
                - kwargs for the Binning class
                - used to bin the folded curves and estimate the variance w.r.t. a mean curve
                - the default is `None`
                    - will be set to `dict()`
                - `parallel_kwargs`
                    - `dict`, optional
                    - additional kwargs to pass to `joblib.parallel.Parallel()`
                    - the default is `None`
                        - will be set to `dict(backend='threading')`                    
        
        Infered Attributes
        ------------------
            - `theta_tolerance`
                - float
                - tolerance value up to which a period is considered an improvement over previously found best periods
            - `best_theta`
                - float
                - theta statistics corresponding to the best period
            - `trial_periods`
                - np.ndarray
                - all tested periods
            - `thetas`
                - np.ndarray
                - theta statistics corresponding to `trial_periods`
            - `var_norms`
                - np.ndarray
                - normalized variances corresponding to `trial_periods`
            - `best_period`
                - float
                - predicted best period
                    - period of minimum dispersion
            - `errestimate`
                - float
                - error estimation of `best_period`
            - `best_theta`
                - float
                - theta statistics corresponding to `best_period`
            - `best_var`
                - float
                - normalized variance corresponding to `best_period`
            - `best_fold_x`
                - np.ndarray
                - folded array of the input (`x`) onto `best_period`
            - `best_fold_y`
                - np.ndarray
                - `y` values corresponding to `best_fold_x`

        Methods
        -------
            - `generate_period_grid()`
            - `plot_result()`
            - `test_one_p()`
            - `fit()`
            - `predict()`
            - `fit_predict()`

        Dependencies
        ------------
            - joblib
            - matplotlib
            - numpy
            - typing

        Comments
        --------

        """ 

    def __init__(self,
        #initial period determination
        period_start:float=None, period_stop:float=None, nperiods:int=None,
        n_nyq:float=None,
        n0:int=None,
        trial_periods:np.ndarray=None,
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
        verbose:int=0,
        binning_kwargs:dict=None,
        parallel_kwargs:dict=None, 
        ) -> None:

        self.period_start   = period_start
        self.period_stop    = period_stop
        self.nperiods       = nperiods
        self.trial_periods  = trial_periods
        self.n_nyq          = n_nyq
        self.n0             = n0



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
        if binning_kwargs is None:  self.binning_kwargs = dict()
        else:                       self.binning_kwargs = binning_kwargs
        if parallel_kwargs is None: self.parallel_kwargs= dict(backend='threading')
        else:                       self.parallel_kwargs= parallel_kwargs

        #adopt period_start and period_stop if trial_periods were passed
        if self.trial_periods is not None:
            self.period_start = np.nanmin(self.trial_periods)
            self.period_stop  = np.nanmax(self.trial_periods)
        
        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    period_start={repr(self.period_start)}, period_stop={repr(self.period_stop)}, nperiods={repr(self.nperiods)},\n'
            f'    n_nyq={repr(self.n_nyq)},\n'
            f'    n0={repr(self.n0)},\n'
            f'    trial_periods={repr(self.trial_periods)},\n'
            f'    npoints_per_interval={repr(self.npoints_per_interval)},\n'
            f'    n_retries={repr(self.n_retries)},\n'
            f'    nperiods_retry={repr(self.nperiods_retry)},\n'
            f'    retry_range={repr(self.retry_range)},\n'
            f'    tolerance_expression={repr(self.tolerance_expression)},\n'
            f'    tolerance_decay={repr(self.tolerance_decay)},\n'
            f'    breakloop={repr(self.breakloop)},\n'
            f'    variance_mode={repr(self.variance_mode)},\n'
            f'    sort_output_by={repr(self.sort_output_by)},\n'
            f'    normalize={repr(self.normalize)},\n'
            f'    n_jobs={repr(self.n_jobs)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f'    binning_kwargs={repr(self.binning_kwargs)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def generate_period_grid(self,
        period_start:float=None, period_stop:float=None, nperiods:float=None,
        x:np.ndarray=None,
        n_nyq:int=None,
        n0:int=None,
        ) -> np.ndarray:
        """
            - method to generate a period grid
            - inspired by `astropy.timeseries.LombScargle().autofrequency()` and VanderPlas (2018)
                - https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html
                - https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract

            Parameters
            ----------
                - `period_start`
                    - float, optional
                    - the period to consider as starting point for the analysis
                    - the default is `None`
                        - will default to `self.period_start`
                - `period_stop`
                    - float, optional
                    - the period to consider as stopping point for the analysis
                    - the default is `None`
                        - will default to 100 if `x` is also `None`
                        - otherwise will consider `x` to generate `period_stop`
                - `nperiods`
                    - int, optional
                    - how many trial periods to consider during the analysis
                    - the default is `None`
                        - will default to `self.nperiods`
                - `x`
                    - np.ndarray, optional
                    - input array
                    - x-values of the data-series
                    - the default is `None`
                        - if set and `period_stop` is `None`, will use `max(x)-min(x)` as `period_stop`
                - `n_nyq`
                    - float, optional
                    - nyquist factor
                    - the average nyquist frequency corresponding to `x` will be multiplied by this value to get the minimum period
                    - the default is `None`
                        - will default to `self.n_nyq`
                        - if `self.n_nyq` is also `None` will default to 1
                - `n0`
                    - int, optional
                    - oversampling factor
                    - i.e. number of datapoints to use on each peak in the periodogram
                    - the default is `None`
                        - will default to `self.n0`
                        - if `self.n0` is also `None` will default to 5
            
            Raises
            ------

            Returns
            -------
                - `trial_periods`
                    - np.ndarray
                    - final trial periods used for the execution of `PDM`

            Comments
            --------

        """

        if n_nyq is None:
            if self.n_nyq is not None:
                n_nyq = self.n_nyq
            else:
                n_nyq = 1
        if n0 is None:
            if self.n0 is not None:
                n0 = self.n0
            else:
                n0 = 5

        
        #overwrite defaults if requested
        if period_start is None:
            if x is not None:
                #get average nyquist frequency
                nyq_bar = 0.5*(x.size / (np.nanmax(x) - np.nanmin(x)))
                #convert to nyquist period
                period_start = 1/(n_nyq*nyq_bar)
            else:
                period_start = 1
        if period_stop is None:
            if x is not None:
                #maximum determinable period (signal has to be observed at least once)
                period_stop = (np.nanmax(x) - np.nanmin(x))
            else:
                period_stop = 100
        if nperiods is None:
            if self.nperiods is not None:
                nperiods = self.nperiods
            else:
                # nperiods = int(n0*(np.nanmax(x)-np.nanmin(x))*period_stop)
                nperiods = int(n0*(np.nanmax(x)-np.nanmin(x))*1/period_start)

        trial_periods = np.linspace(period_start, period_stop, nperiods)

        #update period_start, period_stop and trial_periods
        self.period_start = period_start
        self.period_stop = period_stop
        self.trial_periods = trial_periods

        if self.verbose > 2:
            print(f'INFO(PDM): generated grid:')
            print(f'    start        = {period_start}')
            print(f'    stop         = {period_stop}')
            print(f'    trial points = {trial_periods.shape}')

        return trial_periods

    def get_theta_for_p(self,
        x:np.ndarray, y:np.ndarray, p:float,
        ) -> Tuple[float,float]:
        """
            - function to get the theta-statistics for one particular period (`p`) w.r.t. `x` and `y`

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - `p`
                    - float
                    - test period to calculate the theta-statistics for
            
            Raises
            ------
                - `ValueError`
                    - if wrong `self.variance_mode` has an invalid value

            Returns
            -------
                - `theta`
                    - float
                    - theta-statistics for `p` w.r.t. `x` and `y`
                - `var_norm`
                    - float
                    - normalized variance w.r.t. the mean input curve
                    - calculated via estimating the variance in bins
                    - normalized w.r.t. the number of datapoints per bin


            Comments
            --------
                - Calculation of the variance
                    - bin the curve
                    - calculate the variance as a measure of scatter within each bin
                    - (interpolate the variances to create a variance curve)
                    - (evaluate interpolated result ar each x value of the timeseries)
                    - normalize variance by weighting it w.r.t to the number of datapoints per bin      
        
        """
        #fold curve on test period
        folded, _ = fold(x, p)

        #calculate mean standard deviation
        binning = Binning(
            ddof=1,
            verbose=0,
            **self.binning_kwargs
        )
        mean_x, mean_y, std_y = binning.fit_transform(folded, y)
        n_per_bin = binning.n_per_bin
        
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
        verbose:int=None,
        parallel_kwargs:dict=None,
        ) -> None:
        """
            - method to fit the `PDM`-estimator
            - will execute the calculation and assign results as attributes
            - similar to fit-method in scikit-learn

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - `tolerance_expression`
                    - str, optional
                    - expression to define the tolerance (w.r.t. the best period) up to which theta a new period is considered an improvement
                    - will get evaluated to define the tolerance
                        - i.e. `eval(f'{best_theta}{tolerance_expression}*{tolerance_decay}**(retry-1)')` will be called
                    - will overwrite the `self.tolerance_expression`
                    - the default is `'*1.1'`
                        - results in `tolerance = best_period*1.1`
                - `tolerance_decay`
                    - float, optional
                    - factor specifiying the decay of the tolerance as the amount of retries increases
                    - will overwrite the `self.tolerance_decay`
                    - i.e. `tolerance_expression` will be multiplied by `tolerance_decay**retry`
                        - retry specifies the n-th retry
                    - the default is
                        - 1
                        - i.e. no decay
                - `breakloop`
                    - bool, optional
                    - will overwrite the `self.breakloop`
                    - whether to quit retrying once no improvement of the variance is achieved anymore
                    - the default is `True`
                - `n_jobs`
                    - int, optional
                    - will overwrite the `n_jobs` attribute
                    - number of jobs to use in the `joblib.Parallel()` function
                    - the default is -1
                        - will use all available workers
                - `verbose`
                    - int, optional
                    - will overwrite the `self.verbose`
                    - verbosity level
                    - the default is 0
                - `parallel_kwargs`
                    - `dict`, optional
                    - additional kwargs to pass to `joblib.parallel.Parallel()`
                    - overrides `self.parallel_kwargs`
                    - the default is `None`
                        - will fall back to `self.parallel_kwargs`
                                        
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - Will determine the best period in 2 steps
                    - find an initial period
                    - try mutiplicities of the initial period and return the period of lowest variance as the best period            

        """

        if tolerance_expression is None:tolerance_expression = self.tolerance_expression
        if tolerance_decay is None:     tolerance_decay      = self.tolerance_decay
        if breakloop is None:           breakloop            = self.breakloop
        if n_jobs is None:              n_jobs               = self.n_jobs
        if verbose is None:             verbose              = self.verbose
        if parallel_kwargs is None:     parallel_kwargs      = self.parallel_kwargs

        if self.trial_periods is None:
            self.trial_periods = self.generate_period_grid(self.period_start, self.period_stop, self.nperiods, x=x)

        #calculate total variance of curve
        self.tot_var = np.nanvar(y)

        #calculate variance of folded curve for all trial periods
        res = Parallel(n_jobs=self.n_jobs, verbose=verbose, **parallel_kwargs)(delayed(self.get_theta_for_p)(
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

        if self.n_retries < 1:
            #set theta_tolerance to 0 if no retry-iteration was executed
            self.theta_tolerance = 0
            
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

    def predict(self,
        x:np.ndarray=None, y:np.ndarray=None, 
        ) -> Tuple[float, float, float]:
        """
            - method to predict with the fitted `PDM`-estimator
            - will return relevant results
            - similar to predict-method in scikit-learn

            Parameters
            ----------
                - `x`
                    - np.ndarray, optional
                    - x values of the dataseries to run `PDM` on
                    - only here for consistency, will not be considered in the method
                    - the default is `None`
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run `PDM` on
                    - only here for consistency, will not be considered in the method
                    - the default is `None`
            
            Raises
            ------

            Returns
            -------
                - `best_period`
                    - float
                    - best period estimate
                - `errestimate`
                    - float
                    - error estimation of the best period
                - `best_theta`
                    - float
                    - theta-statistics of best period

            Comments
            --------

        """
        return self.best_period, self.errestimate, self.best_theta
    
    def fit_predict(self,
        x:np.ndarray, y:np.ndarray,
        fit_kwargs:dict=None,
        ) -> Tuple[float,float,float]:
        """
            - method to fit classifier and predict the results

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - `fit_kwargs`
                    - keyword arguments passed to `fit()`
                    - the default is `None`
                        - will be set to `{}`

            Raises
            ------
                    
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
            
            Comments
            --------

        """
        
        if fit_kwargs is None:
            fit_kwargs = {}

        self.fit(x, y, **fit_kwargs)
        best_period, errestimate, best_theta = self.predict()

        return best_period, errestimate, best_theta

    def plot_result(self,
        x:np.ndarray=None, y:np.ndarray=None,
        fig_kwargs:dict=None,
        sctr_kwargs:dict=None,                    
        ) -> Tuple[Figure, plt.Axes]:
        """
            - method to plot the result of the pdm
            - will produce a plot with 2 panels
                - top panel contains the periodogram
                - bottom panel contains the input-dataseries folded onto the best period

            Parameters
            ----------
                - `x`
                    - np.ndarray, optional
                    - x-values of a dataseries to plot folded with the determined period
                    - usually the dataseries the analysis was done one
                    - the default is `None`
                        - will not plot a dataseries
                - `y`
                    - np.ndarray, optional
                    - y-values of a dataseries to plot folded with the determined period
                    - usually the dataseries the analysis was done on
                    - the default is `None`
                        - will not plot a dataseries            
                - `fig_kwargs`
                    - dict, optional
                    - kwargs for matplotlib `plt.figure()` method
                    - the default is `None`
                        - will initialize with an empty dict (`{}`)
                - `sctr_kwargs`
                    - dict, optional
                    - kwargs for matplotlib `ax.scatter()` method used to plot `theta(period)`
                    - the default is `None`
                        - will initialize with an empty dict (`{}`)

            Raises
            ------

            Returns
            -------
                - `fig`
                    - matplotlib Figure
                    - figure created if verbosity level specified accordingly
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`
                    
            Comments
            --------
        
        """

        if fig_kwargs  is None: fig_kwargs = {}
        if sctr_kwargs is None: sctr_kwargs = {}
        

        fig = plt.figure(**fig_kwargs)
        #check if folded dataseries shall be plotted as well
        if x is not None and y is not None:
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        else:
            ax1 = fig.add_subplot(111)

        ax1.set_title("PDM-result")
        ax1.scatter(self.trial_periods, self.thetas, color='tab:green', s=1, zorder=1, **sctr_kwargs)
        ax1.axvline(self.best_period, color='tab:grey', linestyle="--", label=r'$\mathrm{P_{PDM}}$ = %.3f'%(self.best_period), zorder=2)
        ax1.fill_between([np.nanmin(self.trial_periods), np.nanmax(self.trial_periods)], y1=[self.best_theta]*2, y2=[max(self.theta_tolerance, self.best_theta)]*2, color='tab:grey', alpha=0.2, label='Tolerated as improvement')
        # ax1.axhline(self.best_theta, color="tab:grey", linestyle="--", zorder=2)
        ax1.set_xlabel(r'Period')
        ax1.set_ylabel(r'$\theta$')
        ax1.legend()
        
        #plot folded dataseries if requested
        if x is not None and y is not None:
                        
            ax2.set_title("Folded Input")
            ax2.scatter(fold(x, self.best_period, 0)[0], y, color='tab:blue', s=1, label='Folded Input-Dataseries')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')

        plt.tight_layout()
                

        axs = fig.axes

        return fig, axs


class HPS:
    """
        - HPS = Hybrid Period Search
        - class to execute a period search inspired by Saha et al., 2017
            - https://ui.adsabs.harvard.edu/abs/2017AJ....154..231S/abstract
        
        Attributes
        ----------
            - `period_start`
                - float, optional
                - the period to consider as starting point for the analysis
                - the default is 1
            - `period_stop`
                - float, optional
                - the period to consider as stopping point for the analysis
                - the default is 100
            - `nperiods`
                - int, optional
                - how many trial periods to consider during the analysis
                - the default is 100
            - `trial_periods`
                - np.ndarray, optional
                - if passed will use the values in that array and ignore
                    - `period_start`
                    - `period_stop`
                    - `nperiods`
            - `n_nyq`
                - float, optional
                - nyquist factor
                - the average nyquist frequency corresponding to `x` will be multiplied by this value to get the minimum period
                - the default is `None`
                    - will default to 1
            - `n0`
                - int, optional
                - oversampling factor
                - i.e. number of datapoints to use on each peak in the periodogram
                - the default is `None`
                    - will default to 5                
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0
            - `pdm_kwargs`
                - dict, optional
                - kwargs for the `astroLuSt.periodanalysis.PDM` class
            - `ls_kwargs`
                - dict, optional
                - kwargs for the `astropy.timeseries.LombScargle` class
            - `lsfit_kwargs`
                - dict, optional
                - kwargs for the `autopower()` method of `the astropy.timeseries.LombScargle` class
            

        Infered Attributes
        ------------------
            - `best_period`
                - float
                - best period according to the metric of `HPS`
            - `best_frequency_ls`
                - float
                - best frequency according to Lomb-Scargle
            - `best_period_pdm`
                - float
                - best period according to `PDM`
            - `best_power_ls`
                - float
                - power corresponding to `best_period_ls`
            - `best_psi`
                - float
                - metric corresponding to `best_period`
            - `best_theta_pdm`
                - float
                - best theta statistics corresponding to `best_period_pdm`
            - `errestimate_pdm`
                - float
                - estimate of the error of `best_period_pdm`
            - `powers_ls`
                - np.ndarray
                - powers corresponding to `trial_periods_ls`
            - `powers_hps`
                - np.ndarray
                - powers of the hps alogrithm
                - calculated by
                    - squeezing `powers_ls` into `range(0,1)`
            - `psis_hps`
                - np.ndarray
                - psi corresponding to `trial_periods_hps`
            - `thetas_pdm`
                - np.ndarray
                - thetas corresponding to `trial_periods_pdm`
            - `thetas_hps`
                - np.ndarray
                - thetas of the hps alogrithm
                - calculated by
                    - evaluating `1-thetas_pdm `
                    - squeezing result into `range(0,1)`
            - `trial_frequencies`
                - np.ndarray
                - trial frequencies used for execution of HPS algorithm
                    - relevant in execution of `LombScargle`
                - `trial_frequencies = 1/trial_periods`
            - `trial_periods`
                - np.ndarray
                - final trial periods used for the execution of HPS algorithm
                    - relevant in execution of `PDM`
                - `trial_periods = 1/trial_frequencies`
            - `pdm`
                - instance of `PDM` class
                - contains all information of the pdm fit
            - `ls`
                - instance of `LombScargle` class
                - contains all information of the LombScargle fit
                            
        Methods
        -------
            - `run_pdm()`
            - `run_lombscargle()`
            - `get_psi()`
            - `fit()`
            - `predict()`
            - `fit_predict()`
            - `plot_result()`

        Raises
        ------

        Dependencies
        ------------
            - astropy
            - matplotlib
            - numpy
            - sklearn
            - typing
            - warnings

        Comments
        --------
            - basic runthrough of computation
                - take dataseries
                - compute Lomb-Scargle
                - compute PDM
                - rescale Lomb-Scargle power to `range(0,1)` (:=Pi_hps)
                - invert PDM theta statistics (theta) by evaluating `1-theta`
                - rescale inverted PDM theta statistics to `range(0,1)` (:=theta_hps)
                - calculate new metric as `Psi = Pi_hps * theta_hps`
            - essentially upweights periods where Lomb-Scargle and PDM agree and downweights those where they disagree
                - if at some period Lomb-Scargle has a strong peak and PDM has a strong minimum the inverted PDM minimum will amplify the Lomb-Scargle peak
                - if one has a peak and the other one does not, then the respective peak gets dammed

    """

    def __init__(self,
        period_start:float=None, period_stop:float=None, nperiods:int=None,        
        trial_periods:np.ndarray=None,
        n_nyq:float=None,
        n0:float=None,
        verbose:int=0,
        pdm_kwargs:dict=None, ls_kwargs:dict=None, lsfit_kwargs:dict=None
        ) -> None:

        self.period_start = period_start
        self.period_stop = period_stop
        self.nperiods = nperiods
        self.trial_periods = trial_periods
        self.n_nyq = n_nyq
        self.n0 = n0

        self.verbose = verbose

        if pdm_kwargs is None:
            self.pdm_kwargs = {'n_retries':0, 'n_jobs':1}
        else:
            self.pdm_kwargs = pdm_kwargs
            self.pdm_kwargs['n_retries'] = 0    #set n_retries to 0 because it does not work yet
            self.pdm_kwargs['n_jobs']    = 1    #set n_jobs to 1 because it does not work otherwise
        
        if ls_kwargs is None:
            self.ls_kwargs = {}
        else:
            self.ls_kwargs = ls_kwargs
        
        if lsfit_kwargs is None:
            self.lsfit_kwargs = {}
        else:
            self.lsfit_kwargs = lsfit_kwargs

        if 'n_retries' in self.pdm_kwargs.keys():
            if self.pdm_kwargs['n_retries'] > 0:
                self.pdm_kwargs['n_retries'] = 0
                warnings.warn(f'Currently only n_retries == 0 works properly! Will ignore provided value for "n_retries" ({self.pdm_kwargs["n_retries"]})!')

        pass

    def __repr__(self) -> str:        
        return (
            f'{self.__class__.__name__}(\n'
            f'    period_start={repr(self.period_start)}, period_stop={repr(self.period_stop)}, nperiods={repr(self.nperiods)},\n'
            f'    trial_periods={repr(self.trial_periods)},\n'
            f'    n_nyq={repr(self.n_nyq)},\n'
            f'    n0={repr(self.n0)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f'    pdm_kwargs={repr(self.pdm_kwargs)}, ls_kwargs={repr(self.ls_kwargs)}, lsfit_kwargs={repr(self.lsfit_kwargs)}\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def generate_period_frequency_grid(self,
        period_start:float=None, period_stop:float=None, nperiods:float=None,
        n_nyq:float=None,
        n0:int=None,
        x:np.ndarray=None,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
            - method to generate a grid of test-periods and test-frequencies
            - inspired by `astropy.timeseries.LombScargle().autofrequency()` and VanderPlas (2018)
                - https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html
                - https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract


            Parameters
            ----------
                - `period_start`
                    - float, optional
                    - the period to consider as starting point for the analysis
                    - overrides `self.period_start`
                    - the default is `None`
                        - will default to `self.period_start`
                - `period_stop`
                    - float, optional
                    - the period to consider as stopping point for the analysis
                    - overrides `self.period_stop`
                    - the default is `None`
                        - will default to 100 if `x` is also `None`
                        - otherwise will consider `x` to generate `period_stop`
                - `nperiods`
                    - int, optional
                    - how many trial periods to consider during the analysis
                    - overrides `self.nperiods`
                    - the default is `None`
                        - will default to `self.nperiods`
                - `n_nyq`
                    - float, optional
                    - nyquist factor
                    - the average nyquist frequency corresponding to `x` will be multiplied by this value to get the minimum period
                    - will override `self.n_nyq`
                    - the default is `None`
                        - will default to `self.n_nyq`
                - `n0`
                    - int, optional
                    - oversampling factor
                    - i.e. number of datapoints to use on each peak in the periodogram
                    - overrides `self.n0`
                    - the default is `None`
                        - will default to `self.n0`
                        - if `self.n0` is also `None` will default to 5
                - `x`
                    - np.ndarray, optional
                    - input array
                    - x-values of the data-series
                    - the default is `None`
                        - if set and `period_stop` is `None`, will use `max(x)-min(x)` as `period_stop`
                        
            Raises
            ------

            Returns
            -------
                - `trial_frequencies`
                    - np.ndarray
                    - trial frequencies used for execution of HPS algorithm
                        - relevant in execution of LombScargle
                    - `trial_frequencies = 1/trial_periods`
                - `trial_periods`
                    - np.ndarray
                    - final trial periods used for the execution of HPS algorithm
                        - relevant in execution of PDM
                    - `trial_periods = 1/trial_frequencies`

            Comments
            --------
        """

        #overwrite defaults if requested
        if period_start is None: period_start = self.period_start
        if period_stop is None: period_stop = self.period_stop
        if nperiods is None:
            if self.nperiods is not None:
                nperiods = self.nperiods//2 #divide by 2 because two grids will be generated and combined
            else:
                nperiods = self.nperiods    #if nperiods not provided, infer them based on the dataseries in grid_gen.generate_period_grid()
        else:
            nperiods = nperiods//2      #divide by 2 because two grids will be generated and combined
        if n_nyq is None: n_nyq = self.n_nyq
        if n0 is None: n0 = self.n0

        grid_gen = PDM(verbose=self.verbose)
        trial_periods_pdm    = grid_gen.generate_period_grid(period_start, period_stop, nperiods, x=x, n_nyq=n_nyq, n0=n0)
        trial_frequencies_ls = grid_gen.generate_period_grid(1/trial_periods_pdm.max(), 1/trial_periods_pdm.min(), nperiods=trial_periods_pdm.size, x=None)

        trial_periods     = np.sort(np.append(trial_periods_pdm, 1/trial_frequencies_ls))
        trial_frequencies = np.sort(np.append(1/trial_periods_pdm, trial_frequencies_ls))


        if self.verbose > 2:
            c_p = 'tab:blue'
            c_f = 'tab:orange'

            fig = plt.figure()
            fig.suptitle('Generated test periods and frequencies')
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            ax1.hist(trial_periods,     histtype='bar',  color=c_p, bins='sqrt', linewidth=2, linestyle='-', alpha=0.8)# label='Period')
            ax2.hist(trial_frequencies, histtype='step', color=c_f, bins='sqrt', linewidth=2, linestyle='-', alpha=1.0)# label='Frequency')
            ax1.set_xlabel('Period', color=c_p)
            ax2.set_xlabel('Frequency', color=c_f)
            ax1.set_ylabel('Counts')
            ax1.set_xticks(ax1.get_xticks())
            ax2.set_xticks(ax2.get_xticks())
            ax1.set_xticklabels(ax1.get_xticklabels(), color=c_p)
            ax2.set_xticklabels(ax2.get_xticklabels(), color=c_f)

            ax2.invert_xaxis()

            plt.show()


        # trial_periods = trial_periods_pdm
        # trial_frequencies = trial_frequencies_ls

        return trial_periods, trial_frequencies

    def run_pdm(self,
        x:np.ndarray, y:np.ndarray,
        trial_periods:np.ndarray=None,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
            - method to execute a Phase-Dispersion Minimization
            
            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - `trial_periods`
                    - np.ndarray, optional
                    - trial periods to for folding and dispersion determination
                    - if passed will overwrite `self.trial_periods`
                    - the default is `None`
                        - will use `self.trial_periods`
            
            Raises
            ------

            Returns
            -------
                - `pdm.thetas`
                    - np.ndarray
                    - thetas corresponding to `pdm.trial_periods`
                - `pdm.trial_periods`
                    - np.ndarray
                    - if passed will overwrite `self.trial_periods`
                    - the default is `None`
                        - will use `self.trial_periods`

            Comments
            --------
        """
        
        if trial_periods is None: trial_periods = self.trial_periods

        self.pdm = PDM(
            period_start=self.period_start, period_stop=self.period_stop, nperiods=self.nperiods,
            trial_periods=trial_periods,
            **self.pdm_kwargs
        )

        self.best_period_pdm, self.errestimate_pdm, self.best_theta_pdm = self.pdm.fit_predict(x, y)

        return self.pdm.thetas, self.pdm.trial_periods

    def run_lombscargle(self,
        x:np.ndarray, y:np.ndarray,
        trial_frequencies:np.ndarray=None,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
            - method for executing the Lomb Scargle
        
            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - `trial_periods`
                    - np.ndarray, optional
                    - trial periods to evaluate Lomb Scargle on
                    - if passed will overwrite `self.trial_periods`
                    - the default is `None`
                        - will use `self.trial_periods`
            
            Raises
            ------

            Returns
            -------
                - `powers_ls`
                    - np.ndarray
                    - powers corresponding to `self.tiral_periods_ls`
                - `self.trial_periods_ls`
                    - np.ndarray
                    - trial periods used for the execution of the Lomb Scargle

            Comments
            --------        
        """

        if trial_frequencies is None: trial_frequencies = self.trial_frequencies

        self.ls = LombScargle(x, y, **self.ls_kwargs)

        #get powers
        powers_ls = self.ls.power(trial_frequencies)
        
        #sortupdate self.trial_frequencies to be comparable with thetas_pdm
        powers_ls = powers_ls[np.argsort(1/trial_frequencies)]
        trial_frequencies = trial_frequencies[np.argsort(1/trial_frequencies)]
        
        #sort frequencies to be comparable
        self.best_frequency_ls = trial_frequencies[np.nanargmax(powers_ls)]
        self.best_power_ls  = np.nanmax(powers_ls)

        return powers_ls, trial_frequencies
    
    def get_psi(self,
        thetas_pdm:np.ndarray, powers_ls:np.ndarray,
        ) -> None:
        """
            - method to compute the HPS-metric
            - essentially calculates the following
                - rescale $\Pi$ to `range(0,1)`
                - calculate $1-\theta$
                - rescale $1-\theta$ to `range(0,1)`
                - $\Phi = \Pi|_0^1 * (1-\theta)|_0^1$
                    - i.e. product of the two calculated metrics
                - $\Pi$             ... powers of Lomb-Scargle
                - $\Pi|_0^1$        ... powers of Lomb-Scargle squeezed into `range(0,1)`
                - $\theta$          ... theta statistics of PDM
                - $(1-\theta)$      ... inverted theta statistics of PDM
                - $(1-\theta)|_0^1$ ... inverted theta statistics of PDM squeezed into `range(0,1)`

            Parameters
            ----------
                - `thetas_pdm`
                    - np.ndarray
                    - thetas resulting from a pdm-analysis
                - `powers_ls`
                    - np.ndarray
                    - powers resulting from a Lomb-Scargle analysis

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        #scaler to 'squeeze' 1-theta and lomb-scargle powers into range(0,1) 
        scaler = MinMaxScaler(feature_range=(0,1))

        self.thetas_hps = scaler.fit_transform((1-thetas_pdm).reshape(-1,1)).reshape(-1)
        self.powers_hps = scaler.fit_transform(powers_ls.reshape(-1,1)).reshape(-1)

        #calculate psi
        self.psis_hps = self.powers_hps * self.thetas_hps

        return

    def fit(self,
        x:np.ndarray, y:np.ndarray,
        trial_periods:np.ndarray=None,
        verbose:int=None,
        ) -> None:
        """
            - method to fit the HPS-estimator
            - will execute the calculation and assign results as attributes
            - similar to fit-method in scikit-learn

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - x values of the dataseries to run `HPS` on
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run `HPS` on
                - `trial_periods`
                    - np.ndarray, optional
                    - if passed will overwrite `self.trial_periods`
                    - the default is `None`
                - `verbose`
                    - int, optional
                    - will overwrite `self.verbose` attribute
                    - verbosity level
                    - the default is 0
                                        
            Raises
            ------

            Returns as Attributes
            ---------------------
                - `best_period`
                    - float
                    - the period yielding the lowest variance in the whole curve
                - `best_psi`
                    - float
                    - psi value corresponding to `best_period`
            
            Returns
            -------

            Comments
            --------


        """


        if verbose is None: verbose = self.verbose

        if trial_periods is None:
            trial_periods, trial_frequencies = self.generate_period_frequency_grid(x=x)
            self.trial_periods = trial_periods
            self.trial_frequencies = trial_frequencies
        else:
            trial_frequencies = 1/trial_periods
            self.trial_periods = trial_periods
            self.trial_frequencies = trial_frequencies


        #execute pdm
        self.thetas_pdm, trial_periods_pdm = self.run_pdm(x, y, trial_periods)

        #execute lomb-scargle
        self.powers_ls, trial_frequencies_ls = self.run_lombscargle(x, y, trial_frequencies)

        #calculate psi
        self.get_psi(self.thetas_pdm, self.powers_ls)

        best_period = self.trial_periods[np.nanargmax(self.psis_hps)]
        best_psi = np.nanmax(self.psis_hps)

        self.best_period = best_period
        self.best_psi = best_psi
        self.trial_periods = trial_periods_pdm
        self.trial_frequencies = trial_frequencies_ls

        return
    
    def predict(self,
        x:np.ndarray=None, y:np.ndarray=None, 
        ) -> Tuple[float, float]:
        """
            - method to predict with the fitted HPS-estimator
            - will return relevant results
            - similar to predict-method in scikit-learn

            Parameters
            ----------
                - `x`
                    - np.ndarray, optional
                    - x values of the dataseries to run HPS on
                    - only here for consistency, will not be considered in the method
                    - the default is `None`
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run HPS on
                    - only here for consistency, will not be considered in the method
                    - the default is `None`

            Raises
            ------
            
            Returns
            -------
                - `best_period`
                    - float
                    - best period estimate
                - `best_psi`
                    - float
                    - psi-value of `best_period`
            
            Comments
            --------

        """
        
        return self.best_period, self.best_psi
    
    def fit_predict(self,
        x:np.ndarray, y:np.ndarray,
        trial_periods:np.ndarray=None,
        ) -> Tuple[float, float]:
        """
            - method to fit classifier and predict the results

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - `y`
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - `trial_periods`
                    - np.ndarray, optional
                    - if passed will overwrite `self.trial_periods`
                    - the default is `None`

            Raises
            ------
                    
            Returns
            -------
                - `best_period`
                    - float
                    - the period yielding the lowest variance in the whole curve
                - `best_psi`
                    - float
                    - psi value corresponding to `best_period`
            
            Comments
            --------

        """

        self.fit(x, y, trial_periods)
        best_period, best_psi = self.predict()

        return best_period, best_psi
    
    def plot_result(self,
        x:np.ndarray=None, y:np.ndarray=None,
        fig_kwargs:dict=None,
        plot_kwargs:dict=None,
        ) -> Tuple[Figure, plt.Axes]:
        """
            - method to plot the result of the pdm
            - will produce a plot with 2 panels
                - top panel contains the periodogram
                - bottom panel contains the input-dataseries folded onto the best period

            Parameters
            ----------
                - `x`
                    - np.ndarray, optional
                    - x-values of a dataseries to plot folded with the determined period
                    - usually the dataseries the analysis was done on
                    - the default is `None`
                        - will not plot a dataseries
                - `y`
                    - np.ndarray, optional
                    - y-values of a dataseries to plot folded with the determined period
                    - usually the dataseries the analysis was done on
                    - the default is `None`
                        - will not plot a dataseries
                - `fig_kwargs`
                    - dict, optional
                    - kwargs for matplotlib `plt.figure()` method
                    - the default is `None`
                        - will initialize with `{}`
                - `plot_kwargs`
                    - dict, optional
                    - kwargs for matplotlib `ax.plot()` method
                    - the default is `None`
                        - will initialize with an empty dict
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - matplotlib Figure
                    - figure created if verbosity level specified accordingly
                - `axs`
                    - matplotlib axes
                    - axes corresponding to `fig`

            Comments
            --------

        """

        if fig_kwargs  is None: fig_kwargs = {}
        if plot_kwargs is None: plot_kwargs = {}
        
        c_ls = 'tab:olive'
        c_pdm = 'tab:green'
        c_hps = 'tab:orange'
        
        fig = plt.figure(**fig_kwargs)
        #check if folded dataseries shall be plotted as well
        if x is not None and y is not None:
            ax1 = fig.add_subplot(211)
            ax4 = fig.add_subplot(212)
        else:
            ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))

        ax1.set_title("HPS-result")


        #sort axis
        ax1.set_zorder(3)
        ax2.set_zorder(2)
        ax3.set_zorder(1)
        ax1.patch.set_visible(False)
        ax2.patch.set_visible(False)

        l_hps,  = ax1.plot(self.trial_periods,  self.psis_hps,   color=c_hps, zorder=3, **plot_kwargs, label=r'HPS')
        l_pdm,  = ax2.plot(self.trial_periods, self.thetas_hps, color=c_pdm,  zorder=2, **plot_kwargs, label=r'PDM')
        l_ls,   = ax3.plot(1/self.trial_frequencies,  self.powers_hps, color=c_ls,   zorder=1, **plot_kwargs, label=r'Lomb-Scargle')
        
        vline   = ax1.axvline(self.best_period, linestyle='--', color='tab:grey', zorder=3, label=r'$\mathrm{P_{HPS}}$ = %.3f'%(self.best_period))

        ax1.set_xlabel('Period')
        ax1.set_ylabel(r'$\Psi$',   color=c_hps)
        ax2.set_ylabel(r'$(1-\theta)\left|_0^1\right.$', color=c_pdm)
        ax3.set_ylabel(r'$\Pi\left|_0^1\right.$',    color=c_ls)

        lines = [l_hps, l_pdm, l_ls, vline]
        ax1.legend(lines, [l.get_label() for l in lines])

        
        ax1.spines['right'].set_color(c_pdm)
        ax1.spines['left'].set_color(c_hps)
        ax2.spines['right'].set_color(c_pdm)
        ax3.spines['right'].set_color(c_ls)

        ax1.tick_params(axis='y', colors=c_hps)
        ax2.tick_params(axis='y', colors=c_pdm)
        ax3.tick_params(axis='y', colors=c_ls)
        
        #plot folded dataseries if requested
        if x is not None and y is not None:
            
            ax4.set_title('Folded Input')
            ax4.scatter(fold(x, self.best_period, 0)[0], y, color='tab:blue', s=1, label='Folded Input-Dataseries')
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')

        plt.tight_layout()

        axs = fig.axes

        return fig, axs
    
# %%
