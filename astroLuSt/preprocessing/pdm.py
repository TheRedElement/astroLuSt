
#%%imports
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Tuple, Callable

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
            - period_start
                - float, optional
                - the period to consider as starting point for the analysis
                - the default is None
                    - will try to consider timeseries and estimate nyquist frequency from that
                    - if that fails defaults to 1
            - period_stop
                - float, optional
                - the period to consider as stopping point for the analysis
                - if n_retries is > 0 will be increased by n_retries*nperiods_retry
                - the default is None
                    - will try to consider the length of the timeseries to analyze (i.e. maximum determinable period = length of dataseries)
                    - if that fails will be set to 100
            - nperiods
                - int, optional
                - how many trial periods to consider during the analysis
                - the default is 100
            - n_nyq
                - float, optional
                - nyquist factor
                - the average nyquist frequency corresponding to 'x' will be multiplied by this value to get the minimum period
                - the default is None
                    - will default to 1
            - n0
                - int, optional
                - oversampling factor
                - i.e. number of datapoints to use on each peak in the periodogram
                - the default is None
                    - will default to 5
            - trial_periods
                - np.ndarray, optional
                - if passed will use the values in that array and ignore
                    - period_start
                    - period_stop
                    - nperiods
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
            - binning_kwargs
                - dict, optional
                - kwargs for the Binning class
                - used to bin the folded curves and estimate the variance w.r.t. a mean curve            
        
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
        if binning_kwargs is None:
            self.binning_kwargs = {'nintervals':100}
        else:
            self.binning_kwargs = binning_kwargs

        #adopt period_start and period_stop if trial_periods were passed
        if self.trial_periods is not None:
            self.period_start = np.nanmin(self.trial_periods)
            self.period_stop  = np.nanmax(self.trial_periods)
        
        pass

    def __repr__(self) -> str:

        return (
            f'PDM(\n'
            f'    period_start={self.period_start},\n'
            f'    period_stop={self.period_stop},\n'
            f'    nperiods={self.nperiods},\n'
            f'    trial_periods={self.trial_periods},\n'
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
            f'    binning_kwargs={self.binning_kwargs},\n'
            f')'       
        )

    def generate_period_grid(self,
        period_start:float=None, period_stop:float=None, nperiods:float=None,
        x:np.ndarray=None,
        n_nyq:int=None,
        n0:int=None,
        ) -> np.ndarray:
        """
            - method to generate a period grid
            - inspired by astropy.timeseries.LombScargle().autofrequency() and VanderPlas (2018)
                - https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html
                - https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract

            Parameters
            ----------
                - period_start
                    - float, optional
                    - the period to consider as starting point for the analysis
                    - the default is None
                        - will default to self.period_start
                - period_stop
                    - float, optional
                    - the period to consider as stopping point for the analysis
                    - the default is None
                        - will default to 100 if "x" is also None
                        - otherwise will consider x to generate period_stop
                - nperiods
                    - int, optional
                    - how many trial periods to consider during the analysis
                    - the default is None
                        - will default to self.nperiods
                - x
                    - np.ndarray, optional
                    - input array
                    - x-values of the data-series
                    - the default is None
                        - if set and period_stop is None, will use max(x)-min(x) as 'period_stop'
                - n_nyq
                    - float, optional
                    - nyquist factor
                    - the average nyquist frequency corresponding to 'x' will be multiplied by this value to get the minimum period
                    - the default is None
                        - will default to self.n_nyq
                        - if self.n_nyq is also None will default to 1
                - n0
                    - int, optional
                    - oversampling factor
                    - i.e. number of datapoints to use on each peak in the periodogram
                    - the default is None
                        - will default to self.n0
                        - if self.n0 is also None will default to 5
            
            Raises
            ------

            Returns
            -------
                - trial_periods
                    - np.ndarray
                    - final trial periods used for the execution of PDM

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
        ) -> Tuple[float, float]:
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
        verbose:int=None
        ) -> None:
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

            Returns as Attributes
            ---------------------
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
            self.trial_periods = self.generate_period_grid(self.period_start, self.period_stop, self.nperiods, x=x)

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
            - method to predict with the fitted pdm-estimator
            - will return relevant results
            - similar to predict-method in scikit-learn

            Parameters
            ----------
                - x
                    - np.ndarray, optional
                    - x values of the dataseries to run PDM on
                    - only here for consistency, will not be considered in the method
                    - the default is None
                - y
                    - np.ndarray
                    - y values of the dataseries to run PDM on
                    - only here for consistency, will not be considered in the method
                    - the default is None
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
        return self.best_period, self.errestimate, self.best_theta
    
    def fit_predict(self,
        x:np.ndarray, y:np.ndarray,
        fit_kwargs:dict={}
        ) -> Tuple[float, float, float]:
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
                - fit_kwargs
                    - keyword arguments passed to fit()

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
                - x
                    - np.ndarray, optional
                    - x-values of a dataseries to plot folded with the determined period
                    - usually the dataseries the analysis was done one
                    - the default is None
                        - will not plot a dataseries
                - y
                    - np.ndarray, optional
                    - y-values of a dataseries to plot folded with the determined period
                    - usually the dataseries the analysis was done one
                    - the default is None
                        - will not plot a dataseries            
                - fig_kwargs
                    - dict, optional
                    - kwargs for matplotlib plt.figure() method
                    - the default is None
                        - will initialize with an empty dict
                - sctr_kwargs
                    - dict, optional
                    - kwargs for matplotlib ax.scatter() method used to plot theta(period)
                    - the default is None
                        - will initialize with an empty dict
            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure
                    - figure created if verbosity level specified accordingly
                - axs
                    - matplotlib axes
                    - axes corresponding to 'fig'  
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
        ax1.scatter(self.trial_periods, self.thetas, color='tab:blue', s=1, zorder=1, **sctr_kwargs)
        ax1.axvline(self.best_period, color='tab:grey', linestyle="--", label=r'$\mathrm{P_{HPS}}$ = %.3f'%(self.best_period), zorder=2)
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