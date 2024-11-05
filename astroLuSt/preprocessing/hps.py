


#%%imports
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Union, Tuple, Callable
import warnings


from astroLuSt.preprocessing.pdm import PDM
from astroLuSt.preprocessing.timeseries import fold


#%%definitions
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

        warnings.warn('THIS FUNCTION HAS BEEN MOVED TO `astroLuSt.preprocessing.periodanalysis` AND WILL ONLY BE ACCESSIBLE FROM THERE IN THE FUTURE!')

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
            f'HPS(\n'
            f'    period_start={repr(self.period_start)}, period_stop={repr(self.period_stop)}, nperiods={repr(self.nperiods)},\n'
            f'    trial_periods={repr(self.trial_periods)},\n'
            f'    n_nyq={repr(self.n_nyq)},\n'
            f'    n0={repr(self.n0)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f'    pdm_kwargs={repr(self.pdm_kwargs)}, ls_kwargs={repr(self.ls_kwargs)}, lsfit_kwargs={repr(self.lsfit_kwargs)}\n'
            f')'
        )


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
    
    