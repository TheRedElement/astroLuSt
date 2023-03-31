
#TODO: 
#   - Generate trial periods (for PDM)
#   - Generate trial frequencies (for LS)
#   - combine into two arrays (one of P, one of f)
#   - Run LS on combined f
#   - Run PDM on combined P
#   - Generate trial periods for refinement of PDM
#   - Generate trial frequencies for refinement of LS
#   - combine into two arrays (one for P, one for f)
#   - Refine periods
#   - repeat


#%%imports
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


from astroLuSt.preprocessing.pdm import PDM


#%%definitions
class HPS:
    """
        - HPS = Hybrid Period Search
        - class to execute a period search inspired by Saha et al., 2017
            - https://ui.adsabs.harvard.edu/abs/2017AJ....154..231S/abstract
        
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
            - verbose
                - int, optional
                - verbosity level
                - the default is 0
            - pdm_kwargs
                - dict, optional
                - kwargs for the PDM class
            - ls_kwargs
                - dict, optional
                - kwargs for the astropy.timeseries.LombScargle class
            - lsfit_kwargs
                - dict, optional
                - kwargs for the autopower() method of the astropy.timeseries.LombScargle class
            

        Infered Attributes
        ------------------
            - best_period
                - float
                - best period according to the metric of HPS
            - best_period_ls
                - float
                - best period according to Lomb-Scargle
            - best_period_pdm
                - float
                - best period according to PDM
            - best_power_ls
                - float
                - power corresponding to best_period_ls
            - best_psi
                - float
                - metric corresponding to best_period
            - best_theta_pdm
                - float
                - best theta statistics corresponding to best_period_pdm
            - errestimate_pdm
                - float
                - estimate of the error of best_period_pdm
            - powers_ls
                - np.ndarray
                - powers corresponding to trial_periods_ls
            - powers_hps
                - np.ndarray
                - powers of the hps alogrithm
                - calculated by
                    - squeezing powers_ls into range(0,1)
            - psis_hps
                - np.ndarray
                - phi corresponding to trial_periods_hps
            - thetas_pdm
                - np.ndarray
                - thetas corresponding to trial_periods_pdm
            - thetas_hps
                - np.ndarray
                - thetas of the hps alogrithm
                - calculated by
                    - evaluating 1-thetas_pdm 
                    - squeezing result into range(0,1)
            - trial_periods_ls
                - np.ndarray
                - final trial periods used for the execution of Lomb Scargle
            - trial_periods_pdm
                - np.ndarray
                - final trial periods used for the execution of PDM
            - trial_periods_hps
                - np.ndarray
                - final trial periods used for the execution of HPS algorithm
            - pdm
                - instance of PDM class
                - contains all information of the pdm fit
            - ls
                - instance of LombScargle class
                - contains all information of the LombScargle fit
                            
        Methods
        -------
            - run_pdm()
            - run_lombscargle()
            - get_psi()
            - fit()
            - predict()
            - fit_predict()
            - plot_result()

        Raises
        ------

        Dependencies
        ------------
            - astropy
            - matplotlib
            - numpy
            - sklearn

        Comments
        --------
            - basic runthrough of computation
                - take dataseries
                - compute Lomb-Scargle
                - compute PDM
                - rescale Lomb-Scargle power to range(0,1) (:=Pi_hps)
                - invert PDM theta statistics (theta) by evaluating 1-theta
                - rescale inverted PDM theta statistics to range(0,1) (:=theta_hps)
                - calculate new metric as Psi = Pi_hps * theta_hps
            - essentially upweights periods where Lomb-Scargle and PDM agree and downweights those where they disagree
                - if at some period Lomb-Scargle has a strong peak and PDM has a strong minimum the inverted PDM minimum will amplify the Lomb-Scargle peak
                - if one has a peak and the other one does not, then the respective peak gets dammed

    """

    def __init__(self,
        period_start:float=0.1, period_stop:float=None, nperiods:int=100,        
        trial_periods:np.ndarray=None,
        verbose:int=0,
        pdm_kwargs:dict=None, ls_kwargs:dict=None, lsfit_kwargs:dict=None
        ) -> None:

        self.period_start = period_start
        self.period_stop = period_stop
        self.nperiods = nperiods
        self.trial_periods = trial_periods

        self.verbose = verbose

        if pdm_kwargs is None:
            self.pdm_kwargs = {'n_retries':1, 'n_jobs':1}
        else:
            self.pdm_kwargs = pdm_kwargs
        
        if ls_kwargs is None:
            self.ls_kwargs = {}
        else:
            self.ls_kwargs = ls_kwargs
        
        if lsfit_kwargs is None:
            self.lsfit_kwargs = {}
        else:
            self.lsfit_kwargs = lsfit_kwargs
        

        pass

    def __repr__(self) -> str:
        
        return (
            f'HPS(\n'
            f'    period_start={self.period_start}, period_stop={self.period_stop}, nperiods={self.nperiods},\n'
            f'    trial_periods={self.trial_periods},\n'
            f'    verbose={self.verbose},\n'
            f'    pdm_kwargs={self.pdm_kwargs},\n'
            f')'
        )

    def generate_period_grid(self,
        period_start:float=None, period_stop:float=None, nperiods:float=100,
        x:np.ndarray=None,
        ):
        """
 
        """

        #overwrite defaults if requested
        if period_start is None: period_start = self.period_start
        if period_stop is None: period_stop = self.period_stop
        if nperiods is None: nperiods = self.nperiods


        test_periods_pdm = PDM().generate_period_grid(period_start, period_stop, nperiods, x)
        print(test_periods_pdm)

        return 

    def run_pdm(self,
        x:np.ndarray, y:np.ndarray,
        trial_periods:np.ndarray=None,
        ):
        """
            - method to execute a Phase-Dispersion Minimization
            
            Parameters
            ----------
                - x
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - y
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - trial_periods
                    - np.ndarray, optional
                    - if passed will overwrite self.trial_periods
                    - the default is None
            
            Raises
            ------

            Returns
            -------
                - pdm.thetas
                    - np.ndarray
                    - thetas corresponding to pdm.trial_periods
                - pdm.trial_periods
                    - np.ndarray
                    - trial periods used for the execution of the PDM

            Comments
            --------
        """
        
        if trial_periods is None: trial_periods = self.trial_periods

        self.pdm = PDM(
            self.period_start, self.period_stop, self.nperiods,
            trial_periods,
            **self.pdm_kwargs
        )

        self.best_period_pdm, self.errestimate_pdm, self.best_theta_pdm = self.pdm.fit_predict(x, y)

        #update pdm trial periods 
        self.trial_periods_pdm = self.pdm.trial_periods

        #update self.trial_periods to be aligned with pdm (in case n_retries in pdm_kwargs > 0)
        self.trial_periods = self.pdm.trial_periods

        return self.pdm.thetas, self.pdm.trial_periods

    def run_lombscargle(self,
        x:np.ndarray, y:np.ndarray,
        trial_periods:np.ndarray=None,
        ):
        """
            - method for executing the Lomb Scargle
        
            Parameters
            ----------
                - x
                    - np.ndarray
                    - x values of the dataseries to run phase dispersion minimization on
                - y
                    - np.ndarray
                    - y values of the dataseries to run phase dispersion minimization on
                - trial_periods
                    - np.ndarray, optional
                    - if passed will overwrite self.trial_periods
                    - the default is None
            
            Raises
            ------

            Returns
            -------
                - powers_ls
                    - np.ndarray
                    - powers corresponding to self.tiral_periods_ls
                - self.trial_periods_ls
                    - np.ndarray
                    - trial periods used for the execution of the Lomb Scargle

            Comments
            --------        
        """

        if trial_periods is None: trial_periods = self.trial_periods

        self.ls = LombScargle(x, y, **self.ls_kwargs)

        if trial_periods is None:
            frequencies_ls, powers_ls = self.ls.autopower(
                minimum_frequency=1/self.period_stop, maximum_frequency=1/self.period_start,
                **self.lsfit_kwargs,
            )

            #update pdm trial periods 
            self.trial_periods_ls = 1/frequencies_ls

            #update self.trial_periods to be aligned with lomb-scargle
            self.trial_periods    = 1/frequencies_ls
        else:
            powers_ls = self.ls.power(1/self.trial_periods)
            
            #update pdm trial periods 
            self.trial_periods_ls = self.trial_periods

        self.best_period_ls = self.trial_periods_ls[np.nanargmax(powers_ls)]
        self.best_power_ls  = np.nanmax(powers_ls)
        return powers_ls, self.trial_periods_ls

    def get_psi(self,
        thetas_pdm:np.ndarray, powers_ls:np.ndarray,
        ):
        """
            - method to compute the HPS-metric
            - essentially calculates the following
                - rescale $\Pi$ to range(0,1)
                - calculate $1-\theta$
                - rescale $1-\theta$ to range(0,1)
                - $\Phi = \Pi|_0^1 * (1-\theta)|_0^1$
                    - i.e. product of the two calculated metrics
                - $\Pi$             ... powers of Lomb-Scargle
                - $\Pi|_0^1$        ... powers of Lomb-Scargle squeezed into range(0,1)
                - $\theta$          ... theta statistics of PDM
                - $(1-\theta)$      ... inverted theta statistics of PDM
                - $(1-\theta)|_0^1$ ... inverted theta statistics of PDM squeezed into range(0,1)

            Parameters
            ----------
                - thetas_pdm
                    - np.ndarray
                    - thetas resulting from a pdm-analysis
                - powers_ls
                    - np.ndarray
                    - powers resulting from a Lomb-Scargle analysis

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        if self.trial_periods is None:
            trial_periods_hps = self.trial_periods_pdm
        elif self.trial_periods_pdm is None:
            trial_periods_hps = self.trial_periods_ls
        else:
            trial_periods_hps = self.trial_periods

        #scaler to 'squeeze' 1-theta and lomb-scargle powers into range(0,1) 
        scaler = MinMaxScaler(feature_range=(0,1))

        self.thetas_hps = scaler.fit_transform((1-thetas_pdm).reshape(-1,1)).reshape(-1)
        self.powers_hps = scaler.fit_transform(powers_ls.reshape(-1,1)).reshape(-1)

        self.psis_hps = self.powers_hps * self.thetas_hps
        self.trial_periods_hps = trial_periods_hps

        return

    def fit(self,
        x:np.ndarray, y:np.ndarray,
        trial_periods:np.ndarray=None,
        verbose:int=None,
        ):
        """
            - method to fit the PSearch-estimator
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
                - trial_periods
                    - np.ndarray, optional
                    - if passed will overwrite self.trial_periods
                    - the default is None                
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
                - best_psi
                    - float
                    - psi value corresponding to best_period


        """


        if verbose is None: verbose = self.verbose

        #execute pdm
        self.thetas_pdm, trial_periods_pdm = self.run_pdm(x, y, trial_periods)

        #execute lomb-scargle
        self.powers_ls, trial_periods_ls = self.run_lombscargle(x, y, trial_periods)

        #calculate psi
        self.get_psi(self.thetas_pdm, self.powers_ls)

        best_period = self.trial_periods_hps[np.nanargmax(self.psis_hps)]
        best_psi = np.nanmax(self.psis_hps)

        self.best_period = best_period
        self.best_psi = best_psi

        return
    
    def predict(self):
        """
            - method to predict with the fitted PSearch-estimator
            - will return relevant results
            - similar to predict-method in scikit-learn

            Returns
            -------
                - best_period
                    - float
                    - best period estimate
                - best_psi
                    - float
                    - psi-value of best period
        """
        return self.best_period, self.best_psi
    
    def fit_predict(self,
        x:np.ndarray, y:np.ndarray,
        trial_periods:np.ndarray=None,
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
                - trial_periods
                    - np.ndarray, optional
                    - if passed will overwrite self.trial_periods
                    - the default is None                      

            Returns
            -------
                - best_period
                    - float
                    - the period yielding the lowest variance in the whole curve
                - best_psi
                    - float
                    - psi value corresponding to best_period                          
        """
        self.fit(x, y, trial_periods)
        best_period, best_psi = self.predict()

        return best_period, best_psi
    
    def plot_result(self,
        fig_kwargs:dict={},
        plot_kwargs:dict={},
        ):
        """
            - method to plot the result of the pdm
            - will produce a plot with 2 panels
                - top panel contains the periodogram
                - bottom panel contains the input-dataseries folded onto the best period

            Parameters
            ----------
                - fig_kwargs
                    - dict, optional
                    - kwargs for matplotlib plt.figure() method
                - plot_kwargs
                    - dict, optional
                    - kwargs for matplotlib ax.plot() method
        """
        
        c_ls = 'tab:olive'
        c_pdm = 'tab:green'
        c_hps = 'tab:orange'
        
        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))

        #sort axis
        ax1.set_zorder(3)
        ax2.set_zorder(2)
        ax3.set_zorder(1)
        ax1.patch.set_visible(False)
        ax2.patch.set_visible(False)

        l_hps, = ax1.plot(self.trial_periods_hps,  self.psis_hps,   color=c_hps, zorder=3, **plot_kwargs, label=r'HPS')
        l_pdm,  = ax2.plot(self.trial_periods_pdm, self.thetas_hps, color=c_pdm,  zorder=2, **plot_kwargs, label=r'PDM')
        l_ls,   = ax3.plot(self.trial_periods_ls,  self.powers_hps, color=c_ls,   zorder=1, **plot_kwargs, label=r'Lomb-Scargle')
        
        ax1.axvline(self.best_period, linestyle='--', color='tab:grey', zorder=3, label=r'$\mathrm{P_{HPS}}$ = %.3f'%(self.best_period))

        ax1.set_xlabel('Period')
        ax1.set_ylabel(r'$\Psi$',   color=c_hps)
        ax2.set_ylabel(r'$(1-\theta)\left|_0^1\right.$', color=c_pdm)
        ax3.set_ylabel(r'$\Pi\left|_0^1\right.$',    color=c_ls)

        lines = [l_hps, l_pdm, l_ls]
        ax1.legend(lines, [l.get_label() for l in lines])

        
        # ax1.spines['right'].set_color(l_hps.get_color())
        ax2.spines['right'].set_color(c_pdm)
        ax3.spines['right'].set_color(c_ls)

        ax1.tick_params(axis='y', colors=c_hps)
        ax2.tick_params(axis='y', colors=c_pdm)
        ax3.tick_params(axis='y', colors=c_ls)
        
        plt.tight_layout()
        plt.show()

        axs = fig.axes

        return fig, axs