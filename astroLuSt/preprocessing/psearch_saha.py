
#%%imports
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import numpy as np


from astroLuSt.preprocessing.pdm import PDM


#%%definitions
class PSearch_Saha:
    """
        - class to execute a period search according to Saha et al., 2017
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
                - best period according to the metric of Saha et al., 2017
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
            - psis_saha
                - np.ndarray
                - phi corresponding to trial_periods_saha
            - thetas_pdm
                - np.ndarray
                - thetas corresponding to trial_periods_pdm
            - trial_periods_ls
                - np.ndarray
                - final trial periods used for the execution of Lomb Scargle
            - trial_periods_pdm
                - np.ndarray
                - final trial periods used for the execution of PDM
            - trial_periods_saha
                - np.ndarray
                - final trial periods used for the execution of Saha et al., 2017 algorithm
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

        Comments
        --------
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
            f'PSearch_Saha(\n'
            f'    period_start={self.period_start}, period_stop={self.period_stop}, nperiods={self.nperiods},\n'
            f'    trial_periods={self.trial_periods},\n'
            f'    verbose={self.verbose},\n'
            f'    pdm_kwargs={self.pdm_kwargs},\n'
            f')'
        )

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
            - method to execute the period analysis according to Saha et al., 2017
            - essentially calculates the following
                - $\frac{\Pi}{\theta}$
                - $\Pi$ ... powers of Lomb-Scargle
                - $\theta$ ... theta statistics of PDM

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
            trial_periods_saha = self.trial_periods_pdm
        elif self.trial_periods_pdm is None:
            trial_periods_saha = self.trial_periods_ls
        else:
            trial_periods_saha = self.trial_periods

        self.psis_saha = powers_ls / thetas_pdm
        self.trial_periods_saha = trial_periods_saha

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

        best_period = self.trial_periods_saha[np.nanargmax(self.psis_saha)]
        best_psi = np.nanmax(self.psis_saha)

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
        c_saha = 'tab:orange'
        
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

        l_saha, = ax1.plot(self.trial_periods_saha,self.psis_saha,  color=c_saha, zorder=3, **plot_kwargs, label=r'Saha et al., 2017)')
        l_pdm,  = ax2.plot(self.trial_periods_pdm, self.thetas_pdm, color=c_pdm,  zorder=2, **plot_kwargs, label=r'PDM (Stellingwerf, 1978)')
        l_ls,   = ax3.plot(self.trial_periods_ls,  self.powers_ls,  color=c_ls,   zorder=1, **plot_kwargs, label=r'Lomb-Scargle (Astropy)')
        
        ax1.axvline(self.best_period, linestyle='--', color='tab:grey', zorder=3, label=r'$\mathrm{P_{Saha}}$ = %.3f'%(self.best_period))

        ax1.set_xlabel('Period')
        ax1.set_ylabel(r'$\Psi$',   color=c_saha)
        ax2.set_ylabel(r'$\theta$', color=c_pdm)
        ax3.set_ylabel(r'$\Pi$',    color=c_ls)

        lines = [l_saha, l_pdm, l_ls]
        ax1.legend(lines, [l.get_label() for l in lines])

        
        # ax1.spines['right'].set_color(l_saha.get_color())
        ax2.spines['right'].set_color(c_pdm)
        ax3.spines['right'].set_color(c_ls)

        ax1.tick_params(axis='y', colors=c_saha)
        ax2.tick_params(axis='y', colors=c_pdm)
        ax3.tick_params(axis='y', colors=c_ls)
        
        plt.tight_layout()
        plt.show()

        axs = fig.axes

        return fig, axs