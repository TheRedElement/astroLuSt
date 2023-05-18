
#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import scipy.stats as stats
from typing import Union, Tuple

#%%definitions
class MLE:
    """
    
        - Class to execute a simple Maximum Likelihood Estimat on data following a Gaussian distibution

        Attributes
        ----------
            - dataseries
                - np.ndarray of np.ndarrays
                - contains the different datasets to compare
                - rows are different samples
                - columns are different features
            - series_labels
                - np.ndarray, optional
                - contains labels corresponding to the dataseries in 'data'
                - the default is None
        
        Infered Attributes
        ------------------
            - self.mus
                - np.ndarray
                - array containing the mean of the gaussians
            - self.sigmas
                - np.ndarray
                - array containing the standard-deviations of the gaussians
            - self.covmat
                - np.ndarray
                - covariance matrix of the input data
            - self.corrcoeff
                - np.ndarray
                - correlation matrix containing pearson correlation coefficient

        Methods
        -------
            - get_mu
            - get_sigma
            - get_covmat
            - fit
            - predict
            - fit_predict
            - corner_plot

        Dependencies
        ------------
            - matplotlib
            - numpy
            - scipy
            - typing
    
    """


    def __init__(self,
        ):

        self.mus = np.array([])
        self.sigmas = np.array([])
        self.covmat = np.array([])
        self.corrcoeff = np.array([])

        pass

    def get_mu(self,
        X:np.ndarray
        ):
        """
            - method to estimate the mean of the gaussian via MLE

            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        self.mus = np.nanmean(X, axis=0)

        return
    
    def get_sigma(self,
        X:np.ndarray
        ):
        """
            - method to estimate the standard deviation of the gaussian via MLE
            - estimate for 1d-case (i.e. 1D histogram)
            
            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        self.sigmas = np.nanstd(X, axis=0)

        return
    
    def get_covmat(self, X:np.ndarray=None):
        """
            - method to estimate the N-D covariance matrix

            Parameters
            ----------
                - data
                    - np.array, optional
                    - 2D array
                        - 0th axis contains the samples
                        - 1st axis contains the measurements for each sample
                    - the default is None
                        - will use 'self.dataseries' as fallback
                        
                - mus
                    - np.array, optional
                    - contains the mean values (mu) corresponding to 'data'
                        - has to have a lenght equal to the second dimension of 'data'
                    - the default is None
                        - will use 'self.mus' as fallback
                
                Raises
                ------

                Returns
                -------
                    - covmat
                        - np.array
                        - covariance matrix for a given set of datasets ('data')
                
                Dependencies
                ------------
                    - numpy

        """

        #N-D data
        if X is None:
            X = self.X
        self.covmat = np.cov(X.T)

        return self.covmat

    def fit(self,
        X:np.ndarray, y:np.ndarray=None
        ):
        """
            - method to fit the estimator
            - similar to scikit-learn framework

            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #adopt inputs
        self.X = X
        self.y = y

        #fit estimator
        self.get_mu(X)
        self.get_sigma(X)
        self.covmat = self.get_covmat(X)
        self.corrcoeff = np.corrcoef(X.T)

        return

    def predict(self,
        X:np.ndarray=None, y:np.ndarray=None):
        """
            - method to predict using the estimator
            - similar to scikit-learn framework

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - self.mus
                - self.sigmas
                - self.convmat

            Comments
            --------
            
        """


        return self.mus, self.sigmas, self.covmat

    def fit_predict(self,
        X:np.ndarray, y:np.ndarray=None):
        """
            - method to fit the estimator and predict the result afterwards
            
            Returns
            -------
                - self.mus
                - self.sigmas
                - self.convmat        
        """

        self.fit(X=X, y=y)
        mus, sigmas, covmat = self.predict()

        return mus, sigmas, covmat

    def corner_plot(self,
        X:np.ndarray=None, y:Union[np.ndarray,str]=None, featurenames:np.ndarray=None,
        mus:np.ndarray=None, sigmas:np.ndarray=None,
        bins:int=100, equal_range:bool=False, asstandardnormal:bool=False,
        save:str=False,
        sctr_kwargs:dict=None
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to create a corner plot of a given set of input distributions
        
            Parameters
            ----------
                - data
                    - list of np.arrays, optional
                    - contains the different datasets to compare
                    - the default is None
                        - will use the class attribute 'dataseries'
                - labels
                    - list, optional
                    - contains labels corresponding to the dataseries in 'data'
                    - the default is None
                        - will use the class attribute 'series_labels'
                - mus
                    - list, optional
                    - contains the mean value estimates corresponding to 'data'
                    - the default is None
                        - will use the class attribute 'mus'
                - sigmas
                    - list, optional
                    - contains the standard deviation estimates corresponding to 'data'
                    - the default is None
                        - will use the class attribute 'sigmas'
                - bins
                    - int, optional
                    - number of bins to use in plt.histogram
                    - also the resolution for the estiation of the normal distribution
                    - the default is 100
                - equal_range
                    - bool, optional
                    - whether to plot the data with equal x- and y-limits
                    - the default is False
                - asstandardnormal
                    - bool, optional
                    - whether to plot the data rescaled to zero mean and unit variance
                    - the default is False
                - save
                    - str, optional
                    - location of where to save the created figure
                    - the default is None
                        - will not save the figure

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure object
                - axs
                    - matplotlib axes object
                    - corresponding to 'fig'

        """


        #initialize correctly
        if X is None: X = self.X
        if y is None:
            y = 'tab:blue'
            # y = self.y
            # if self.y is None:
        if featurenames is None: featurenames = [f"Feature {i}" for i in np.arange(X.shape[1])]

        if mus is None:
            if len(self.mus) == 0:
                mus = [None]*len(X)
            else:
                mus = self.mus
        if sigmas is None:
            if len(self.sigmas) == 0:
                sigmas = [None]*len(X)
            else:
                sigmas = self.sigmas
        if sctr_kwargs is None:
            sctr_kwargs = {'s':1, 'alpha':0.5, 'zorder':2}


        fig = plt.figure()
        nrowscols = X.shape[1]


        idx = 0
        for idx1, (d1, l1, mu1, sigma1) in enumerate(zip(X.T, featurenames, mus, sigmas)):
            for idx2, (d2, l2, mu2, sigma2) in enumerate(zip(X.T, featurenames, mus, sigmas)):
                idx += 1

                if asstandardnormal and mu1 is not None and sigma1 is not None:
                    d1 = (d1-mu1)/sigma1
                    d2 = (d2-mu2)/sigma2
                    mu1, mu2 = 0, 0
                    sigma1, sigma2 = 1, 1
                
                #plotting 2D distributions
                if idx1 > idx2:

                    ax1 = fig.add_subplot(nrowscols, nrowscols, idx)
                    
                    if mu1 is not None: ax1.axhline(mu1, color="tab:orange", linestyle="--")
                    if mu2 is not None: ax1.axvline(mu2, color="tab:orange", linestyle="--")

                    #data
                    sctr = ax1.scatter(
                        d2, d1,
                        c=y,               
                        **sctr_kwargs,
                    )


                    #normal distribution estimate
                    if mu1 is not None and sigma1 is not None:
                        
                        covmat = self.get_covmat(np.array([d1,d2]).T)

                        xvals = np.linspace(np.nanmin([d1, d2]), np.nanmax([d1, d2]), bins)
                        yvals = np.linspace(np.nanmin([d1, d2]), np.nanmax([d1, d2]), bins)
                        xx, yy = np.meshgrid(xvals, yvals)
                        mesh = np.dstack((xx, yy))
                        
                        norm = stats.multivariate_normal(
                            mean=np.array([mu1, mu2]),
                            cov=covmat,
                            allow_singular=True
                        )
                        cont = ax1.contour(yy, xx, norm.pdf(mesh), cmap="gray", zorder=1)

                    #labelling
                    if idx1 == nrowscols-1:
                        ax1.set_xlabel(l2)
                    else:
                        ax1.set_xticklabels([])
                    if idx2 == 0:
                        ax1.set_ylabel(l1)
                    else:
                        ax1.set_yticklabels([])
                    ax1.tick_params()

                    if not equal_range:
                        ax1.set_xlim(np.nanmin(d2), np.nanmax(d2))
                        ax1.set_ylim(np.nanmin(d1), np.nanmax(d1))

                    if len(self.corrcoeff.shape) == 2:
                        ax1.errorbar(np.nan, np.nan, color="none", label=r"$r_\mathrm{P}=%.4f$"%(self.corrcoeff[idx1, idx2]))
                        ax1.legend()


                #plotting histograms
                elif idx1 == idx2:
                    if idx != 1:
                        orientation = "horizontal"
                    else:
                        orientation = "vertical"


                    axhist = fig.add_subplot(nrowscols, nrowscols, idx)

                    axhist.hist(d1, bins=bins, orientation=orientation, density=True)

                    #normal distribution estimate
                    if mu1 is not None and sigma1 is not None:
                        xvals = np.linspace(np.nanmin(d1), np.nanmax(d1), bins)
                        normal = stats.norm.pdf(xvals, mu1, sigma1)
                        
                        if orientation == "horizontal":
                            axhist.axhline(mu1, color="tab:orange", linestyle="--", label=r"$\mu=%.2f$"%(mu1))
                            axhist.plot(normal, xvals)
                            if not equal_range:
                                axhist.set_ylim(np.nanmin(d1), np.nanmax(d1))
                            
                            axhist.xaxis.set_ticks_position("top")
                            axhist.set_yticklabels([])
                        
                        elif orientation == "vertical":
                            axhist.plot(xvals, normal)
                            axhist.axvline(mu1, color="tab:orange", linestyle="--", label=r"$\mu=%.2f$"%(mu1))
                            if not equal_range:
                                axhist.set_xlim(np.nanmin(d1), np.nanmax(d1))
                            
                            axhist.yaxis.set_ticks_position("right")
                            axhist.set_xticklabels([])
                    
                        axhist.errorbar(np.nan, np.nan, color="none", label=r"$\sigma=%.2f$"%(sigma1))
                        axhist.legend()
                    

                    axhist.tick_params()
        
        #make x and y limits equal if requested
        if equal_range and not asstandardnormal:
            for idx, ax in enumerate(fig.axes):
                
                #first 1D histogram
                if idx == 0:
                    xymin = np.nanmin([fig.axes[idx+1].get_xlim(), fig.axes[idx+1].get_ylim()])
                    xymax = np.nanmax([fig.axes[idx+1].get_xlim(), fig.axes[idx+1].get_ylim()])
                    ax.set_xlim(xymin, xymax)

                #all other 1D histograms
                elif idx > 0 and (idx+1)%nrowscols == 0:
                    xymin = np.nanmin([fig.axes[idx-1].get_xlim(), fig.axes[idx-1].get_ylim()])
                    xymax = np.nanmax([fig.axes[idx-1].get_xlim(), fig.axes[idx-1].get_ylim()])
                    ax.set_ylim(xymin, xymax)

                #2D histograms
                else:
                    xymin = np.nanmin([fig.axes[idx].get_xlim(), fig.axes[idx].get_ylim()])
                    xymax = np.nanmax([fig.axes[idx].get_xlim(), fig.axes[idx].get_ylim()])

                    ax.set_xlim(xymin, xymax)
                    ax.set_ylim(xymin, xymax)


        plt.tight_layout()
        if isinstance(save, str):
            plt.savefig(save, dpi=180, bbox_inches="tight")
        plt.show()
        axs = fig.axes

        return fig, axs
