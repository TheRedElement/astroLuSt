
#%%imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

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
    
    """


    def __init__(self,
        dataseries:np.ndarray, series_labels:np.ndarray=None,
        ):

        self.dataseries = dataseries
        if series_labels is None:
            self.series_labels = [f"Dataseries {i}" for i in np.arange(self.dataseries.shape[0])]
        else:
            self.series_labels = series_labels
        self.mus = np.array([])
        self.sigmas = np.array([])
        self.covmat = np.array([])
        self.corrcoeff = np.array([])

        pass

    def get_mu(self):
        """
            - method to estimate the mean of the gaussian via MLE
        """
        
        self.mus = np.nanmean(self.dataseries, axis=0)

        return
    
    def get_sigma(self):
        """
            - method to estimate the standard deviation of the gaussian via MLE
            - estimate for 1d-case (i.e. 1D histogram)
        """

        self.sigmas = np.nanstd(self.dataseries, axis=0)

        return
    
    def get_covmat(self, data=None):
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
        if data is None:
            data = self.dataseries
        self.covmat = np.cov(data.T)

        return self.covmat

    def fit(self):
        """
            - method to fit the estimator
            - similar to scikit-learn framework
        """
        self.get_mu()
        self.get_sigma()
        self.covmat = self.get_covmat()
        self.corrcoeff = np.corrcoef(self.dataseries.T)

        return

    def predict(self):
        """
            - method to predict using the estimator
            - similar to scikit-learn framework

            Returns
            -------
                - self.mus
                - self.sigmas
                - self.convmat
            
        """


        return self.mus, self.sigmas, self.covmat

    def fit_predict(self):
        """
            - method to fit the estimator and predict the result afterwards
            
            Returns
            -------
                - self.mus
                - self.sigmas
                - self.convmat        
        """

        self.fit()
        self.predict()

        return self.mus, self.sigmas, self.covmat

    def corner_plot(self,
        data=None, labels=None, mus=None, sigmas=None,
        bins=100, equal_range=False,
        save=False,
        fontsize=16, figsize=(9,9)):
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
                - save
                    - str, optional
                    - location of where to save the created figure
                    - the default is None
                        - will not save the figure
                - fontsize
                    - int, optional
                    - fontsize to use for the plot
                    - the default si 16
                - figsize
                    - tupel, optional
                    - figure dimensions
                    - the default is (9,9)

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


        if data is None:
            data = self.dataseries
        if labels is None:
            labels = self.series_labels
        if mus is None:
            if len(self.mus) == 0:
                mus = [None]*len(data)
            else:
                mus = self.mus
        if sigmas is None:
            if len(self.sigmas) == 0:
                sigmas = [None]*len(data)
            else:
                sigmas = self.sigmas
        

        fig = plt.figure(figsize=figsize)
        nrowscols = data.shape[1]


        idx = 0
        for idx1, (d1, l1, mu1, sigma1) in enumerate(zip(data.T, labels, mus, sigmas)):
            for idx2, (d2, l2, mu2, sigma2) in enumerate(zip(data.T, labels, mus, sigmas)):
                idx += 1
                
                #plotting 2D distributions
                if idx1 > idx2:

                    ax1 = fig.add_subplot(nrowscols, nrowscols, idx)
                    
                    if mu1 is not None: ax1.axhline(mu1, color="tab:orange", linestyle="--")
                    if mu2 is not None: ax1.axvline(mu2, color="tab:orange", linestyle="--")

                    #data
                    sctr = ax1.scatter(d2, d1,
                        s=1, alpha=.5, zorder=2,
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
                        )
                        cont = ax1.contour(yy, xx, norm.pdf(mesh), cmap="gray", zorder=1)

                    #labelling
                    if idx1 == nrowscols-1:
                        ax1.set_xlabel(l2, fontsize=fontsize)
                    else:
                        ax1.set_xticklabels([])
                    if idx2 == 0:
                        ax1.set_ylabel(l1, fontsize=fontsize)
                    else:
                        ax1.set_yticklabels([])
                    ax1.tick_params(labelsize=fontsize)

                    if not equal_range:
                        ax1.set_xlim(np.nanmin(d2), np.nanmax(d2))
                        ax1.set_ylim(np.nanmin(d1), np.nanmax(d1))

                    if len(self.corrcoeff.shape) == 2:
                        ax1.errorbar(np.nan, np.nan, color="none", label=r"$r_\mathrm{P}=%.4f$"%(self.corrcoeff[idx1, idx2]))
                        ax1.legend(fontsize=fontsize)


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
                        axhist.legend(fontsize=fontsize)
                    

                    axhist.tick_params(labelsize=fontsize)
        
        #make x and y limits equal if requested
        if equal_range:
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
