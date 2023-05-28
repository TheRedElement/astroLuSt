
#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import scipy.stats as stats
from typing import Union, Tuple

from astroLuSt.visualization.plots import CornerPlot

#%%definitions
class MLE:
    """
    
        - Class to execute a simple Maximum Likelihood Estimate on some data following a Gaussian distibution

        Attributes
        ----------
        
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
        ) -> np.ndarray:
        """
            - method to estimate the mean of the gaussian via MLE

            Parameters
            ----------
                - X
                    - np.ndarray
                    - input dataset
                    - contains samples as rows and features as columns

            Raises
            ------

            Returns
            -------
                - mus
                    - np.ndarray
                    - estimated means

            Comments
            --------
        """
        
        mus = np.nanmean(X, axis=0)

        return mus
    
    def get_sigma(self,
        X:np.ndarray
        ) -> np.ndarray:
        """
            - method to estimate the standard deviation of the gaussian via MLE
            - estimate for 1d-case (i.e. 1D histogram)
            
            Parameters
            ----------
                - X
                    - np.ndarray
                    - input dataset
                    - contains samples as rows and features as columns
            Raises
            ------

            Returns
            -------
                - sigmas
                    - np.ndarray
                    - standard deviation for every feature

            Comments
            --------
        """

        sigmas = np.nanstd(X, axis=0)

        return sigmas
    
    def get_covmat(self,
        X:np.ndarray=None
        ) -> np.ndarray:
        """
            - method to estimate the N-D covariance matrix

            Parameters
            ----------
                - X
                    - np.ndarray
                    - input dataset
                    - contains samples as rows and features as columns
                
            Raises
            ------

            Returns
            -------
                - self.covmat
                    - np.ndarray
                    - covariance matrix for a given set of datasets ('data')
                
            Comments
            --------

        """

        #N-D data
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
                - X
                    - np.ndarray
                    - input dataset
                    - contains samples as rows and features as columns
                - y
                    - np.ndarray, optional
                    - labels corresponding to X
                    - not needed for actual calculation
                        - only used for plotting
                    - the default is None

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
        self.mus = self.get_mu(X)
        self.sigmas = self.get_sigma(X)
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
                - X
                    - np.ndarray, optional
                    - input dataset
                    - contains samples as rows and features as columns
                    - not needed for prediction
                    - the default is None
                - y
                    - np.ndarray, optional
                    - labels corresponding to X
                    - not needed for prediction
                        - only used for plotting
                    - the default is None

            Raises
            ------

            Returns
            -------
                - self.mus
                    - np.ndarray
                    - array containing the mean of the gaussians
                - self.sigmas
                    - np.ndarray
                    - array containing the standard-deviations of the gaussians
                - self.covmat
                    - np.ndarray
                    - covariance matrix of the input data

            Comments
            --------
            
        """


        return self.mus, self.sigmas, self.covmat

    def fit_predict(self,
        X:np.ndarray, y:np.ndarray=None):
        """
            - method to fit the estimator and predict the result afterwards
            
            Parameters
            ----------
                - X
                    - np.ndarray
                    - input dataset
                    - contains samples as rows and features as columns
                - y
                    - np.ndarray, optional
                    - labels corresponding to X
                    - not needed for actual calculation
                        - only used for plotting
                    - the default is None

            Raises
            ------

            Returns
            -------
                - mus
                    - np.ndarray
                    - array containing the mean of the gaussians
                - sigmas
                    - np.ndarray
                    - array containing the standard-deviations of the gaussians
                - covmat
                    - np.ndarray
                    - covariance matrix of the input data

            Comments
            --------
   
        """

        self.fit(X=X, y=y)
        mus, sigmas, covmat = self.predict()

        return mus, sigmas, covmat

    def plot_result(self,
        featurenames:np.ndarray=None,
        corner_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the result generated by MLE

            Parameters
            ----------
                - featurenames
                    - np.ndarray, optional
                    - names to give to the features present in self.X
                    - the deafault is None
                - corner_kwargs
                    - dict, optional
                    - kwargs to pass to astroLuSt.visualization.plots.CornerPlot().plot()
                    - the default is None
                        - will initialize with {}

            Raises
            ------

            Returns
            -------
                - fig
                    - Figure
                    - the created matplotlib figure
                - axs
                    - plt.Axes
                    - axes corresponding to 'fig'
                    
            Comments
            --------
        """

        if corner_kwargs is None: corner_kwargs = {}

        CP = CornerPlot()
        fig, axs = CP.plot(
            self.X, self.y, featurenames=featurenames,
            mus=self.mus, sigmas=self.sigmas, corrmat=self.corrcoeff,
            **corner_kwargs,
        )

        return fig, axs

