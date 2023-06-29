
#%%imports
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple, List

from astroLuSt.monitoring.timers import ExecTimer

#%%definitions
class DTW:
    #TODO: DTW, correct for wrong assignment of high correlation
    """
        - class for executing Dynamic Time Warping
        - makes a prediction based on several template-curves (`X_template`)

        Attributes
        ----------
            - `X_template`
                - np.ndarray
                - contains arrays of template data-series
                    - act as role models, to which new samples will be compared
                    - can have different lengths
            - `y_template`
                - np.ndarray, optional
                - array of labels corresponding to `X_template`
                - relevant in `self.predict()` if `threshold is not None`
                - the default `None`
                    - will generate a unique label for every sampel in `X_template`
            - `window`
                - int, optional
                - locality-constraint for the distance determination
                - i.e. a distance between the points `X[i,j]` and `X_template[k,l]` is not allowed to be larger than the window parameter
                    - `X` is hereby the training dataset
                    - `X_template` is the template dataset
                    - `i`, `k` are sample-indices
                    - `j`, `l` are feature-indices
                - the default is `None`
                    - allows any distance
            - `cost_fct`
                - callable, optional
                - cost function to use for the calculation
                    - calculates the distance between two points
                - has to take two arguments
                    - `x`
                        - float
                        - feature-value 1
                    - `y`
                        - float
                        - feature-value 2
                - the default is `None`
                    - Will use the euclidean distance
            - `threshold`
                - float, optional
                - a classification threshold
                    - optimal warping path which has an absolute correlation coefficient (|r_P|) higher than `threshold` will be classified as being of the same type as the template-curve
                - has to be value in the interval [0,1]
                - the default is `None`
                    - will output correlations instead of binary classification during prediction

        Methods
        -------
            - `accumulate_cost_matrix()`
            - `optimal_warping_path()`
            - `fit()`
            - `predict()`
            - `fit_predict()`
            - `plot_result()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing
        
        Comments
        --------

    """

    def __init__(self,
        X_template:np.ndarray, y_template:np.ndarray=None,
        window:int=None, cost_fct:Callable=None,
        threshold:float=None,
        ) -> None:
        
        if threshold is not None: assert 0 <= threshold and threshold <= 1, f"`theshold` has to be in the interval [0,1] but is {threshold}"
        try:
            len(X_template[0])
        except:
            raise ValueError(f"`X_template` has to be a list of np.ndarrays!")

        self.X_template = X_template
        if y_template is None: self.y_template = np.arange(len(self.X_template)) #unique labels for every template sample
        else:                  self.y_template = y_template
        if cost_fct is None:   self.cost_fct = lambda x, y: np.abs(x-y)
        else:                  self.cost_fct = cost_fct
        self.threshold  = threshold
        self.window     = window


        return
    
    def __repr__(self) -> str:
        return (
            f'DTW(\n'
            f'    X_template={repr(self.X_template)}, y_template={repr(self.y_template)},\n'
            f'    threshold={repr(self.threshold)}, window={repr(self.window)}, cost_fct={repr(self.cost_fct)},\n'
            f')'
        )


    def accumulate_cost_matrix(self,
        x1:np.ndarray, x2:np.ndarray,
        window:int=None, cost_fct:Callable=None
        ) -> np.ndarray:
        """
            - method to determine a distance matrix for two arrays `x1` and `x2`
                - `x1` and `x2` can have different lengths
            - implementation similar to Silva et al. (2016)
                - DOI:https://doi.org/10.1137/1.9781611974348.94
                - https://epubs.siam.org/doi/abs/10.1137/1.9781611974348.94
            
            Paramters
            ---------
                - `x1`
                    - np.ndarray
                    - some 1D data series
                - `x2`
                    - np.ndarray
                    - some 1D data series
                - `window`
                    - int, optional
                    - locality-constraint for the distance determination
                    - i.e. a distance between the points `x1[j]` and `x2[l]` is not allowed to be larger than the window parameter
                        - `j`, `l` are feature-indices
                    - overrides `self.window`
                    - the default is `None`
                        - will fallback to `self.window`
                        - if that is `None` as well
                            - allows any distance
                - `cost_fct`
                    - callable, optional
                    - cost function to use for the calculation
                        - calculates the distance between two points
                    - has to take two arguments
                        - `x`
                            - float
                            - feature-value 1
                        - `y`
                            - float
                            - feature-value 2
                    - overrides `self.cost_fct`
                    - the default is `None`
                        - will default to `self.cost_fct`
            
            Raises
            ------

            Returns
            -------
                - `C`
                    - np.ndarray
                    - 2D array
                    - cost-matrix of the differences between `x1` and `x2`

            Comments
            --------

        """
        
        #default values
        ##initialize window
        if window is None:
            #fallback to self.window
            if self.window is not None: window = self.window
            #default value if that is also None
            else:                       window = np.max([len(x1), len(x2)])-1
        ##cost function to use
        if cost_fct is None:            cost_fct = self.cost_fct

        #get specification of time-series
        n, m = len(x1), len(x2)
        w = np.max([window, abs(n-m)])
        
        #initialize cost matrix
        C = np.zeros((n+1, m+1)) + np.inf        
        C[0,0] = 0

        #fill cost matrix
        for i in range(1, n+1):
            for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
                cost = self.cost_fct(x1[i-1], x2[j-1])
                last_min = np.min([
                    C[i-1,j],
                    C[i,j-1],
                    C[i-1,j-1]
                ])
                C[i,j] = cost + last_min

        return C

    def optimal_warping_path(self,
        C:np.ndarray,
        ) -> np.ndarray:
        """
            - method to compute the optimal warping path given a cost matrix
                - Based on Senin (2008)
                    - https://www.researchgate.net/publication/228785661_Dynamic_Time_Warping_Algorithm_Review


            Parameters
            ----------
                - `C`
                    - np.ndarray
                    - 2D array
                    - cost-matrix of the differences between `X[i]` and `X_template[k]`
                        - `X` is hereby the training dataset
                        - `X_template` is the template dataset
                        - `i`, `k` are sample-indices

            Raises
            ------

            Returns
            -------
                - `path`
                    - np.ndarray
                    - contains indices of the cost-matrix `C`
                        - the indices denote the optimal warping path
                        - the indices contain the indices of the best corresponding points from both data series
                            - i.e. an entry `[0,3]` means that the zeroth element of the first data series best corresponds to the third element in the second data series

            Comments
            --------

        """

        path = []
        i, j = C.shape[0]-1, C.shape[1]-1

        #iterate over all elements
        while i > 0 and j > 0:
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                if C[i-1,j] == np.min([C[i-1,j], C[i,j-1], C[i-1,j-1]]):
                    i -= 1
                elif C[i,j-1] == np.min([C[i-1,j], C[i,j-1], C[i-1,j-1]]):
                    j -= 1
                else:
                    i -= 1
                    j -= 1
                path.append((i,j))

        path = np.array(path)

        return path


    def fit(self,
        X:np.ndarray, y:np.ndarray=None,
        window:int=None, cost_fct:Callable=None,
        ) -> None:
        """
            - method to fit the classifier

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                - `y`
                    - np.ndarray, optional
                    - labels corresponding to `X`
                    - not used in the method
                    - the default is `None`
                - `window`
                    - int, optional
                    - locality-constraint for the distance determination
                    - i.e. a distance between the points `x1[j]` and `x2[l]` is not allowed to be larger than the window parameter
                        - `j`, `l` are feature-indices
                    - overrides `self.window`
                    - the default is `None`
                        - will fallback to `self.window`
                - `cost_fct`
                    - callable, optional
                    - cost function to use for the calculation
                        - calculates the distance between two points
                    - has to take two arguments
                        - `x`
                            - float
                            - feature-value 1
                        - `y`
                            - float
                            - feature-value 2
                    - overrides `self.cost_fct`
                    - the default is `None`
                        - will default to `self.cost_fct`                        

            Raises
            ------

            Returns
            -------

            Comments
            --------

        """

        #default parameters
        if window is None: window = self.window
        if cost_fct is None: cost_fct = self.cost_fct
        

        #initialize result arrays
        self.Cs = np.empty((len(X), len(self.X_template)), dtype=object)
        self.pearsons = np.empty((len(X), len(self.X_template)), dtype=object)
        self.paths = np.empty((len(X), len(self.X_template)), dtype=object)

        #run fit for every sample in X
        for iidx, x in enumerate(X):

            #compare sample to every template-time-series
            for jidx, xt in enumerate(self.X_template):

                #fitting procedure
                C = self.accumulate_cost_matrix(x, xt, cost_fct=cost_fct)
                path = self.optimal_warping_path(C)
                pearson = np.abs(np.corrcoef(path.T)[0,1])
                
                #append results
                self.Cs[iidx, jidx] = C
                self.pearsons[iidx, jidx] = pearson
                self.paths[iidx, jidx] = path

        return

    def predict(self,
        X:np.ndarray=None, y:np.ndarray=None,
        threshold:float=None, multi_hot_encoded:bool=False
        ) -> np.ndarray:
        """
            - method to predict with the fitted classifier
            
            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - not used in the method
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                - `y`
                    - np.ndarray, optional
                    - labels corresponding to `X`
                    - not used in the method
                    - the default is `None`
                - `threshold`
                    - float, optional
                    - a classification threshold
                        - optimal warping path which has an absolute correlation coefficient (|r_P|) higher than `threshold` will be classified as being of the same type as the template-curve
                    - has to be value in the interval [0,1]
                    - overrides `self.threshold`
                    - the default is `None`
                        - will fall back to `self.threshold`
                - `multi_hot_encoded`
                    - bool, optional
                    - whether to convert the calculated correlations to a multi-hot encoded matrix
                        - will consider every unique class in `self.y_template`
                        - will calculate the overall correlation trend w.r.t. `threshold` for all template-curves
                        - i.e. computes $\sum_{class} r_P(class) - threshold$ for every sample in `X` and every unique class
                            - if the the result is > 0 this means that (w.r.t. `threshold`) there is a correlation
                                - Hence a 1 will be assigned for that class
                            - if the the result is < 0 this means that (w.r.t. `threshold`) there is no correlation
                                - Hence a 0 will be assigned for that class
                    - only relevant if `threshold` and `self.threshold` are not `None`
                    - the default is False
                        - will return the overall correlation trend instead
                        - i.e. $\sum_{class} r_P(class) - threshold$

            Raises
            ------

            Returns
            -------
                - `y_pred`
                    - np.ndarray
                    - array containing the labes for `X`
                    - has shape `(X.shape[0],self.X_template.shape[0])`
                    - returns matrix of 0 and 1 if `multi_hot_encoded == True`
                        - 1 if the overall correlation exceeds `threshold`
                        - 0 otherwise
                    - the second axis of `y_pred` will be sorted in ascending order w.r.t. the classlabels if `multi_hot_encoded == True`
                    - otherwise returns the overall correlation as entries

            Comments
            --------
        """

        if threshold is None: threshold = self.threshold

        if threshold is None:
            y_pred = self.pearsons
        else:
            uniques = np.sort(np.unique(self.y_template))
            y_pred = np.empty((self.pearsons.shape[0], uniques.shape[0]))
            for idx, yi in enumerate(uniques):
                #get pearsons per class
                class_pearsons = self.pearsons[:,(yi==self.y_template)]
                #get difference of correlation and threshold (samples below threshold will be negative, samples above will be positive)
                diff = (class_pearsons - threshold)

                #sum to check what to overall correlation trend is (part of class if > 0,  not part of class if < 0)
                y_pred[:,idx] = diff.sum(axis=1)
        
            #define that sample is part of class, if the overall correlation exceeds threshold (i.e. more positive values in diff than negative ones)
            if multi_hot_encoded:
                y_pred = (y_pred>0).astype(int)


        return y_pred

    def fit_predict(self,
        X:np.ndarray, y:np.ndarray=None,
        fit_kwargs:dict=None, predict_kwargs:dict=None
        ):
        """
            - method to fit the classifier and make the prediction at the same time

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - not used in the method
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                - `y`
                    - np.ndarray, optional
                    - labels corresponding to `X`
                    - not used in the method
                    - the default is `None`
                - `fit_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `{}`
                - `predict_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.predict()`
                    - the default is `None`
                        - will be set to `{}`
            
            Raises
            ------

            Returns
            -------
             - `y_pred`
                    - np.ndarray
                    - array containing the labes for `X`
                    - has shape `(X.shape[0],self.X_template.shape[0])`
                    - returns matrix of 0 and 1 if `multi_hot_encoded == True`
                        - 1 if the overall correlation exceeds `threshold`
                        - 0 otherwise
                    - the second axis of `y_pred` will be sorted in ascending order w.r.t. the classlabels if `multi_hot_encoded == True`
                    - otherwise returns the overall correlation as entries

            Comments
            --------

        """

        if fit_kwargs is None:      fit_kwargs = {}
        if predict_kwargs is None:  predict_kwargs = {}

        self.fit(X, y, **fit_kwargs)
        y_pred = self.predict(X, y, **predict_kwargs)

        return y_pred

    def plot_result(self,
        X:list,
        X_idx:int=0, Xtemp_idx:int=0,
        ):
        """
            - method to display a brief summary plot of the DTW-result for an example combination of template- and train- curve

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                    - should be same array as used to fit the classifier
                - `X_idx`
                    - int, optional
                    - index of the sample in `X` to plot
                    - the default is 0
                - `Xtemp_idx`
                    - int, optional
                    - index of the sample in `self.X_template` to plot
                    - the default is 0
            
            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure object
                - axs
                    - list of matpotlib axes object

            Comments
            --------

        """

        X_plot = X[X_idx]
        Xtemp_plot = self.X_template[Xtemp_idx]
        C = self.Cs[X_idx, Xtemp_idx]
        path = self.paths[X_idx, Xtemp_idx]
        pearson = self.pearsons[X_idx, Xtemp_idx]

        # if pearson is None: tit = None
        # else: tit = f"corr:  {pearson:.3f}"

        #adjust path to ignore first row and column
        path = np.array(path) - np.array([1,1])

        cx = np.arange(0, C.shape[1]-1, 1)
        cy = np.arange(0, C.shape[0]-1, 1)
        cxx, cyy = np.meshgrid(cx,cy)

        fig = plt.figure(figsize=(9,9))
        axleg = fig.add_subplot(4,4,10, frameon=False)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(426)

        ax1.set_title("Cost-Matrix")
        
        #plot
        contour = ax1.contourf(cxx, cyy, C[1:,1:], zorder=1)    #exclude np.inf row and column
        c1,     = ax2.plot(X_plot, cy,         color="tab:blue",   label=f'X[{X_idx}]')
        c2,     = ax3.plot(cx,     Xtemp_plot, color="tab:orange", label=f'X_template[{Xtemp_idx}]')
        owp,    = ax1.plot(path[:,1], path[:,0], color="r", zorder=2, label="Optimal\nWarping Path")
        ax1.plot(path[:,1], path[:,0], color="w", linewidth=5, zorder=1)
        handles = [c1, c2, owp]

        #add colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.53, 0.05, 0.35])
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label("Cost")

        #hide ticks of cost-matrix
        ax1.set_xticks([])
        ax1.set_yticks([])

        #rotate labels, adjust ticklabelsizes
        ax2.tick_params("both", rotation=-90)
        ax3.tick_params("both", rotation=0)
        
        #invert axes accordingly
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        #Position ticks
        ax3.yaxis.tick_right()
        ax2.xaxis.tick_top()

        #push graph to box-boundary
        ax2.margins(y=0, tight=True)
        ax3.margins(x=0, tight=True)

        #add legend
        axleg.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
        leg = axleg.legend(handles=handles, title=r'$|r_P| =$%-4g'%pearson, loc="best")
        leg._legend_box.align = "left"

        #reduce white space between plots
        plt.subplots_adjust(wspace=0, hspace=0)

        axs = fig.axes

        return fig, axs

