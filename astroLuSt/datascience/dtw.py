

#%%imports
import logging
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Callable, Tuple, Literal, Union

logger = logging.getLogger(__name__)
#%%definitions
class DTW:
    """
        - class for executing Dynamic Time Warping
        - makes a prediction based on several template-curves (`X_template`)

        Attributes
        ----------
            - `X_template`
                - `np.ndarray`
                - contains arrays of template data-series
                    - act as role models, to which new samples will be compared
                    - can have different lengths
            - `window`
                - `int`, optional
                - locality-constraint for the distance determination
                - i.e. a distance between the points `X[i,j]` and `X_template[k,l]` is not allowed to be larger than the window parameter
                    - `X` is hereby the training dataset
                    - `X_template` is the template dataset
                    - `i`, `k` are sample-indices
                    - `j`, `l` are feature-indices
                - the default is `None`
                    - allows any distance
            - `cost_fct`
                - `Callable`, optional
                - cost function to use for the calculation
                    - calculates the distance between two points
                - has to take two arguments
                    - `x`
                        - `float`
                        - feature-value 1
                    - `y`
                        - `float`
                        - feature-value 2
                - the default is `None`
                    - Will use the absolute of differnce between inputs

        Infered Attributes
        ------------------
            - `Cs`
                - `List[np.ndarray]`
                - has length `nsamples`
                - elements have shape `(K,L)`
                    - `K` ... length of input series
                    - `L` ... length of output series
                - cost matrices computed for all inputs
            - `dtw_loss`
                - `np.ndarray`
                - has shape `(nsamples,ntemplates)`
                - DTW-losses for all samples
            - `perp_dist`
                - `np.ndarray`
                - has shape `(nsamples,ntemplates)`
                - perpendicular distance for all samples
            - `path`
                - `List[np.ndarray]`
                - has length `nsamples`
                - elements have shape `(K,2)`
                    - `K` ... length of input series
                - optimal warping paths computed for all inputs
                - indices mapping to respective element in `Cs`

        Methods
        -------
            - `perpendicular_distance_2d()`
            - `accumulate_cost_matrix()`
            - `optimal_warping_path()`
            - `fit()`
            - `predict()`
            - `fit_predict()`
            - `plot_result()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`
        
        Comments
        --------

    """

    def __init__(self,
        X_template:np.ndarray,
        window:int=None, cost_fct:Callable=None,
        ) -> None:
        
        try:
            len(X_template[0])
        except:
            raise ValueError(f"`X_template` has to be a list of `np.ndarrays`!")

        self.X_template = X_template
        if cost_fct is None:   self.cost_fct = lambda x, y: np.abs(x-y)
        else:                  self.cost_fct = cost_fct
        self.window     = window


        return
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    X_template={repr(self.X_template)},\n"
            f"    window={repr(self.window)}, cost_fct={repr(self.cost_fct)},\n"
            f")"
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, "dict"))

    def perpendicular_distance_2d(self,
        x:np.ndarray, p1:tuple, p2:tuple, 
        ) -> np.ndarray:
        """
            - function to compute the perpendicular distance of `x` to some line passing through `p1`, and, `p2`

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - has to have shape `(nsamples,2)`
                    - set of test-points to compute the distance for
                - `p1`
                    - `tuple`
                    - coordinates of a point
                    - defines line to compute distance to together with `p2`
                - `p2`
                    - `tuple`
                    - coordinates of a point
                    - defines line to compute distance to together with `p1`
            
            Raises
            ------
            
            Returns
            -------
                - `dist`
                    - `np.ndarray`
                    - distances of all points in `x` to the line from `p1` to `p2`
            
            Comments
            --------
        """

        x0, y0 = x.T
        x1, y1 = p1
        x2, y2 = p2

        num = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        dist = num/den
        return dist

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
                    - `np.ndarray`
                    - some 1D data series
                - `x2`
                    - `np.ndarray`
                    - some 1D data series
                - `window`
                    - `int`, optional
                    - locality-constraint for the distance determination
                    - i.e. a distance between the points `x1[j]` and `x2[l]` is not allowed to be larger than the window parameter
                        - `j`, `l` are feature-indices
                    - overrides `self.window`
                    - the default is `None`
                        - will fallback to `self.window`
                        - if that is `None` as well
                            - allows any distance
                - `cost_fct`
                    - `Callable`, optional
                    - cost function to use for the calculation
                        - calculates the distance between two points
                    - has to take two arguments
                        - `x`
                            - `float`
                            - feature-value 1
                        - `y`
                            - `float`
                            - feature-value 2
                    - overrides `self.cost_fct`
                    - the default is `None`
                        - will default to `self.cost_fct`
            
            Raises
            ------

            Returns
            -------
                - `C`
                    - `np.ndarray`
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
            for j in range(max(1, i-w), min(m, i+w)+1):
                cost = cost_fct(x1[i-1], x2[j-1])
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
                - Based on [Senin (2008)](https://www.researchgate.net/publication/228785661_Dynamic_Time_Warping_Algorithm_Review)

            Parameters
            ----------
                - `C`
                    - `np.ndarray`
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
                    - `np.ndarray`
                    - contains indices of the cost-matrix `C`
                        - the indices denote the optimal warping path
                        - the indices contain the indices of the best corresponding points from both data series
                            - i.e. an entry `[0,3]` means that the zeroth element of the first data series best corresponds to the third element in the second data series
                    - `path[:,0]` corresponds to training sample indices
                    - `path[:,1]` corresponds to template indices

            Comments
            --------

        """

        path = []
        i, j = C.shape[0]-1, C.shape[1]-1

        #iterate over all elements from bottom right corner to top left
        while i > 0 or j > 0:
            path.append((i-1, j-1))

            #direction of the minimum previous cost
            choices = []
            if i > 0 and j > 0:
                choices.append([C[i-1, j-1], i-1, j-1]) #diagonal
            if i > 0:
                choices.append([C[i-1, j], i-1, j])     #up
            if j > 0:
                choices.append([C[i, j-1], i, j-1])     #left

            cost, i, j = min(choices)

        path.reverse()      #reverse to make forward path

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
                    - `np.ndarray`
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                - `y`
                    - `np.ndarray`, optional
                    - labels corresponding to `X`
                    - not used in the method
                    - the default is `None`
                - `window`
                    - `int`, optional
                    - locality-constraint for the distance determination
                    - i.e. a distance between the points `x1[j]` and `x2[l]` is not allowed to be larger than the window parameter
                        - `j`, `l` are feature-indices
                    - overrides `self.window`
                    - the default is `None`
                        - will fallback to `self.window`
                - `cost_fct`
                    - `Callable`, optional
                    - cost function to use for the calculation
                        - calculates the distance between two points
                    - has to take two arguments
                        - `x`
                            - `float`
                            - feature-value 1
                        - `y`
                            - `float`
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
        self.perp_dist = np.empty((len(X), len(self.X_template)), dtype=np.float64)
        self.path = np.empty((len(X), len(self.X_template)), dtype=object)
        self.dtw_loss = np.empty((len(X), len(self.X_template)), dtype=np.float64)

        #run fit for every sample in X
        for iidx, x in enumerate(X):

            #compare sample to every template-time-series
            for jidx, xt in enumerate(self.X_template):

                #fitting procedure
                C = self.accumulate_cost_matrix(x, xt, cost_fct=cost_fct)
                path = self.optimal_warping_path(C)

                #compute metrics
                ##perpendicular distance
                p1 = (path[:,1].min(),path[:,0].min())
                p2 = (path[:,1].max(),path[:,0].max())
                perp_dist = self.perpendicular_distance_2d(path[:,::-1], p1, p2).mean()
                
                ##dtw loss = total accumlated cost
                dtw_loss = C[-1,-1]
                
                #append results
                self.Cs[iidx, jidx] = C
                self.path[iidx, jidx] = path
                self.perp_dist[iidx, jidx] = perp_dist
                self.dtw_loss[iidx, jidx] = dtw_loss

        return

    def predict(self,
        X:np.ndarray=None, y:np.ndarray=None,
        method:Union[Literal["dtwloss"],None]=None,
        ) -> np.ndarray:
        """
            - method to predict with the fitted classifier
            
            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - not used in the method
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                - `y`
                    - `np.ndarray`, optional
                    - labels corresponding to `X`
                    - not used in the method
                    - the default is `None`
                - `method`
                    - `Literal["dtwloss"]`, optional
                    - method to use for the prediction
                    - if `dtwloss`
                        - will will use DTW loss
                    - otherwise
                        - will compute the perpendicular distance to the simplest line (diagonal)
                    - the default is `None`
                        - will use perpendicular distance

            Raises
            ------

            Returns
            -------
                - `y_pred`
                    - `np.ndarray`
                    - array containing metrics specified in `method` mapped to probabilities
                    - predictions for `X`
                    - has shape `(X.shape[0],self.X_template.shape[0])`

            Comments
            --------
        """

        #get metric to use
        if method == "dtwloss": class_metric = self.dtw_loss
        else:                   class_metric = self.perp_dist

        y_pred = np.empty((self.dtw_loss.shape[0], len(self.X_template)))
        for idx, _ in enumerate(self.X_template):
            cm = class_metric[:,idx]
            score = (-(cm / cm.sum()).flatten())    #convert metric to score (pseudo-probability, `-` to interpret lower loss as better score)
            y_pred[:,idx] = score / score.sum()     #convert score to probabilities

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
                    - `np.ndarray`
                    - not used in the method
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                - `y`
                    - `np.ndarray`, optional
                    - labels corresponding to `X`
                    - not used in the method
                    - the default is `None`
                - `fit_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `{}`
                - `predict_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.predict()`
                    - the default is `None`
                        - will be set to `{}`
            
            Raises
            ------

            Returns
            -------
                - `y_pred`
                    - `np.ndarray`
                    - array containing metrics specified in `method` mapped to probabilities
                    - predictions for `X`
                    - has shape `(X.shape[0],self.X_template.shape[0])`

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
        reference_line:bool=False,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to display a brief summary plot of the DTW-result for an example combination of template- and train- curve

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - can contain samples of different lengths
                    - training set to be compared to `self.X_template`
                    - should be same array as used to fit the classifier
                - `X_idx`
                    - `int`, optional
                    - index of the sample in `X` to plot
                    - the default is `0`
                - `Xtemp_idx`
                    - `int`, optional
                    - index of the sample in `self.X_template` to plot
                    - the default is `0`
                - `reference_line`
                    - `bool`, optional
                    - whether to show the reference line used for computation of `perp_dist`
                    - the default is `False`
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `Figure`
                    - matplotlib figure object
                - `axs`
                    - `plt.Axes`
                    - list of matpotlib axes object

            Comments
            --------

        """

        cur_bg = plt.rcParams["axes.facecolor"]

        X_plot = X[X_idx]
        Xtemp_plot = self.X_template[Xtemp_idx]
        C = self.Cs[X_idx, Xtemp_idx]
        path = self.path[X_idx, Xtemp_idx]
        perp_dist = self.perp_dist[X_idx, Xtemp_idx]
        dtw_loss = self.dtw_loss[X_idx, Xtemp_idx]

        #adjust path to ignore first row and column
        path = np.array(path)

        cx = np.arange(0, C.shape[1]-1, 1)
        cy = np.arange(0, C.shape[0]-1, 1)
        cxx, cyy = np.meshgrid(cx,cy)

        fig = plt.figure(figsize=(9,9))
        axleg = fig.add_subplot(4,4,10, frameon=False)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(426)
        ax1.set_xmargin(0)
        ax1.set_ymargin(0)
        
        ax1.set_title("Cost-Matrix")
        #plot
        contour = ax1.contourf(cxx, cyy, C[1:,1:], zorder=1)    #exclude np.inf row and column
        c1,     = ax2.plot(X_plot, cy,         color="C2",   label=f"X[{X_idx}]")
        c2,     = ax3.plot(cx,     Xtemp_plot, color="C1", label=f"X_template[{Xtemp_idx}]")
        owp,    = ax1.plot(path[:,1], path[:,0], color="C0", zorder=2, label="Optimal\nWarping Path")
        ax1.plot(path[:,1], path[:,0], color=cur_bg, linewidth=5, zorder=1)
        handles = [c1, c2, owp]

        if reference_line: #add reference line for perpendicular distance
            p1 = (path[:,1].min(),path[:,0].min())
            p2 = (path[:,1].max(),path[:,0].max())
            ax1.plot((p1[0],p2[0]), (p1[1],p2[1]), color="C0", ls="--")

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
        ax2.spines[["bottom"]].set_visible(False)
        ax2.spines[["top"]].set_visible(True)
        ax3.spines[["left"]].set_visible(False)
        ax3.spines[["right"]].set_visible(True)
        
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
        tit = (
            r"$\mathcal{L}_{d_\perp} =$%-4.4g"%perp_dist + "\n" + 
            r"$\mathcal{L}_\mathrm{DTW} = %.4g$"%dtw_loss
        )
        axleg.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
        leg = axleg.legend(handles=handles, title=tit, loc="best")
        leg._legend_box.align = "left"

        #reduce white space between plots
        plt.subplots_adjust(wspace=0, hspace=0)

        axs = fig.axes

        return fig, axs

