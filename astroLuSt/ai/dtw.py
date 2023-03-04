

class DTW:
    #TODO: DTW, correct for wrong assignment of high correlation
    """
        - class for executing Dynamic Time Warping
        - makes a prediction based on several template-curves (X_template)
            - prediction made via majority voting

        Attributes
        ----------
            - X_template
                - list
                - contains arrays of template time-series
                    - those act as role models, to which new samples will be compared
                    - can have different leghts
            - threshold
                - float
                - a classification threshold
                    - optimal warping path which has a coe
            - window
                - int, optional
                - locality-constraint for the distance determination
                - i.e. a distance between x1[i] and x2[j] is not allowed to be larger than the window parameter
                - the default is None
            - cost_fct
                - callable, optional
                - cost function to use for the calculation
                    - calculates the "distance between two points"
                - the default is None
                    - Will use the euclidean distance

        Methods
        -------
            - accumulate_cost_matrix
                - method to determine a distance matrix for two arrays
                    - x1 and x2 can have different lengths
                - implementation similar to Silva et al. (2016)
                    - DOI:https://doi.org/10.1137/1.9781611974348.94
                    - https://epubs.siam.org/doi/abs/10.1137/1.9781611974348.94
            - optimal_warping_path
                - computes the optimal warping path given a cost matrix
                    - Based on Senin (2008)
                    - https://www.researchgate.net/publication/228785661_Dynamic_Time_Warping_Algorithm_Review
            - fit_predict
                - fits the classifier and makes a prediction
            - summary_plot
                - function to display a brief summary plot of the DTW-result for two timeseries

    """

    def __init__(self, X_template, threshold=0.9, window=None, cost_fct=None, y_template=None):
        import numpy as np
        
        assert -1 <= threshold and threshold <= 1, f"'theshold' has to be in the range [-1,1] but has a value of {threshold}"
        try:
            len(X_template[0])
        except:
            raise ValueError(f"'X_template' has to be a list of lists!")

        self.X_template = X_template
        self.y_template = y_template
        self.threshold = threshold
        self.window = window
        #initialize cost_function
        self.cost_fct = cost_fct
        if cost_fct is None:
            self.cost_fct = lambda x, y:  np.abs(x-y)
        if y_template is None:
            self.y_template = np.arange(len(self.X_template))  #auto-generate class labels, if none are provided
        self.pearsons = None
        self.y_pred = None
    
    def __repr__(self):
        return ("DTW(\n"
                f"    X_template = {self.X_template},\n"
                f"    threshold  = {self.threshold},\n"
                f"    window     = {self.window},\n"
                f"    cost_fct   = {self.cost_fct},\n"
                f"    y_template = {self.y_template},\n"
                ")\n")


    def accumulate_cost_matrix(self, x1, x2, testplot=False):
        """
            - method to determine a distance matrix for two arrays
                - x1 and x2 can have different lengths
            - implementation similar to Silva et al. (2016)
                - DOI:https://doi.org/10.1137/1.9781611974348.94
                - https://epubs.siam.org/doi/abs/10.1137/1.9781611974348.94
            
            Paramters
            ---------
                - x1
                    - np.array
                    - some time series
                - x2
                    - np.array
                    - some time series
                - testplot
                    - bool, optional
                    - whether to display a testplot of the result
                    - the default is False
            Raises
            ------

            Returns
            -------
                - C
                    - np.array
                    - 2D array of the cost-matrix

            Dependencies
            ------------
                - numpy
            
            Comments
            --------


        """
        
        import numpy as np

        #initialize window
        if self.window is None:
            window = np.max([len(x1), len(x2)])-1
        else:
            window = self.window

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

        #plot
        if testplot:
            self.summary_plot()


        return C

    def optimal_warping_path(self, C, testplot=False):
        """
            - computes the optimal warping path given a cost matrix
                - Based on Senin (2008)
                    - https://www.researchgate.net/publication/228785661_Dynamic_Time_Warping_Algorithm_Review


            Parameters
            ----------
                - C
                    - np.array
                    - 2D array of the cost-matrix            
                - testplot
                    - bool, optional
                    - whether to display a testplot of the result
                    - the default is False

            Raises
            ------

            Returns
            -------
                - path
                    - np.array
                    - contains tuples
                        - indices of the cost-matrix C
                        - the tuples are the optimal warping path
                        - the tuples contain the indices of the best corresponding points from both timeseries
                            - i.e. a tuple (0,3) means that the zeroth element of the first timeseries best corresponds to the third element in the second timeseries

            Dependencies
            ------------
                - numpy
            
            Comments
            --------

        """
        import numpy as np


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

        if testplot:
            self.summary_plot()

        return path

    def fit_predict(self, X, expand_prediction=False, testplot=False, timeit=False):
        #TODO: Corrcoeff not great (if really bad match, no shift helps => corr in cost mat = 1, which is wrong!!)
        """
            - function to fit the classifier and make a prediction
            - prediction based on a majority vote from 'X_template'

            Parameters
            ----------
                - X
                    - np.array
                    - 2D array 
                    - design matrix
                        - contains time-series (of potentially different lengths) as samples
                - expand_prediction
                    - bool, optional
                    - if True will return the prediction for each time-series in 'X_template'
                    - otherwise a majority vote of all the predictions will be returned
                        - multiple classes if multiple classes had the same number of votes
                    - the default is False
                - testplot
                    - bool, optional
                    - whether to display the testplots of the results
                    - the default is False
                - timeit
                    - bool, optional
                    - whether to time the execution
                    - the default is False

            Raises
            ------

            Returns
            -------
                - y_pred
                    - list of lists
                    - the predicted classes based on a majority vote of the predictions w.r.t. all time-series in 'X_template' will be returned
                        - multiple classes if multiple classes had the same number of votes
                    - if 'expand_prediction' is set to True, the prediction for every time-series in 'X_template' will be returned
                        - in this case y_pred will have the shape '(X.shape[0], X_template.shape[0])'
                - Cs
                    - np.array
                    - 3D array of the cost-matrices
                - paths
                    - np.array
                    - 2D array
                    - contains tuples as individual entries
                        - indices of the cost-matrix C
                        - the tuples are the optimal warping path
                - pearsons
                    - list
                    - contains floats
                    - can take values in the range [-1, 1]
                    - the pearson correlation coefficients for paths
                    - i.e. the similarity between the two respective curves
                        - high similarity for 'pearson' close to 1
                        - low similarity otherwise
            
            Dependencies
            ------------
                - numpy
                - astroLuSt

            Comments
            --------


        """

        import numpy as np

        #initialize return-lists
        y_pred = []
        Cs = []
        pearsons = []
        paths = []

        #run fit for every sample in X
        for x in X:
            y_pred_sample = []  #list to save prediction for each template-time-series

            #compare sample to every template-time-series
            for xt, yt in zip(self.X_template, self.y_template):

                #fitting procedure
                C = self.accumulate_cost_matrix(x, xt, testplot=False)
                path = self.optimal_warping_path(C, testplot=False)
                pearson = np.corrcoef(np.array(path).T)[0,1]
                
                #append results
                Cs.append(C)
                pearsons.append(pearson)
                paths.append(path)

                #plotting
                if testplot:
                    self.summary_plot(C, x, xt, path=path, pearson=pearson, save=False)

                #append if curves are more similar than threshold
                if pearson > self.threshold:
                    y_pred_sample.append(yt)
                    
                    #TODO: correct for wrong assignment of good correlation
                    path_cost = C[tuple(np.array(path).T)]
                    print(path_cost.min(), path_cost.max(), (path_cost.max()-path_cost.min())/path_cost.max())
                else:
                    #append sample "not" similar to this curve
                    y_pred_sample.append("~"*expand_prediction+str(yt)) #only append "not" if expand_prediction is True
            

            #append prediction for every timeseries in X_template
            if expand_prediction:
                y_pred.append(y_pred_sample)
            
            #execute majority voting
            else:
                uniques, counts = np.unique(y_pred_sample, return_counts=True)
                y_pred.append(list(uniques[(counts==counts.max())]))

        #update attributes
        self.y_pred = y_pred
        self.pearson = pearsons

        if timeit:
            timer.end_task()

        return y_pred, Cs, paths, pearsons

    def summary_plot(self, C, x1, x2, path=None, pearson=None, save=False):
        """
            - function to display a brief summary plot of the DTW-result

            Parameters
            ----------
                - save
                    - str, optional
                    - location of where to save the plot to
                    - the default is False

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure object
                - axs
                    - list of matpotlib axes object

            Dependencies
            ------------
                - numpy
                - matplotlib
            
            Comments
            --------

        """
        import matplotlib.pyplot as plt
        import numpy as np

        if pearson is None:
            tit = None
        else:
            tit = f"corr:  {pearson:.3f}"

        cx = np.arange(0, C.shape[1]-1, 1)
        cy = np.arange(0, C.shape[0]-1, 1)
        cxx, cyy = np.meshgrid(cx,cy)

        fontsize=16
        fig = plt.figure(figsize=(9,9))
        axleg = fig.add_subplot(4,4,10, frameon=False)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(426)

        ax1.set_title("Cost-Matrix", fontsize=fontsize+4)
        
        #plot
        contour = ax1.contourf(cxx, cyy, C[1:,1:], zorder=1)    #exclude np.inf row and column
        c1, = ax2.plot(x1, cy, color="tab:blue", label="x1")
        c2, = ax3.plot(cx, x2, color="tab:orange", label="x2")
        if path is not None:
            path = np.array(path) - np.array([1,1])             #adjust path to ignore first row and column
            owp, = ax1.plot(path[:,1], path[:,0], color="r", zorder=2, label="Optimal\nWarping Path")
            ax1.plot(path[:,1], path[:,0], color="w", linewidth=5, zorder=1)
            handles = [c1, c2, owp]
        else:
            handles = [c1, c2]

        #add colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.53, 0.05, 0.35])
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label("Cost", fontsize=fontsize)
        cbar.ax.tick_params("both", labelsize=fontsize-4)

        #hide ticks of cost-matrix
        ax1.set_xticks([])
        ax1.set_yticks([])

        #rotate labels, adjust ticklabelsizes
        ax1.tick_params("both", labelsize=fontsize)
        ax2.tick_params("both", rotation=-90, labelsize=fontsize)
        ax3.tick_params("both", rotation=0, labelsize=fontsize)
        
        #invert axes accordingly
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        # ax1.set_aspect(1/ax1.get_data_ratio())    #ensure square subplot
        
        #Position ticks
        ax3.yaxis.tick_right()
        ax2.xaxis.tick_top()

        #push graph to box-boundary
        ax2.margins(y=0, tight=True)
        ax3.margins(x=0, tight=True)

        #add legend
        axleg.tick_params(labelcolor="none", which="both", labelsize=fontsize, top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
        leg = axleg.legend(handles=handles, fontsize=fontsize-4, title=tit, title_fontsize=fontsize-4, loc="best")
        leg._legend_box.align = "left"

        #reduce white space between plots
        plt.subplots_adjust(wspace=0, hspace=0)

        # plt.tight_layout()
        #save
        if type(save) == str:
            plt.savefig(save, dpi=180, bbox_inches="tight")
        plt.show()

        axs = fig.axes

        return fig, axs

