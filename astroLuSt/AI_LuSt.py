#TODO: create class to store functions

def plot_confusion_matrix(
    vals, classes,
    xlab="Class", ylab="Class", cbarlabel="", cmap="viridis",
    annotate=True, annotationcolor="w", textfontsize=None,
    figsize=(9,5), fontsize=16,
    save=False):
    """
        - function to plot a confusion matrix

        Parameters
        ----------
            - vals
                - np.array
                    - 2d array
                    - value assigned to each pair of classes
            - classes
                - list, np.array
                    - 1d array
                - classes the values got calculated for
            - xlab
                - str, optional
                - xaxis label
                - the default is 'Class'
            - ylab
                - str, optional
                - xaxis label
                - the default is 'Class'
            - cbarlab
                - str, optional
                - colorbar label
                - the default is ''
            - cmap
                - str, optional
                - matplotlib colormap to use
                - the default is 'viridis'
            - annotate
                - bool, optional
                - whether to depict the heatmap values in the individiual cells
                - the default is true
            - annotationcolor
                - str, optional
                - color of the heatmapvalues, when depicted in the cells
                - the default is "w"
            - annotationfontsize
                - float, optional
                - fontsize of the heatmapvalues, when depicted in the cells
                - the default is None
                    - will result in the same value as 'fontsize'
            - figsize
                - tuple, optional
                - size of the figure
                - the default is (9,5)
            - fontsize
                - float, optional
                - fontsize of labels
                - the default is 16
            - save
                - str, optional
                - location of where to save the plot
                - the default is False
                    - will not save the plot
        
        Raises
        ------

        Returns
        -------
            - fig
                - matplotlib figure object
            - ax
                - list
                - contains matplotlib axes objects

        Dependencies
        ------------
            - numpy
            - matplotlib

        Comments
        --------

    """

    import numpy as np
    import matplotlib.pyplot as plt

    if textfontsize is None:
        textfontsize = fontsize

    #generate tick positions
    tick_positions = np.arange(0, vals.shape[0], 1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(vals, cmap=cmap)
    cbar= fig.colorbar(im)

    #annotating
    if annotate:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                text = ax.text(j, i, f"{vals[i, j]:.1f}",
                            ha="center", va="center", color=annotationcolor, fontsize=textfontsize)

    #set locators
    ax.xaxis.set_major_locator(plt.MaxNLocator(vals.shape[0]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(vals.shape[0]))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    #fontsize
    ax.tick_params("both", labelsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    #labelling
    cbar.set_label(cbarlabel, fontsize=fontsize)
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)

    if type(save) == str:
        plt.savefig(save, dpi=180)

    return fig, ax


def kdist(X, k=4, eps_pred=None, testplot=False, saveplot=False):
    #TODO: Autodetermine kink-distance
    """
        - function to return the distance of every datapoint in "X" to its k-th nearest neighbour
        - useful for generating kdist-graphs (Ester et al., 1996)
            - kdist-plots are used to determing the epsilon environment for DBSCAN

        Parameters
        ----------
            - X
                - np.array
                - some database
                    - i.e. feature matrix
                - database to determine the distance of every point to its k-th neighbor
            - k
                - int, optional
                - which neighbor to calculate the distance to
                - the default is 4
                    - that is the standard value of 'npoints' used for DBSCAN (Ester et al., 1996)
            - epspred
                - float, optional
                - your prediction of the epsilon-environment
                - used for the plot to display an horizontal line at that 4-distance
                - the default is None
                    - Will try to estimate the point from the second derivative
                    - Very inaccurate!
            - testplot
                - bool, optional
                - whether to generate a kdist-plot using the caculated distances
                - the default is False
            - saveplot
                - str, optional
                - location of where to save the generated plot
                - the default is False

        Raises
        ------

        Returns
        -------
            - kth_dist
                - np.array
                - array of sorted kth-distances
            - eps_pred_aute
                - float
                - autogenerated prediction for the epsilon-environment
                - based on the maximum change in slope, i.e. maximum of the second derivative
        
        Dependencies
        ------------
            - matplotlib
            - numpy

        Comments
        --------
    """
    import numpy as np
    import matplotlib.pyplot as plt

    #all distance kombinations
    alldist = np.linalg.norm(X - X[:,None], axis=-1)

    kth_dist = np.sort(np.sort(alldist, axis=1)[:,k])  #first element is the distance to the point itself

    slope = np.diff(kth_dist)                       #no denominator, since points are equidistant
    inflection_idx = np.diff(slope).argmax()        #maximum change in slope = maximum of second derivative
    
    eps_pred_auto = kth_dist[inflection_idx+2]      #+2 because np.diff always returns arrays 1 less long than input

    if testplot:
        fontsize=16
        fig = plt.figure(figsize=(9,5))
        fig.suptitle("k-distance graph", fontsize=fontsize+4)
        ax = fig.add_subplot(111)
        ax.plot(kth_dist, ".")
        if eps_pred is not None:
            ax.hlines(eps_pred, xmin=0, xmax=kth_dist.shape[0], color="tab:orange", linestyle="--", label=f"User Prediction: {eps_pred:.2f}")
        ax.hlines(eps_pred_auto, xmin=0, xmax=kth_dist.shape[0], color="tab:orange", label=f"Autogenerated Prediction: {eps_pred_auto:.2f}")
        
        ax.invert_xaxis()
        
        ax.set_xlabel("Points", fontsize=fontsize)
        ax.set_ylabel(f"{k} distance", fontsize=fontsize)
        ax.tick_params("both", labelsize=fontsize)
        
        ax.legend(fontsize=fontsize)
        plt.tight_layout()
        if type(saveplot)==str:
            plt.savefig(saveplot, dpi=180)

        plt.show()


    return kth_dist, eps_pred_auto

def plot_feature_spaces(
    X, y_pred=None,
    class_names=None,
    markersize=20, alpha=0.8, 
    cmap="viridis", vmin=None, vmax=None, vcenter=None, ncolors=None,
    figtitle="Feature Spaces",
    figsize=(10,10), fontsize=16,
    save=False):
    """
        - function to plot all possible feature-space combinations in a correlation-plot
            - will result in a figure of X.shape[0]**2 subplots

        Parameters
        ----------
            - X
                - np.array
                - feature matrix containing the feature vectors of each sample
                - every combination of those features will be plotted
            - y_pred
                - np.array, optional
                - has to contain float or int as values
                - has to be of the same length as X
                - used to color-code the datapoints
                - the defult is None
                    - will result in all datapoints having the same label
            - class_names
                - list, np.array, optional
                - contains alternative names for the classes
                    - can also be strings
                - the default is None
                    - will copy the values passed to y_pred
            - markersize
                - float, optional
                - the markersize in points**2
                - the default is 20
            - alpha
                - float, optional
                - the alpha-value to use for the datapoints
                - the default is 0.8
            - cmap
                - str, optional
                - name of the colormap to use for coloring the different classes
                - the default is 'viridis'
            - vmin
                - int, optional
                - vmin value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.min()
            - vmax
                - int, optional
                - vmax value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.max()
            -vcenter
                - int, optional
                - vcenter value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.mean()
            - ncolors
                - int, optional
                - number of different colors to generate
                - the default is None
                    - will be set to the number of unique classes in y_pred

            - figtitle
                - str, optional
                - title of the plot
                - the default is 'Feature Spaces'
            - figsize
                - tuple, optional
                - size of the created figure
                - the default is (10,10)
            - fontsize
                - float, optional
                - fontsize to consider for the plot
                - the default is 16
            - save
                - str, optional
                - location of where to save the created figure to 
                - the default is False

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure object
                - axs
                    - list
                    - contains matplotlib axes objects

            Dependencies
            ------------
                - numpy
                - matplotlib
            
            Comments
            --------
                

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors


    if y_pred is None:
        y_pred = np.zeros(X.shape[0])
    if class_names is None:
        class_names = np.sort(np.unique(y_pred))

    #generate colors
    if y_pred is None:
        y_pred = np.zeros(X.shape[0])
    if vmin is None:
        vmin = y_pred.min()
    if vmax is None:
        vmax = y_pred.max()
    if vcenter is None:
        vcenter = y_pred.mean()
    if class_names is None:
        class_names = np.sort(np.unique(y_pred))

    if ncolors is None:
        ncolors = len(np.unique(y_pred))
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    colors = plt.cm.get_cmap(cmap, ncolors)
    colors = colors(divnorm(np.unique(y_pred)))


    fig = plt.figure(figsize=figsize)
    fig.suptitle(figtitle, fontsize=fontsize+4)
    rows = X.shape[1]
    cols = X.shape[1]
    pos = 1

    for idx1, x1 in enumerate(X.T):
        for idx2, x2 in enumerate(X.T):
            ax = fig.add_subplot(rows, cols, pos)
            
            #plot every class individually to allow labels
            for y, c, lab in zip(np.sort(np.unique(y_pred)), colors, class_names):
                
                #only label the first subplot
                if idx1 > 0 or idx2 > 0:
                    lab = None

                #plot
                ax.scatter(
                    x1[(y == y_pred)], x2[(y == y_pred)],
                    s=markersize, alpha=alpha,
                    c=[c]*len(x1[(y == y_pred)]), label=lab)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect(1/ax.get_data_ratio())    #ensure square subplots

            pos += 1

    
    fig.legend(title="Class", title_fontsize=fontsize, fontsize=fontsize)

    ax0 = fig.add_subplot(111, frameon=False)
    ax0.tick_params(labelcolor='none', which='both', labelsize=fontsize, top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
    ax0.set_xlabel("Feature i", fontsize=fontsize)
    ax0.set_ylabel("Feature j", fontsize=fontsize)
    
    plt.tight_layout()
    if type(save) == str:
        plt.savefig(save, dpi=180)

    axs = fig.axes

    return fig, axs

def estimate_DB(x1, x2, y_pred=None,
    class_names=None,
    markersize=20, alpha=1, 
    db_res=100, db_alpha=0.8,
    cmap="viridis", vmin=None, vmax=None, vcenter=None, ncolors=None,
    hide_desicion_boundaries=False,
    figtitle="Estimated Desicion Boundaries",
    figsize=(10,10), fontsize=16,
    save=False):
    """
        - function to plot estimated desicion-boundaries of data
        - uses voronoi diagrams to to do so
            - estimates the decision boundaries using KNN with K=1
            - Source: https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data
                - last access: 22.04.2022
    
        Parameters
        ----------
            - x1
                - np.array
                - some feature of the data
            - x2
                - np.array
                - some feature of the data
            - y_pred, optional
                - np.array, optional
                - has to contain float or int as values
                - has to be of the same length as x1 and x2
                - used to color-code the datapoints
                - the defult is None
                    - will result in all datapoints having the same label
            - class_names
                - list, np.array, optional
                - contains alternative names for the classes
                    - can also be strings
                - the default is None
                    - will copy the values passed to y_pred
            - markersize
                - float, optional
                - the markersize in points**2
                - the default is 20
            - alpha
                - float, optional
                - the alpha-value to use for the datapoints
                - the default is 1
            - db_res
                - int, optional
                - resolution which to plot the estimated decision boundary with
                - the default is 100
            - db_alpha
                - float, optional
                - the alpha-value to use for the decision boundary
                - the default i 0.8
            - cmap
                - str, optional
                - name of the colormap to use for coloring the different classes
                - the default is 'viridis'
            - vmin
                - int, optional
                - vmin value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.min()
            - vmax
                - int, optional
                - vmax value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.max()
            -vcenter
                - int, optional
                - vcenter value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.mean()
            - ncolors
                - int, optional
                - number of different colors to generate
                - the default is None
                    - will be set to the number of unique classes in y_pred
            - figtitle
                - str, optional
                - title of the plot
                - the default is 'Feature Spaces'
            - figsize
                - tuple, optional
                - size of the created figure
                - the default is (10,10)
            - fontsize
                - float, optional
                - fontsize to consider for the plot
                - the default is 16
            - save
                - str, optional
                - location of where to save the created figure to 
                - the default is False

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure object
                - axs
                    - list
                    - contains matplotlib axes objects
                - contour_coords
                    - tuple
                    - tuple of coordinates used to create the contour plot

            Dependencies
            ------------
                - numpy
                - matplotlib
                - sklearn
            
            Comments
            --------

    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    X = np.append(x1, x2).reshape(x1.shape[0],2)
    

    #generate colors
    if y_pred is None:
        y_pred = np.zeros(X.shape[0])
    if vmin is None:
        vmin = y_pred.min()
    if vmax is None:
        vmax = y_pred.max()
    if vcenter is None:
        vcenter = y_pred.mean()
    if class_names is None:
        class_names = np.sort(np.unique(y_pred))

    if ncolors is None:
        ncolors = len(np.unique(y_pred))
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    colors = plt.cm.get_cmap(cmap, ncolors)
    colors = colors(divnorm(np.unique(y_pred)))
    

    background_model = KNeighborsClassifier(n_neighbors=1).fit(X, y_pred)
    xx, yy = np.meshgrid(
        np.linspace(x1.min(), x1.max(), db_res),
        np.linspace(x2.min(), x2.max(), db_res),
    )
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape((db_res, db_res))

    #plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(figtitle, fontsize=fontsize+4)

    #plot decision boundary-estimate
    if not hide_desicion_boundaries:
        ax.contourf(
            xx, yy, voronoiBackground,
            alpha=db_alpha,
            cmap=cmap, vmin=y_pred.min(), vmax=y_pred.max())
    
    #plot (projected) datapoints
    for y, c, lab in zip(np.unique(y_pred), colors, class_names):
        sctr = ax.scatter(
            x1[(y == y_pred)], x2[(y == y_pred)],
            s=markersize, alpha=alpha,
            # c=color, cmap=cmap, vmin=y_pred.min(), vmax=y_pred.max())
            c=[c]*len(x1[(y == y_pred)]), label=lab)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_aspect(1/ax.get_data_ratio())    #ensure square subplots

    fig.legend(title="Class", title_fontsize=fontsize, fontsize=fontsize)

    # #add colorbar
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(sctr, cax=cbar_ax)

    # #to avoid lines in cbar with transparency
    # cbar.set_alpha(1)
    # cbar.draw_all()

    # cbar.ax.set_title("Class", ha="left", pad=20, fontsize=fontsize+2)
    # cbar.set_ticks(np.unique(y_pred))
    # cbar.set_ticklabels(class_names)
    # cbar.ax.tick_params(rotation=0, labelsize=fontsize-2)

    plt.tight_layout()
    if type(save) == str:
        plt.savefig(save, dpi=180)
    
    axs = fig.axes
    contour_coords = (xx, yy, voronoiBackground)

    return fig, axs, contour_coords

def plot_projected_feature_space(
    X, y_pred=None,
    class_names=None,
    markersize=20, alpha=1, 
    db_res=100, db_alpha=0.8,
    cmap="viridis",
    figtitle="Projected Feature Spaces",
    figsize=(10,10), fontsize=16,
    save=False):
    #TODO: add parameter to choose between tsne and umap
    """
        - function to plot a high dimensional feature space projected into 2D
        - uses t-sne projection
        - uses voronoi diagrams to estimate desicion-boundaries
            - estimates the decision boundaries using KNN with K=1
            - Source: https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data
                - last access: 22.04.2022
    
        Parameters
        ----------
            - X
                - np.array
                - feature matrix containing the feature vectors of each sample
            - y_pred
                - np.array, optional
                - has to contain float or int as values
                - has to be of the same length as X
                - used to color-code the datapoints
                - the defult is None
                    - will result in all datapoints having the same label
            - class_names
                - list, np.array, optional
                - contains alternative names for the classes
                    - can also be strings
                - the default is None
                    - will copy the values passed to y_pred
            - markersize
                - float, optional
                - the markersize in points**2
                - the default is 20
            - alpha
                - float, optional
                - the alpha-value to use for the datapoints
                - the default is 1
            - db_res
                - int, optional
                - resolution which to plot the estimated decision boundary with
                - the default is 100
            - db_alpha
                - float, optional
                - the alpha-value to use for the decision boundary
                - the default i 0.8
            - cmap
                - str, optional
                - name of the colormap to use for coloring the different classes
                - the default is 'viridis'
            - figtitle
                - str, optional
                - title of the plot
                - the default is 'Feature Spaces'
            - figsize
                - tuple, optional
                - size of the created figure
                - the default is (10,10)
            - fontsize
                - float, optional
                - fontsize to consider for the plot
                - the default is 16
            - save
                - str, optional
                - location of where to save the created figure to 
                - the default is False

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure object
                - axs
                    - list
                    - contains matplotlib axes objects

            Dependencies
            ------------
                - numpy
                - matplotlib
                - sklearn
            
            Comments
            --------

    
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    import umap


    #embed data using t-sne, estimate decision boundary using KNN with K=1
    X_embedded = umap.UMAP(n_compnents=2, random_state=0).fit_transform(X)

    fig, axs = estimate_DB(
        X_embedded[:,0], X_embedded[:,1],
        y_pred=y_pred, class_names=class_names,
        markersize=markersize, alpha=alpha,
        db_res=db_res, db_alpha=db_alpha,
        cmap=cmap,
        figtitle=figtitle,
        figsize=figsize, fontsize=fontsize,
        save=save)

    return fig, axs

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
                f"X_template = {self.X_template},\n"
                f"threshold  = {self.threshold},\n"
                f"window     = {self.window},\n"
                f"cost_fct   = {self.cost_fct},\n"
                f"y_template = {self.y_template},\n"
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
        import astroLuSt.utility_astroLuSt as alu

        if timeit:
            timer = alu.Time_stuff("DTW().fit_predict()")
            timer.start_task()

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



def plot_param_search(
    grid, metric,
    cmap="viridis", vmin=0, vmax=1,
    marker_color="r", marker_size=5,
    metric_name="metric",
    fontsize=16,
    figtitle="Hyperparameter-Search Summary", figsize=(10,8),
    save=False):
    """
        - function to create a corner plot of a parameter grid
        - will plot each tested datapoint and a plt.trisurf() for each parameter combination

        Parameters
        ----------
            - grid
                - list
                - contains dictionaries of the check parameter combinations
                - it might be useful to use sklearn.model_selection.ParameterGrid() to create the grid
            - metric
                - list of same length as 'grid'
                - contains values of some evaluation metric chosen (i.e. accuracy)
            - cmap
                - str, optional
                - name of a matplotlib colormap
                - the default is 'viridis'
            - vmin
                - float, optional
                - the minimum value to use for the colormap/colorbar
            - vmax
                - float, optional
                - the maximum value to use for the colormap/colorbar
            - marker_color, optional
                - color of the markers for the checked hyperparameter combinations
                - the default is 'red'
            - marker_size
                - float, optional
                - size of the markers for the checked hyperparameter combinations
                - the default is 5
            - metric_name
                - str, optional
                - name of your metric
                - will only be used to title the color-bar
                - the default is "metric"
            - fontsize
                - float, optional
                - fontsize to use on the plot
                - the default is 16
            - figtitle
                - str, optional
                - title of the final plot
                - the default is 'Hyperparameter-Search Summary'
            - figsize
                - tuple, optional
                - size of the figure
                - the default is (10,8)
            - save
                - str, optional
                - path of where to save the created figure to
                - the default is False

        Raises
        ------
            - AssertionError
                - if inputs are of wrong shape
            
        Returns
        -------
            - fig
                - matplotlib figure object
            - axs
                - list
                - contains matplotlib axes objects

        Dependencies
        ------------
            - numpy
            - matplotlib
        
        Comments
        --------
            - string variables will be converted to an arange
            - i.e. if you use 3 stringvariables in your parameter-grid, every third datapoint will be the same stringvariable

    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert len(grid) == len(metric), "'grid' and 'metric' have to be of the same shape!"

    grid2 = []
    keys = []
    for key in grid[0].keys():
        param = []
        keys.append(key)
        for idx, params in enumerate(grid):
            if type(params[key])==str:
                param.append(idx)
            else:
                param.append(params[key])
        grid2.append(np.array(param))

    rows = len(grid2)
    cols = len(grid2)
    pos = 1

    fig = plt.figure(figsize=figsize)
    fig.suptitle(figtitle, fontsize=fontsize+4)
    for i, keyi in zip(range(rows), keys):
        for j, keyj in zip(range(cols), keys):
            label = None
            if i == 1 and j == 0:
                label = "Test-Points"
            ax = fig.add_subplot(rows, cols, pos)

            if  i != j:
                cont = ax.tricontourf(grid2[j], grid2[i], metric, cmap=cmap, zorder=1, vmin=vmin, vmax=vmax)
            ax.scatter(grid2[j], grid2[i], c=marker_color, s=marker_size, zorder=3, label=label)
            ax.set_aspect(1/ax.get_data_ratio())    #ensure square subplots

            if i == cols-1:
                ax.set_xlabel(keyj, fontsize=fontsize)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(keyi, fontsize=fontsize)
            else:
                ax.set_yticks([])
            ax.tick_params("both", labelsize=fontsize)

            pos += 1

    #add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.83])
    cbar = fig.colorbar(cont, cax=cbar_ax)
    cbar.set_label(metric_name, fontsize=fontsize, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=fontsize-2)

    plt.tight_layout(rect=[0,0,.85,1])
    fig.legend(fontsize=fontsize, loc="upper left")

    if type(save) == str:
        plt.savefig(save, dpi=180)
    plt.show()

    axs = fig.axes

    return fig, axs
