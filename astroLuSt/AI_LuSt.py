#TODO: create class to store functions

def plot_confusion_matrix(
    vals, classes,
    xlab="Class", ylab="Class", cbarlabel="", cmap="viridis",
    annotate=True, annotationcolor="w", textfontsize=None,
    figsize=(9,5), fontsize=16,
    save=False):
    #TODO: Finish documentation
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


def kdist(X, k=4, eps_pred=None, testplot=False):
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
        plt.show()


    return kth_dist, eps_pred_auto

def plot_feature_spaces(
    X, y_pred=None,
    class_names=None,
    markersize=20, alpha=.8,
    cmap="viridis",
    figtitle="Feature Spaces",
    figsize=(10,10), fontsize=16,
    save=False):
    #NOTE: changes from last git push: markersize, alpha
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

    if y_pred is None:
        y_pred = np.zeros(X.shape[0])
        color = eval(f"plt.cm.{cmap}(np.linspace(0,1,1))")
        cmap=None
    else:
        color = y_pred
    if class_names is None:
        class_names = np.sort(np.unique(y_pred))

    fig = plt.figure(figsize=figsize)
    fig.suptitle(figtitle, fontsize=fontsize+4)
    rows = X.shape[1]
    cols = X.shape[1]
    pos = 1
    for idx1, x1 in enumerate(X.T):
        for idx2, x2 in enumerate(X.T):

            ax = fig.add_subplot(rows, cols, pos)
            sctr = ax.scatter(
                x1, x2, 
                s=markersize, alpha=alpha,
                c=color, cmap=cmap, vmin=y_pred.min(), vmax=y_pred.max())
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.set_aspect(1/ax.get_data_ratio())    #ensure square subplots

            pos += 1
    

    ax0 = fig.add_subplot(111, frameon=False)
    ax0.tick_params(labelcolor='none', which='both', labelsize=fontsize, top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
    ax0.set_xlabel("Feature i", fontsize=fontsize)
    ax0.set_ylabel("Feature j", fontsize=fontsize)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.13, 0.05, 0.8])
    cbar = fig.colorbar(sctr, cax=cbar_ax)

    #to avoid lines in cbar with transparency
    cbar.set_alpha(1)
    cbar.draw_all()

    cbar.ax.set_title("Class", ha="left", pad=20, fontsize=fontsize+2)
    cbar.set_ticks(np.unique(y_pred))
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(rotation=0, labelsize=fontsize-2)
    
    
    # plt.tight_layout()
    if type(save) == str:
        plt.savefig(save, dpi=180)

    axs = fig.axes

    return fig, axs
