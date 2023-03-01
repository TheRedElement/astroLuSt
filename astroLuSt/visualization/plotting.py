


def generate_colors(classes, 
    vmin:float=None, vmax:float=None, vcenter:float=None,
    ncolors:int=None,
    cmap="nipy_spectral"):
    #TODO: Include in Model_02 (i.e. the plotting_astroLuSt module)
    """
        - function to generate colors for a given set of unique classes
        - generates colors based on a matplotlib colormap

        Parameters
        ----------
            - classes
                - list, np.array, int
                - the classes to consider
                - if an integer is passed, will be interpreted as the number of unique classes
                - does not have to consist of unique classes
                    - the function will pick out the unique classes by itself
            - cmap
                - str, optional
                - name of the colormap to use for coloring the different classes
                - the default is 'viridis'
            - vmin
                - float, optional
                - vmin value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.min()
            - vmax
                - float, optional
                - vmax value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.max()
            - vcenter
                - float, optional
                - vcenter value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to y_pred.mean()
            - ncolors
                - int, optional
                - number of different colors to generate
                - the default is None
                    - will be set to the number of unique classes in y_pred


        Raises
        ------

        Returns
        -------
            - colors
                - list
                - contains as many different colors as there are unique classes in 'classes'

        Dependencies
        ------------
            - numpy
            - matplotlib

        Comments
        --------

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if isinstance(classes, int):
        classes_int = np.arange(0, classes, 1, dtype=int)    #initialize integer-class array
    else:
        classes_int = np.arange(0, np.unique(classes).shape[0], 1, dtype=int)    #initialize integer-class array

    if vmin is None:
        vmin = 0
    if vmax is None:
        if isinstance(classes, int):
            vmax = classes
        else:
            vmax = np.unique(classes).shape[0]
    if vcenter is None:
        vcenter = (vmin+vmax)/2
    if ncolors is None:
        if isinstance(classes, int):
            ncolors = classes
        else:
            ncolors = np.unique(classes).shape[0]

    #generate colors
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    colors = plt.cm.get_cmap(cmap, ncolors)
    colors = colors(divnorm(np.unique(classes_int)))
    return colors