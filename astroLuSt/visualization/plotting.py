

#%%imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Union

#%%definitions
def generate_colors(classes:Union[list,np.ndarray], 
    vmin:float=None, vmax:float=None, vcenter:float=None,
    ncolors:int=None,
    cmap:Union[str,mcolors.Colormap]="nipy_spectral"
    ) -> list:
    """
        - function to generate colors for a given set of unique classes
        - generates colors based on a matplotlib colormap

        Parameters
        ----------
            - `classes`
                - list, np.array, int
                - the classes to consider
                - if an integer is passed, will be interpreted as the number of unique classes
                - does not have to consist of unique classes
                    - the function will pick out the unique classes by itself
            - `vmin`
                - float, optional
                - `vmin` value of the colormap
                - useful if you want to modify the class-coloring
                - the default is `None`
                    - will be set to `y_pred.min()`
            - `vmax`
                - float, optional
                - vmax value of the colormap
                - useful if you want to modify the class-coloring
                - the default is `None`
                    - will be set to `y_pred.max()`
            - `vcenter`
                - float, optional
                - vcenter value of the colormap
                - useful if you want to modify the class-coloring
                - the default is None
                    - will be set to `y_pred.mean()`
            - `ncolors`
                - int, optional
                - number of different colors to generate
                - the default is `None`
                    - will be set to the number of unique classes in `y_pred`
            - `cmap`
                - str, mcolors.Colormap, optional
                - name of the colormap or ListedColormap to use for coloring the different classes
                - the default is `'nipy_spectral'`

        Raises
        ------

        Returns
        -------
            - `colors`
                - list
                - contains as many different colors as there are unique classes in `classes`

        Dependencies
        ------------
            - numpy
            - matplotlib
            - typing

        Comments
        --------

    """


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

def generate_categorical_cmap(
    colors:Union[list,tuple], res:int=256
    ) -> mcolors.ListedColormap:
    """
        - function to generate a custom (categorical) colormap by passing a list of colors

        Parameters
        ----------
            - `colors`
                - list, tuple
                - colors to use for values in ascending order
                - will assign each entry (color) to `res//len(colors)` values in the colormap
                - can contain strings or RGBA tuples
                    - strings have to be named colors
            - `res`
                - int, optional
                - resolution of the colormap
                - has to be larger than `len(colors)` to get a good result
                - the default is 256

        Raises
        ------

        Returns
        -------
            - `cmap`
                - mcolors.ListedColormap
                - generated colormap

        Dependencies
        ------------
            - matplotlib
            - numpy

        Comments
        --------

    """

    #create custom color map

    #divide 
    npercolor = res//(len(colors))

    #template colormap
    viridis = plt.cm.get_cmap('viridis', res)
    custom_colors = viridis(np.linspace(0, 1, res))
    for idx, c in enumerate(colors):
        #convert to RGBA tuple if named color is passed
        if isinstance(c, str): c = mcolors.to_rgba(c)

        custom_colors[idx*npercolor:, :] = c
    cmap = mcolors.ListedColormap(custom_colors)

    return cmap