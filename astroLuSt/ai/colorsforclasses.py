
#%%imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Union

import warnings

#%%definitions
def generate_colors(
    classes:Union[list,np.ndarray],
    vmin:int=None, vmax:int=None, vcenter:int=None,
    ncolors:int=None,
    cmap:str="viridis"
    ) -> list:
    """
        - function to generate colors for a given set of unique classes
        - generates colors based on a matplotlib colormap

        Parameters
        ----------
            - `classes`
                - list, np.ndarray
                - the `classes` to consider
                - does not have to consist of unique `classes`
                    - the function will pick out the unique `classes` by itself
            - `vmin`
                - int, optional
                - `vmin` value of the colormap
                - useful if you want to modify the class-coloring
                - the default is `None`
                    - will be set to `y_pred.min()`
            - `vmax`
                - int, optional
                - `vmax` value of the colormap
                - useful if you want to modify the class-coloring
                - the default is `None`
                    - will be set to `y_pred.max()`
            - `vcenter`
                - int, optional
                - `vcenter` value of the colormap
                - useful if you want to modify the class-coloring
                - the default is `None`
                    - will be set to `y_pred.mean()`
            - `ncolors`
                - int, optional
                - number of different colors to generate
                - the default is `None`
                    - will be set to the number of unique `classes` in `y_pred`
            - `cmap`
                - str, optional
                - name of the colormap to use for coloring the different `classes`
                - the default is 'viridis'


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

    warnings.warn('This function is deprecated. Use astroLuSt.visualization.plotting.generate_colors() instead!')

    classes_int = np.arange(0, np.unique(classes).shape[0], 1, dtype=int)    #initialize integer-class array

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.unique(classes).shape[0]
    if vcenter is None:
        vcenter = (vmin+vmax)/2
    if ncolors is None:
        ncolors = np.unique(classes).shape[0]

    #generate colors
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    colors = plt.cm.get_cmap(cmap, ncolors)
    colors = colors(divnorm(np.unique(classes_int)))
    
    return colors


