
#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Tuple

#%%classes
class Images2Patches:
    """
        - class to take a series of input images and split each of them into patches of size `(xpatches,ypatches)`

        Attributes
        ----------
            - `xpatches`
                - `int`
                - number of patches to split the images into along their width (axis 1)
                - axis 1 has to be divisible by `xpatches`
            - `ypatches`
                - `int`
                - number of patches to split the images into along their height (axis 0)
                - axis 0 has to be divisible by `ypatches`
            - `verbose`
                - `int`, optional
                -  verbosity level
                - the default is `0`

        Methods
        -------
            - `fit()`
            - `predict()`
            - `fit_predcit()`
            - `plot_results()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------
    """

    def __init__(self,
        xpatches:int, ypatches:int,
        verbose:int=0,
        ) -> None:

        self.xpatches   = xpatches
        self.ypatches   = ypatches
        self.verbose    = verbose

        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    xpatches={repr(self.xpatches)},\n'
            f'    ypatches={repr(self.ypatches)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def fit(self,
        X:np.ndarray, y:np.ndarray=None,
        xpatches:int=None, ypatches:int=None,
        verbose:int=None,
        ) -> None:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - series of images to be split into patches
                    - has to have shape `(nsamples,xpixels,ypixels)`
                - `y`
                    - `np.ndarray`, optional
                    - not needed
                    - only implemented for consitency
                    - the default is `None`
                - `xpatches`
                    - `int`, optional
                    - number of patches to split the images into along their width (axis 1)
                    - overrides `self.xpatches`
                    - the default is `None`
                        - will fall back to `self.xpatches`
                - `ypatches`
                    - `int`, optional
                    - number of patches to split the images into along their height (axis 0)
                    - overrides `self.ypatches`
                    - the default is `None`
                        - will fall back to `self.ypatches`
                - `verbose`
                    - `int`, optional
                    -  verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if xpatches  is None:   xpatches    = self.xpatches
        if ypatches  is None:   ypatches    = self.ypatches
        if verbose   is None:   verbose     = self.verbose
        
        #split into patches
        X_patched = X.reshape(X.shape[0], ypatches, X.shape[1]//ypatches, xpatches, X.shape[2]//xpatches)
        X_patched = X_patched.transpose(0, 1, 3, 2, 4)

        self.X_patched = X_patched

        return
    
    def transform(self,
        X:np.ndarray=None, y:np.ndarray=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to transform the input data
                
            Parameters
            ----------
                - `X`
                    - `np.ndarray`, optional
                    - not needed here
                    - only implemented for consistency
                    - series of images to be split into patches
                    - has to have shape `(nsamples,xpixels,ypixels)`
                    - the default is `None`
                - `y`
                    - `np.ndarray`, optional
                    - not needed
                    - only implemented for consitency
                    - the default is `None`
                - `verbose`
                    - `int`, optional
                    -  verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`

            Raises
            ------

            Returns
            -------
                - `X_patched`
                    - `np.ndarray`
                    - series of patched images corresponding to `X` passed to `self.fit()`
                    - has shape `(nsamples,ypatches,xpatches,ypixels,xpixels)`
                        - `ypatches` are hereby the number of patches (in height)
                        - `xpatches` are hereby the number of patches (in width)
                        - `ypixels` are hereby the number of pixels (in height) every patch posesses
                        - `xpixels` are hereby the number of pixels (in width) every patch posesses

            Comments
            --------           
        """
        if verbose is None:     verbose     = self.verbose
        
        X_patched = self.X_patched

        return X_patched
    
    def fit_transform(self,
        X:np.ndarray, y:np.ndarray=None,
        xpatches:int=None, ypatches:int=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to fit the transforme and transform the input data in one go
                
            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - series of images to be split into patches
                    - has to have shape `(nsamples,xpixels,ypixels)`
                - `y`
                    - `np.ndarray`, optional
                    - not needed
                    - only implemented for consitency
                    - the default is `None`
                - `xpatches`
                    - `int`, optional
                    - number of patches to split the images into along their width (axis 1)
                    - overrides `self.xpatches`
                    - the default is `None`
                        - will fall back to `self.xpatches`
                - `ypatches`
                    - `int`, optional
                    - number of patches to split the images into along their height (axis 0)
                    - overrides `self.ypatches`
                    - the default is `None`
                        - will fall back to `self.ypatches`
                - `verbose`
                    - `int`, optional
                    -  verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`


            Raises
            ------

            Returns
            -------
                - `X_patched`
                    - `np.ndarray`
                    - series of patched images corresponding to `X` passed to `self.fit()`
                    - has shape `(nsamples,ypatches,xpatches,ypixels,xpixels)`
                        - `ypatches` are hereby the number of patches (in height)
                        - `xpatches` are hereby the number of patches (in width)
                        - `ypixels` are hereby the number of pixels (in height) every patch posesses
                        - `xpixels` are hereby the number of pixels (in width) every patch posesses

            Comments
            --------           
        """        

        if verbose is None:     verbose     = self.verbose
        if fit_kwargs is None:  fit_kwargs  = dict()
        
        self.fit(X, y, xpatches=xpatches, ypatches=ypatches, verbose=verbose)
        X_patched = self.transform(X, y, verbose=verbose)

        return X_patched
    
    def plot_result(self,
        X:np.ndarray,
        X_in:np.ndarray,
        fig:Figure=None,
        verbose:int=None,
        pcolormesh_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visualize the result

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - has to be 3d
                    - has to have shape `(ypatches,xpatches,ypixels,xpixels)`
                    - one instance of the transformed version of the input `X` to `self.fit()`
                - `X_in`
                    - `np.ndarray`
                    - has to be 2d
                    - has to have shape `(ypixels,xpixels)`
                    - one instance of the input `X` to `self.fit()`
                - `fig`
                    - `matplotlib.figure.Figure`, optional
                    - figure to plot into
                    - the default is `None`
                        - will generate a new figure
                - `verbose`
                    - `int`, optional
                    -  verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict(vmin=X.min(), vmax=X.max())`
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `matplotlib.figure.Figure`
                    - created figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
                    
            Comments
            --------
        """

        #default parameters
        if pcolormesh_kwargs is None: pcolormesh_kwargs = dict(vmin=X.min(), vmax=X.max())

        #get pixel-indices
        size = (X.shape[0]*X.shape[2],X.shape[1]*X.shape[3])
        xx_in, yy_in = np.meshgrid(
            np.arange(size[1]),
            np.arange(size[0]),
        )
        
        #get pixel-indices of patches
        ypatches = X.shape[0]
        xpatches = X.shape[1]
        ypixels  = X.shape[2]
        xpixels  = X.shape[3]
        
        xx = xx_in.reshape(ypatches, xx_in.shape[0]//ypatches, xpatches, xx_in.shape[1]//xpatches)
        xx = xx.transpose(0, 2, 1, 3)
        yy = yy_in.reshape(ypatches, yy_in.shape[0]//ypatches, xpatches, yy_in.shape[1]//xpatches)
        yy = yy.transpose(0, 2, 1, 3)
        

        #plotting
        if fig is None: fig = plt.figure()

        ##input frame
        ax1 = fig.add_subplot(1, xpatches+1, 1)
        ax1.pcolormesh(xx_in, yy_in, X_in, **pcolormesh_kwargs)
        ax1.set_title('Input')
        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')

        ##patches
        idx = 0
        for i in range(ypatches)[::-1]: #reverse to follow ax.pcolormesh orientation
            for j in range(xpatches+1):
                if j == 0:
                    #ignore first colum (save space for input image)
                    pass
                else:
                    #plot patches
                    axij = fig.add_subplot(ypatches, xpatches+1, idx+1)
                    axij.set_aspect('equal')
                    axij.pcolormesh(xx[i,j-1], yy[i,j-1], X[i,j-1], **pcolormesh_kwargs)    #plot j-1 because j==0 ignored

                    #add title
                    axij.set_title(f'X[{i},{j-1}]')

                    #labelling
                    if j != 1:
                        axij.set_yticks([])
                    else:
                        axij.set_ylabel('Pixel')
                    if i > 0:
                        axij.set_xticks([])
                    if i == 0:
                        axij.set_xlabel('Pixel')

                #update index
                idx += 1

        axs = fig.axes
        
        return fig, axs
    
