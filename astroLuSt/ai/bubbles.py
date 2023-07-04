

#%%imports
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Union, Tuple, Any, Callable

#%%classes
class BUBBLES:
    """
        - 

        Attributes
        ----------

        Methods
        -------

        Dependencies
        ------------
            - joblib
            - matplotlib
            - numpy
            - sklearn
            - typing
        
        Comments
        --------

    """

    def __init__(self,
        func:Union[str,Callable],
        r0:float, min_pts:int=0,
        res:int=10, 
        n_jobs:int=-1,
        verbose:int=0,
        ) -> None:
        
        if func == 'sphere':    self.func   = self.__sphere_nd
        elif func == 'rect':    self.func   = self.__rect_nd
        elif func is None:      self.func   = self.__sphere_nd
        else:                   self.func   = func
        self.r0         = r0
        self.min_pts    = min_pts
        self.res        = res
        
        if n_jobs is None:      self.n_jobs = 1
        else:                   self.n_jobs = n_jobs

        self.verbose = verbose
        return

    def __repr__(self) -> str:
        return (
            f'BUBBLES(\n'
            f'    func={repr(self.func)},\n'
            f'    r0={repr(self.r0)}, min_pts={repr(self.min_pts)},\n'
            f'    res={repr(self.res)},\n'
            f'    n_jobs={repr(self.n_jobs)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def get_most_common(self,
        x:np.ndarray
        ) -> Any:
        """
            - method to obtain the most common element in `x`

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input array to get the most common element from

            Raises
            ------

            Returns
            -------
                - `most_common`
                    - Any
                    - entry of `x` that is most common

            Comments
            --------

        """

        uniques, counts = np.unique(x, return_counts=True)
        most_common = uniques[np.argmax(counts)]
        
        return most_common

    def __sphere_nd(self,
        X:np.ndarray, y:np.ndarray=None,
        X_grid:np.ndarray=None, 
        r0:float=None, min_pts:int=None,
        fit:bool=True,
        **kwargs,
        ) -> Tuple[np.ndarray,Any]:
        """
            - private method to determine if any point in `X` is within a radius `r0` of the entries in `X_grid`
            - used to fit the classifier
            - used to make predictions

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - training/testing data to fit/predict
                - `y`
                    - np.ndarray, optional
                    - labels corresponding to `X`
                    - has to be passed during fitting
                    - will be ignored during inference
                    - the default is `None`
                - `X_grid`
                    - np.ndarray, optional
                    - array of shape `(X.shape[1],*[res]*X.shape[1])`
                        - i.e. `res` datapoints in as many dimensions as `X` has features
                    - has to be passed during fitting
                    - will be ignored during inference
                    - the default is `None`
                - `r0`
                    - float, optional
                    - radius around any point in `X_grid` (during fitting) or `self.X_grid` (during inference) in which any point in `X` has to lie to be considered close to that grid-point
                    - during fitting
                        - the label for any point in `X_grid` will be assigned by the dominant label in `X` that is within the radius `r0`
                        - if the point in `X_grid` contains no points out of `X`, it will be considered as noise and set to `np.nan`
                    - during inference
                        - any point in `X` will be tested against the not-noise-points in `self.X_grid`
                            - if it falls within the radius of any point in `self.X_grid`, it will get the corresponding label assigned to it
                            - otherwise it will be assigned the label `-1`, which means that the datapoint is noise
                    - overwrites `self.r0`
                    - the default is `None`
                        - will fall back to `self.r0`                
                - `min_pts`
                    - int, optional
                    - minimum number of points a sphere around any point in `X_grid` has to contain to not be considered noise
                    - i.e. if the point contains less than `min_pts` points in its `r0` neighborhood, it will be considered as noise and dropped from  the grid
                    - overwrites `self.min_pts`
                    - the default is `None`
                        - will fall back to `self.min_pts`
                - `fit`
                    - bool, optional
                    - flag of the classifier state
                    - if set to `True`
                        - runs in fitting mode
                    - if set to `False`
                        - runs in inference mode
                -`**kwargs`
                    - dict, optional
                    - kwargs passed to the function

            Raises
            ------
                - `ValueError`
                    - if `y` is missing during fitting

            Returns
            -------
                - fitting
                    - Xy_grid
                        - np.ndarray
                        - combined array of 
                            - grid point coordinates (`Xy[:-1]`) 
                            - grid point label ('Xy[-1]`)
                - inference
                    - y_pred
                        - Any
                        - predicted label for the input datapoint `X`

            Comments
            --------
        """

        #while fitting
        if fit:
            if y is None: raise(ValueError('`y` cannot be `None` during fitting (i.e. `fit==True`), since this is a supervised classifier.'))
            
            diff = np.sum(np.sqrt((X_grid-X)**2), axis=1)
            y_bool = (diff < r0)

            if y_bool.sum() > min_pts:  y_grid = self.get_most_common(y[y_bool])
            else:                       y_grid = np.nan

            Xy_grid = np.append(X_grid, y_grid)
            return Xy_grid
        
        #while predicting
        else:
            diff = np.sum(np.sqrt((self.X_grid-X)**2), axis=1)
            #assign class
            if np.nanmin(diff) <= self.r0:
                y_pred = self.y_grid[np.nanargmin(diff)]
            #classify as noise
            else:
                y_pred = -1
            
            return y_pred
        
    def __rect_nd(self,
        X:np.ndarray=None, y:np.ndarray=None,
        X_grid:np.ndarray=None,
        r0:float=None, min_pts:int=None,
        fit:bool=True,
        **kwargs,
        ) -> None:
        """
            - private method to determine if any point in `X` is within a cuboid of sidelength `r0` of the entries in `X_grid`
            - used to fit the classifier
            - used to make predictions

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - training/testing data to fit/predict
                - `y`
                    - np.ndarray, optional
                    - labels corresponding to `X`
                    - has to be passed during fitting
                    - will be ignored during inference
                    - the default is `None`
                - `X_grid`
                    - np.ndarray, optional
                    - array of shape `(X.shape[1],*[res]*X.shape[1])`
                        - i.e. `res` datapoints in as many dimensions as `X` has features
                    - has to be passed during fitting
                    - will be ignored during inference
                    - the default is `None`
                - `r0`
                    - float, optional
                    - side length of the cuboid around any point in `X_grid` (during fitting) or `self.X_grid` (during inference) in which any point in `X` has to lie to be considered close to that grid-point
                    - during fitting
                        - the label for any point in `X_grid` will be assigned by the dominant label in `X` that is within the corresponding cuboid
                        - if the point in `X_grid` contains no points out of `X`, it will be considered as noise and set to `np.nan`
                    - during inference
                        - any point in `X` will be tested against the not-noise-points in `self.X_grid`
                            - if it falls within the cuboid of any point in `self.X_grid`, it will get the corresponding label assigned to it
                            - otherwise it will be assigned the label `-1`, which means that the datapoint is noise
                    - overwrites `self.r0`
                    - the default is `None`
                        - will fall back to `self.r0`
                - `min_pts`
                    - int, optional
                    - minimum number of points a cuboid around any point in `X_grid` has to contain to not be considered noise
                    - i.e. if the point contains less than `min_pts` points in its `r0` neighborhood, it will be considered as noise and dropped from  the grid
                    - overwrites `self.min_pts`
                    - the default is `None`
                        - will fall back to `self.min_pts`
                - `fit`
                    - bool, optional
                    - flag of the classifier state
                    - if set to `True`
                        - runs in fitting mode
                    - if set to `False`
                        - runs in inference mode
                -`**kwargs`
                    - dict, optional
                    - kwargs passed to the function

            Raises
            ------
                - `ValueError`
                    - if `y` is missing during fitting

            Returns
            -------
                - fitting
                    - Xy_grid
                        - np.ndarray
                        - combined array of 
                            - grid point coordinates (`Xy[:-1]`) 
                            - grid point label ('Xy[-1]`)
                - inference
                    - y_pred
                        - Any
                        - predicted label for the input datapoint `X`

            Comments
            --------
        """

        #while fitting
        if fit:
            if y is None: raise(ValueError('`y` cannot be `None` during fitting (i.e. `fit==True`), since this is a supervised classifier.'))
            y_bool = np.all((
                (X_grid-r0/2 < X)&
                (X < X_grid+r0/2)
            ), axis=1)

            if y_bool.sum() > min_pts:  y_grid = self.get_most_common(y[y_bool])
            else:                       y_grid = np.nan

            Xy_grid = np.append(X_grid, y_grid)

            return Xy_grid
        
        #while predicting
        else:
            y_bool = np.all((
                (self.X_grid-self.r0/2 < X)&
                (X < self.X_grid+self.r0/2)
            ), axis=1)

            #assign class
            if y_bool.sum() > 0:
                y_pred = self.get_most_common(self.y_grid[y_bool])
            #classify as noise
            else:
                y_pred = -1

            return y_pred


    def fit(self,
        X:np.ndarray, y:np.ndarray,
        func:Union[str,Callable]=None,
        r0:float=None, min_pts:int=None,
        res:Union[int,tuple]=None,
        n_jobs:int=None,
        verbose:int=None,
        ) -> None:
        """
            - method to fit the classifier

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - training set of shape `(n_samples, n_features)`
                - y
                    - np.ndarray
                    - labels corresponding to `X`
                - `func`
                    - str, callable, optional
                    - function to use for determining if any point in `X_grid` is considered close to any point in `X`
                    - allowed strings
                        - 'sphere'
                            - will use `self.__sphere_nd`
                            - uses spheres around points in `X_grid` to determine closeness
                        - 'rect' 
                            - will use `self.__rect_nd`
                            - uses a cuboids around points in `X_grid` to determine closeness
                    - if callable
                        - has to take the following arguments
                            - `X`
                                - np.ndarray
                                - training set during `self.fit`
                                - a single test point during `self.predict`
                            - `r0`
                                - float
                                - some distance measure
                            - 'min_pts`
                                - int
                                - measure of minimum number of points in `X` any point in `X_grid` has to be close to in order to be classified  NOT as an outlier
                            - `fit`
                                - bool
                                - flag of classifier state
                                - `True` for fitting mode
                                - `False` for inference mode
                            - `**kwargs`
                                - dict
                                - additional kwargs to be passed to `func`
                    - overwrites `self.func`
                    - the default is `None`
                        - will fall back to `self.func`
                - `r0`
                    - float, optional
                    - radius around any point in `X_grid` (during fitting) or `self.X_grid` (during inference) in which any point in `X` has to lie to be considered close to that grid-point
                    - during fitting
                        - the label for any point in `X_grid` will be assigned by the dominant label in `X` that is within the radius `r0`
                        - if the point in `X_grid` contains no points out of `X`, it will be considered as noise and set to `np.nan`
                    - during inference
                        - any point in `X` will be tested against the not-noise-points in `self.X_grid`
                            - if it falls within the radius of any point in `self.X_grid`, it will get the corresponding label assigned to it
                            - otherwise it will be assigned the label `-1`, which means that the datapoint is noise
                    - overwrites `self.r0`
                    - will be adopted as 'self.r0` by the classifer if set
                    - the default is `None`
                        - will fall back to `self.r0`                
                - `min_pts`
                    - int, optional
                    - minimum number of points a sphere around any point in `X_grid` has to contain to not be considered noise
                    - i.e. if the point contains less than `min_pts` points in its `r0` neighborhood, it will be considered as noise and dropped from  the grid
                    - overwrites `self.min_pts`
                    - will be adopted as 'self.min_pts` by the classifer if set
                    - the default is `None`
                        - will fall back to `self.min_pts`
                - `res`
                    - int, optional
                    - resolution of the `X_grid` in all dimensions
                    - will be passed to `np.linspace()`
                    - high values for `res` are especially bad in high dimensions
                        - i.e. "Curse of dimensionality" (https://en.wikipedia.org/wiki/Curse_of_dimensionality)
                    - overwrites `self.res`
                    - the default is `None`
                        - will fallback to `self.res`
                - `n_jobs`
                    - int, optional
                    - number of threads to use for parallel computattion
                    - will be passed to `joblib.Parallel()`
                    - overwrites `self.n_jobs`
                    - the default is `None`
                        - falls back to `self.n_jobs`
                - 'verbose`
                    - int, optional
                    - verbosity level
                    - overwrites `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`

            Raises
            ------

            Returns
            -------

            Comments
            --------
                - make sur to choose `res` acoording to the dimensionality and size of your dataset

        """
        
        #default values
        if func == 'sphere':    func    = self.__sphere_nd
        elif func == 'rect':    func    = self.__rect_nd
        elif func is None:      func    = self.func
        if r0 is None:          r0      = self.r0
        if min_pts is  None:    min_pts = self.min_pts
        if res is None:         res     = self.res

        if n_jobs is None:      n_jobs  = self.n_jobs 

        if verbose is None: verbose = self.verbose

        #generate grid from linspaces
        X_base = np.linspace(np.nanmin(X, axis=0), np.nanmax(X, axis=0), res)
        X_points = np.array(np.meshgrid(*X_base.T))

        if verbose > 2:
            print(f'INFO(BUBBLES.fit): Generated X_points of shape: {X_points.shape}')
        
        #2d array of all points in X_points
        X_points = np.vstack([np.ravel(Xg) for Xg in X_points]).T

        #assign labels to grid-points
        Xy_grid = np.array(Parallel(n_jobs=n_jobs, verbose=verbose, prefer='threads')(
            # delayed(self.__sphere_nd)(
            delayed(func)(
                X=X, y=y,
                r0=r0, min_pts=min_pts,
                X_grid=Xi,
                fit=True,
            ) for Xi in X_points
        ))

        #separate grid from label
        self.X_grid = Xy_grid[:,:-1]
        self.y_grid = Xy_grid[:,-1]


        #remove points that got classified as noise (np.nan)
        self.X_grid = self.X_grid[np.isfinite(self.y_grid)]
        self.y_grid = self.y_grid[np.isfinite(self.y_grid)]

        #adopt hyperparameters
        self.r0 = r0
        self.min_pts = min_pts
        self.func = func

        return
    
    def predict(self,
        X:np.ndarray, y:np.ndarray=None,
        n_jobs:int=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to make predictions using the fitted classifier

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - dataset to predict labels of
                    - has to have shape `(n_samples,n_features)`
                - `y`
                    - np.ndarray, optional
                    - labels corresponding to `X`
                    - not needed during inference
                    - the default is `None`
                - `n_jobs`
                    - int, optional
                    - number of threads to use for parallel computattion
                    - will be passed to `joblib.Parallel()`
                    - overwrites `self.n_jobs`
                    - the default is `None`
                        - falls back to `self.n_jobs`
                - 'verbose`
                    - int, optional
                    - verbosity level
                    - overwrites `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`                    

            Raises
            ------

            Returns
            -------
                - `y_pred`
                    - np.ndarray
                    - predicted labels corresponding to `X`

            Comments
            --------

        """


        if n_jobs is None:  n_jobs  = self.n_jobs 
        if verbose is None: verbose = self.verbose

        y_pred = np.array(Parallel(n_jobs=n_jobs, verbose=verbose, prefer='threads')(
            delayed(self.func)(
                X=Xi,
                fit=False,
            ) for Xi in X
        ))

        return y_pred
    
    def fit_predict(self,
        X:np.ndarray, y:np.ndarray,
        fit_kwargs:dict=None, predict_kwargs:dict=None,
        ) -> np.ndarray:
        """
            - method to fit the classifier and make a prediction on the passed dataset

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - training set of shape `(n_samples, n_features)`
                - y
                    - np.ndarray
                    - labels corresponding to `X`
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
                    - predicted labels corresponding to `X`

            Comments
            --------
        """

        if fit_kwargs is None:      fit_kwargs = {}
        if predict_kwargs is None:  predict_kwargs = {}
        
        self.fit(X, y, **fit_kwargs)
        y_pred = self.predict(X, **predict_kwargs)

        return y_pred

    def score(self,
        X:np.ndarray, y:np.ndarray,
        sample_weight:np.ndarray=None,
        predict_kwargs:dict=None,
        ) -> float:
        """
            - given input-data and true labels return the mean accuracy

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - test set of shape `(n_samples, n_features)`
                - y
                    - np.ndarray
                    - labels corresponding to `X`
                - `sample_weight`
                    - np.ndarray
                    - sample weights for samples in `X`
                    - will be passed to `sklearn.metrics.accuracy_score`
                    - the default is `None`
                - `predict_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.predict()`
                    - the default is `None`
                        - will be set to `{}`

            Raises
            ------

            Returns
            -------
                - score
                    - float
                    - mean accuracy accross all samples in `X`, give the (ground-truth) labels in `y`

            Comments
            --------

        """
        if predict_kwargs is None: predict_kwargs = {}

        y_pred = self.predict(X, **predict_kwargs)

        score = accuracy_score(y, y_pred, sample_weight=sample_weight)

        return score

    def plot_result(self,
        X:np.ndarray=None, y:np.ndarray=None,
        dims:list=None,
        cmap:Union[str,mcolors.Colormap]=None,
        grid_scatter_kwargs:dict=None, data_scatter_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - 
        """

        #default values
        if dims is None:            dims = [0,1]
        if cmap is None:            cmap = 'nipy_spectral'
        if grid_scatter_kwargs is None:  grid_scatter_kwargs = {'alpha':0.5, 's':10, 'vmin':-1,}
        if data_scatter_kwargs is None:  data_scatter_kwargs = {'alpha':0.5, 's':50, 'vmin':-1, 'ec':'w'}

        #select plotting dimension based on self.X_grid and dims
        if self.X_grid.shape[1] < 3 or len(dims) < 3:
            projection = None
            dims = dims[:2]
        else:
            projection = '3d'

        #plotting
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection=projection)

        #plot grid
        mappable = ax1.scatter(*self.X_grid.T[dims],                    c=self.y_grid,                    cmap=cmap, **grid_scatter_kwargs)
        
        #if a dataset has been passed plot that as well
        if X is not None:
            ax1.scatter(*X[:,dims].T, c=y, cmap=cmap, **data_scatter_kwargs)

        #add colorbar
        fig.colorbar(mappable, ax=ax1)

        axs = fig.axes

        return fig, axs