

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
        - TODO: NAME
        - supervised classifier
        - estimates decision-boundaries based on grid with finite number of datapoints
        - also is able to find outliers

        Attributes
        ----------
            - `func`
                - str, callable
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
                        - `y`
                            - np.ndarray, optional
                            - labels corresponding to `X`
                            - required during `fit`
                            - optional during `predict`
                        - `X_grid`
                            - np.ndarray, optional
                            - has to have as many features as `X`
                            - points of the grid to use for decision boundary estimation
                            - will become `self.X_grid` once the calculation is done
                        - `r0`
                            - float, optional
                            - some distance measure
                            - required during `fit`
                            - optional during `predict`
                        - 'min_pts`
                            - int, optional
                            - measure of minimum number of points in `X` any point in `X_grid` has to be close to in order to be classified  NOT as an outlier
                            - required during `fit`
                            - optional during `predict`
                        - `fit`
                            - bool, optional
                            - flag of classifier state
                            - `True` for fitting mode
                            - `False` for inference mode
                        - `**kwargs`
                            - dict
                            - additional kwargs to be passed to `func`
                    - note that optional arguments must be optional in the definition of `func` as well!
                    - has to return the following
                        - during fitting
                            - `Xy_grid`
                                - np.ndarray
                                - 1d array of shape `(n_features+1)`
                                - contains coordinates of the tested point in `self.X_grid` as fisrt elements
                                - contains the estimated label for that point as last element
                        - during inference
                            - `y_pred`
                                - float
                                - predicted label for the point `X` based on `self.X_grid` and `self.y_grid`
                - the default is `None`
                    - will use `sphere`
            - `r0`
                - float
                - radius around any point in `X_grid` (during fitting) or `self.X_grid` (during inference) in which any point in `X` has to lie to be considered close to that grid-point
                - during fitting
                    - the label for any point in `X_grid` will be assigned by the dominant label in `X` that is within the radius `r0`
                    - if the point in `X_grid` contains no points out of `X`, it will be considered as noise and set to `np.nan`
                - during inference
                    - any point in `X` will be tested against the not-noise-points in `self.X_grid`
                        - if it falls within the radius of any point in `self.X_grid`, it will get the corresponding label assigned to it
                        - otherwise it will be assigned the label `-1`, which means that the datapoint is noise
            - `min_pts`
                - int, optional
                - minimum number of points a sphere around any point in `X_grid` has to contain to not be considered noise
                - i.e. if the point contains less than `min_pts` points in its `r0` neighborhood, it will be considered as noise and dropped from  the grid
                - the default is 0
            - `res`
                - int, optional
                - resolution of the `X_grid` in all dimensions
                - will be passed to `np.linspace()`
                - high values for `res` are especially bad in high dimensions
                    - i.e. "Curse of dimensionality" (https://en.wikipedia.org/wiki/Curse_of_dimensionality)
                - the default is 10

        Methods
        -------
            - get_parameters()
            - get_most_common()
            - fit()
            - predict()
            - fit_predict()
            - score()
            - plot_result()

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

        self.allowed_funcs = ['sphere', 'rect']        
        if isinstance(func, str):
            assert func in self.allowed_funcs, f'if `func` is a string it has to be in {self.allowed_funcs}!'
        if func is None:        self.func   = 'sphere'
        # if func == 'sphere':    self.func   = self.__sphere_nd
        # elif func == 'rect':    self.func   = self.__rect_nd
        # elif func is None:      self.func   = self.__sphere_nd
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
    
    def __dict__(self) -> dict:
        
        d = dict(
            func=self.func,
            r0=self.r0, min_pts=self.min_pts,
            res=self.res,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        return d

    def __sphere_nd(self,
        X:np.ndarray, y:np.ndarray=None,
        X_grid:np.ndarray=None, 
        r0:float=None, min_pts:int=None,
        fit:bool=True,
        **kwargs,
        ) -> Union[np.ndarray,Any]:
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
        ) -> Union[np.ndarray,Any]:
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

    def get_parameters(self,
        deep:bool=True,
        ) -> dict:
        """
            - function to get current parameters of the classifier
        
        Parameters
        ----------
            - deep
                - bool optional
                - if True will return parameters for estimator and contained subobjects that are estimators
                - no effect as of now
                    - here just to achieve structure as sklearn
                - the default is True

        Raises
        ------

        Returns
        -------
            - params
                - dict
                - parameters of the estimator

        Comments
        --------


        """

        params = self.__dict__()
        
        return params

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
                            - `y`
                                - np.ndarray, optional
                                - labels corresponding to `X`
                                - required during `fit`
                                - optional during `predict`
                            - `r0`
                                - float, optional
                                - some distance measure
                                - required during `fit`
                                - optional during `predict`
                            - 'min_pts`
                                - int, optional
                                - measure of minimum number of points in `X` any point in `X_grid` has to be close to in order to be classified  NOT as an outlier
                                - required during `fit`
                                - optional during `predict`
                            - `fit`
                                - bool, optional
                                - flag of classifier state
                                - `True` for fitting mode
                                - `False` for inference mode
                            - `**kwargs`
                                - dict
                                - additional kwargs to be passed to `func`
                        - note that optional arguments must be optional in the definition of `func` as well!
                        - has to return the following
                            - during fitting
                                - `Xy_grid`
                                    - np.ndarray
                                    - 1d array of shape `(n_features+1)`
                                    - contains coordinates of the tested point in `self.X_grid` as fisrt elements
                                    - contains the estimated label for that point as last element
                            - during inference
                                - `y_pred`
                                    - float
                                    - predicted label for the point `X` based on `self.X_grid` and `self.y_grid`                    
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
        if func is None:        func    = self.func
        if r0 is None:          r0      = self.r0
        if min_pts is  None:    min_pts = self.min_pts
        if res is None:         res     = self.res

        if n_jobs is None:      n_jobs  = self.n_jobs 

        if verbose is None: verbose = self.verbose

        #adopt hyperparameters
        self.r0 = r0
        self.min_pts = min_pts
        self.func = func

        #use correct function
        if func == 'sphere':    func = self.__sphere_nd
        elif func == 'rect':    func = self.__rect_nd
        else:                   func = func

        #generate grid from linspaces
        X_base = np.linspace(np.nanmin(X, axis=0), np.nanmax(X, axis=0), res)
        X_points = np.array(np.meshgrid(*X_base.T))

        if verbose > 2:
            print(f'INFO(BUBBLES.fit): Generated X_points of shape: {X_points.shape}')
        
        #2d array of all points in X_points
        X_points = np.vstack([np.ravel(Xg) for Xg in X_points]).T

        #assign labels to grid-points
        Xy_grid = np.array(Parallel(n_jobs=n_jobs, verbose=verbose, prefer='threads')(
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

        #default values
        if n_jobs is None:  n_jobs  = self.n_jobs 
        if verbose is None: verbose = self.verbose

        #use correct function
        if self.func == 'sphere':   func = self.__sphere_nd
        elif self.func == 'rect':   func = self.__rect_nd
        else:                       func = self.func

        y_pred = np.array(Parallel(n_jobs=n_jobs, verbose=verbose, prefer='threads')(
            delayed(func)(
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
        features:list=None,
        cmap:Union[str,mcolors.Colormap]=None,
        grid_scatter_kwargs:dict=None, data_scatter_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the result of a fitted classifier

            Parameters
            ----------
                - `X`
                    - np.ndarray, optional
                    - data to plot into the plot of `self.X_grid`
                    - the default is `None`
                        - will only plot datapoints in `self.X_grid` (i.e. the decision boundary)
                - `y`
                    - np.ndarray, optional
                    - labels corresponding to `X`
                    - will be used to color the datapoint in `X`
                    - only relevant if `X` is passed as well
                    - the default is `None`
                        - no coloring
                - `features`
                    - list, optional
                    - indices of the features of `self.X_grid` and `X` to plot
                    - can have a maximum length of 3
                    - if you want to only look at one feature simply pass `[0,0]` for the 0-th feature
                    - the default is `None`
                        - will be set to [0,1]
                        - i.e. plots the first two feature
                - `cmap`
                    - str, mcolors.Colormap, optional
                    - colormap to use for encoding the labels (`self.y_grid`, `y`)
                    - the default if `None`
                        - will be set to `Accent`
                - `grid_scatter_kwargs`
                    - dict, optional
                    - kwargs to pass `ax.scatter()` for plotting `self.X_grid` and `self.y_grid`
                    - the default is `None`
                        - will be set to {'alpha':0.5, 'vmin':-1}
                - `data_scatter_kwargs`
                    - dict, optional
                    - kwargs to pass `ax.scatter()` for plotting `X` and `y`
                    - the default is `None`
                        - will be set to {'alpha':0.5, 'vmin':-1, 'ec':'w'}

            Raises
            ------

            Returns
            -------
                - `fig`
                    - Figure
                    - created figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------
        """

        #default values
        if features is None:             features            = [0,1]
        if cmap is None:                 cmap                = 'Accent'
        if grid_scatter_kwargs is None:  grid_scatter_kwargs = {'alpha':0.5, 'vmin':-1,}
        if data_scatter_kwargs is None:  data_scatter_kwargs = {'alpha':0.5, 'vmin':-1, 'ec':'w'}

        #select plotting dimension based on self.X_grid and dims
        if self.X_grid.shape[1] < 3 or len(features) < 3:
            projection = None
            features = features[:2]
        else:
            projection = '3d'

        #plotting
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection=projection)

        #plot grid
        mappable = ax1.scatter(*self.X_grid.T[features], c=self.y_grid, cmap=cmap, **grid_scatter_kwargs)
        
        #if a dataset has been passed plot that as well
        if X is not None:
            ax1.scatter(*X[:,features].T, c=y, cmap=cmap, **data_scatter_kwargs)

        #add colorbar
        cbar = fig.colorbar(mappable, ax=ax1)
        cbar.set_label('Class')

        #labelling
        ax1.set_xlabel(f'X[:,{features[0]}]')
        ax1.set_ylabel(f'X[:,{features[1]}]')
        if projection=='3d': ax1.set_zlabel(f'X[:,{features[2]}]')

        axs = fig.axes

        return fig, axs