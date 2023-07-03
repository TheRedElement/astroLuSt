

#%%imports
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy as np
from typing import Union, Tuple, Any, Callable

#%%classes
class BUBBLES:

    def __init__(self,
        func:Union[str,Callable],
        n_jobs:int=-1,
        verbose:int=0,
        ) -> None:
        
        if func == 'sphere':    self.func   = self.__sphere_nd
        elif func == 'rect':    self.func   = self.__rect_nd
        elif func is None:      self.func   = self.__sphere_nd
        else:                   self.func   = func
        if n_jobs is None:      self.n_jobs = 1
        else:                   self.n_jobs = n_jobs

        self.verbose = verbose
        return

    def __repr__(self) -> str:
        return (
            f'BUBBLES(\n'
            f')'
        )
    
    def get_most_common(self,
        x:np.ndarray
        ) -> Any:
        uniques, counts = np.unique(x, return_counts=True)
        most_common = uniques[np.argmax(counts)]
        
        return most_common

    def __sphere_nd(self,
        X:np.ndarray, y:np.ndarray=None,
        X_grid:np.ndarray=None, 
        r0:float=None, min_pts:int=None,
        fit:bool=True,
        **kwargs,
        ) -> None:

        #while fitting
        if fit:
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
        
        if fit:
            y_bool = np.all((
                (X_grid-r0/2 < X)&
                (X < X_grid+r0/2)
            ), axis=1)

            if y_bool.sum() > min_pts:  y_grid = self.get_most_common(y[y_bool])
            else:                       y_grid = np.nan

            Xy_grid = np.append(X_grid, y_grid)

            return Xy_grid
        
        else:
            y_bool = np.all((
                (self.X_grid-self.r0/2 < X)&
                (X < self.X_grid+self.r0/2)
            ), axis=1)

            if y_bool.sum() > 0:
                y_pred = self.get_most_common(self.y_grid[y_bool])
            else:
                y_pred = -1


            return y_pred


    def fit(self,
        X:np.ndarray, y:np.ndarray,
        res:Union[int,tuple]=10,
        func:Union[str,Callable]=None,
        min_pts:int=0, r0:float=None,
        n_jobs:int=None,
        verbose:int=None,
        ) -> None:
        
        #default values
        if func == 'sphere':    func    = self.__sphere_nd
        elif func == 'rect':    func    = self.__rect_nd
        elif func is None:      func    = self.func

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


        if n_jobs is None:      n_jobs  = self.n_jobs 
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
        ) -> np.ndarray:
        
        return
    
    def plot_result(self,
        X:np.ndarray=None, y:np.ndarray=None,
        dims:list=None,
        cmap:Union[str,mcolors.Colormap]=None,
        grid_scatter_kwargs:dict=None, data_scatter_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:

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