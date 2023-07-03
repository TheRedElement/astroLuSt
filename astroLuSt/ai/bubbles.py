

#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Tuple, Any

#%%classes
class BUBBLES:

    def __init__(self) -> None:
        
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

    def __rect_nd(self,
        X:np.ndarray, y:np.ndarray,
        r0:float, min_pts:int,
        ) -> None:
        
        X_min = self.X_grid-r0/2
        X_max = self.X_grid+r0/2
        for idx, (x_min, x_max) in enumerate(zip(X_min, X_max)):
            
            y_bool = np.all((
                (x_min < X)&
                (X < x_max)
            ), axis=1)

            if y_bool.sum() > min_pts:
                self.y_grid[idx] = self.get_most_common(y[y_bool])

        return

    def __sphere_nd(self,
        X:np.ndarray, y:np.ndarray,
        min_pts:int,
        r0:float,
        ) -> None:
        for idx, Xi in enumerate(self.X_grid):
            diff = np.sum(np.sqrt((Xi-X)**2), axis=1)

            y_bool = (diff < r0)

            if y_bool.sum() > min_pts:
                self.y_grid[idx] = self.get_most_common(y[y_bool])

        return

    def fit(self,
        X:np.ndarray, y:np.ndarray,
        res:Union[int,tuple]=10, min_pts:int=0,
        r0:float=None,
        ) -> None:
        
        print(X.shape)
        X_base = np.linspace(np.nanmin(X, axis=0), np.nanmax(X, axis=0), res)
        print(f'X_base.shape: {X_base.shape}')

        self.X_grid = np.array(np.meshgrid(*X_base.T))
        
        #2d array of points for X_grid
        self.X_grid = np.vstack([np.ravel(Xg) for Xg in self.X_grid]).T
        print(f'X_grid.shape: {self.X_grid.shape}')

        self.y_grid = np.zeros(self.X_grid.shape[:-1])-1
        print(f'y_grid.shape: {self.y_grid.shape}')


        self.__rect_nd(
            X=X, y=y,
            min_pts=min_pts,
            r0=r0,
        )
        # self.__sphere_nd(
        #     X=X, y=y,
        #     min_pts=min_pts,
        #     r0=r0,
        # )

        return
    
    def predict(self,
        X:np.ndarray, y:np.ndarray,
        ) -> np.ndarray:

        return
    
    def fit_predict(self,
        X:np.ndarray, y:np.ndarray,
        ) -> np.ndarray:
        
        return
    
    def plot_result(self,
        X:np.ndarray=None, y:np.ndarray=None,
        dims:list=None,
        remove_noise:bool=False,
        ) -> Tuple[Figure,plt.Axes]:

        if dims is None: dims = [0,1]

        cmap = 'Set2'
        if self.X_grid.shape[1] < 3 or len(dims) < 3: projection = None
        else:                                          projection = '3d'

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection=projection)

        
        # if self.X_grid.shape[1] == 2:
        #     # mappable = ax1.contourf(*self.X_grid.T, self.y_grid)
        #     mappable = ax1.tricontourf(*self.X_grid.T, self.y_grid)
        if remove_noise:
            mappable = ax1.scatter(*self.X_grid[(self.y_grid!=-1)].T[dims], c=self.y_grid[(self.y_grid!=-1)], alpha=0.5, s=10, vmin=-1, cmap=cmap)
        else:
            mappable = ax1.scatter(*self.X_grid[:].T[dims],                 c=self.y_grid,                    alpha=0.5, s=10, vmin=-1, cmap=cmap)
        

        if X is not None:
            ax1.scatter(*X[:,dims].T, c=y, alpha=0.5, s=50, ec='w', vmin=-1, cmap=cmap)


        fig.colorbar(mappable)

        axs = fig.axes

        return fig, axs