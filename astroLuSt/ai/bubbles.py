

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


    def nd_sphere(self,
        x0:np.ndarray, r:float,
        ) -> np.ndarray:

        x = r*np.cos(np.linspace(0,2*np.pi,100))
        y = r*np.sin(np.linspace(0,2*np.pi,100))

        xy = np.array([x,y]).T+x0

        return xy

    def __rect_2d(self,
        y_grid:np.ndarray, X_grid:np.ndarray,
        X:np.ndarray, y:np.ndarray,
        min_pts:int,
        ) -> np.ndarray:

        for idx, Xi in enumerate(X_grid):
            y_bool

        # for i in range(len(X_grid)-1):
        #     for j in range(len(X_grid[i])-1):
        #         y_bool = (
        #             (X[:,0] > X_grid[i  ,j  ][0]) &
        #             (X[:,1] > X_grid[i  ,  j][1]) &
        #             (X[:,0] < X_grid[i+1,j+1][0]) &
        #             (X[:,1] < X_grid[i+1,j+1][1]) 
        #         )

        #         if y_bool.sum() > min_pts:
        #             y[i, j] = self.get_most_common(y[y_bool])


        return y_grid

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


        # self.y_grid = self.__rect_nd(
        #     X=X, y=y,
        #     min_pts=min_pts,
        # )
        self.__sphere_nd(
            X=X, y=y,
            min_pts=min_pts,
            r0=r0,
        )

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
        dims:Union[list,slice]=None,
        ) -> Tuple[Figure,plt.Axes]:

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        sctr1 = ax1.scatter(*self.X_grid[:,dims].T, c=self.y_grid, alpha=0.1, s=10)
        if X is not None:
            sctr2 = ax1.scatter(*X[:,:3].T, c=y, alpha=0.5, s=50, ec='w', vmin=-1)


        fig.colorbar(sctr1)

        axs = fig.axes

        return fig, axs