

#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Tuple

#%%classes
class BUBBLES:

    def __init__(self) -> None:
        
        return

    def __repr__(self) -> str:
        return (
            f'BUBBLES(\n'
            f')'
        )
    
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

        for i in range(len(X_grid)-1):
            for j in range(len(X_grid[i])-1):
                y_bool = (
                    (X[:,0] > X_grid[i  ,j  ][0]) &
                    (X[:,1] > X_grid[i  ,  j][1]) &
                    (X[:,0] < X_grid[i+1,j+1][0]) &
                    (X[:,1] < X_grid[i+1,j+1][1]) 
                )

                if y_bool.sum() > min_pts:
                    uniques, counts = np.unique(y[y_bool], return_counts=True)
                    y_grid[i, j] = uniques[np.argmax(counts)]


        return y_grid

    def fit(self,
        X:np.ndarray, y:np.ndarray,
        res:Union[int,tuple]=10, min_pts:int=0,
        r0:float=None,
        ) -> None:
        
        print(X.shape)
        X_base = np.linspace(np.nanmin(X, axis=0), np.nanmax(X, axis=0), res)
        print(f'X_base.shape: {X_base.shape}')

        X_grid = np.array(np.meshgrid(*X_base.T)).T
        print(f'X_grid.shape: {X_grid.shape}')

        y_grid = np.zeros(X_grid.shape[:-1])-1

        
        y_grid = self.__rect_2d(
            y_grid=y_grid, X_grid=X_grid,
            X=X, y=y,
            min_pts=min_pts,
        )

        

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cont1 = ax1.contourf(*X_grid.T, y_grid.T)
        # cont1 = ax1.contour(*X_grid.T, y_grid.T)
        ax1.scatter(*X.T, c=y, alpha=0.5, s=10, ec='w', vmin=-1)
        fig.colorbar(cont1)
        plt.show()

        return X_grid, y_grid
    
    def predict(self,
        X:np.ndarray, y:np.ndarray,
        ) -> np.ndarray:

        return
    
    def fit_predict(self,
        X:np.ndarray, y:np.ndarray,
        ) -> np.ndarray:
        
        return
    
    def plot_result(self,
        ) -> Tuple[Figure,plt.Axes]:

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        axs = fig.axes

        return fig, axs