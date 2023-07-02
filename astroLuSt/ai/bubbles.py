

#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Tuple

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

    def fit(self,
        X:np.ndarray, y:np.ndarray,
        r0:float=None,
        ) -> None:
        
        print(X.shape)
        uidxs = np.array([np.where(y==yu)[0] for yu in np.unique(y)])

        x0 = np.zeros((uidxs.shape[0], X.shape[1]))

        for idx, ui in enumerate(uidxs):
            x0[idx] = X[np.random.choice(ui),:]

        sphs = np.zeros((X.shape[0],100,X.shape[1]))
        print(sphs.shape)
        for idx, x in enumerate(X):
            # print(x.shape)
            sphs[idx] = self.nd_sphere(x, r0)


        sphs = np.sum(sphs, axis=(1))/sphs.shape[1]
        print(sphs.shape)
    
        
        print(X.shape, x0.shape)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(*X.T, c=y, alpha=0.5, s=10)
        ax1.scatter(*x0.T, marker='v')
        ax1.scatter(sphs[:,0], sphs[:,1], s=5, marker='s')
        # for sph in sphs:
        #     ax1.plot(sph[:,0], sph[:,1], color='tab:blue')
        

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
        ) -> Tuple[Figure,plt.Axes]:

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        axs = fig.axes

        return fig, axs