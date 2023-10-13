#%%imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

from typing import Union, Tuple, Callable

#%%classes
class TPF:

    def __init__(self,
        size:Union[Tuple,int],
        ) -> None:

        if isinstance(size, int): size = (size,size)

        x = np.arange(size[0])
        y = np.arange(size[1])
        xx, yy = np.meshgrid(x, y)

        self.frame = np.zeros(shape=(*size,1))
        self.frame = np.concatenate((np.expand_dims(xx,2), np.expand_dims(yy,2), self.frame), axis=2)

        pass

    
    def add_noise(self,
        amp:float=1,
        ):

        self.frame[:,:,2] += np.random.randn(*self.frame.shape[:2])

        return
    
    def add_stars(self,
        amp:float=1,
        ):

        star = self.star(pos=np.array([10,10]), m=10)

        self.frame[:,:,2] += star

        return
    
    def star(self,
        pos:np.ndarray,
        m:Union[float,np.ndarray]=None, f:Union[float,np.ndarray]=None
        ):

        star = sps.multivariate_normal(
            mean=pos, cov=m, allow_singular=True
        ).pdf(self.frame[:,:,:2])

        return star
    
    def plot_result(self,
        ):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        mesh = ax1.pcolormesh(self.frame[:,:,0], self.frame[:,:,1], self.frame[:,:,2], cmap='viridis_r')
        # ax1.scatter(*star_targ[:2]+.5, marker='x', c='r', label='Target')

        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')

        cbar = fig.colorbar(mesh, ax=ax1)
        cbar.ax.invert_yaxis()
        # cbar.ax.set_ylim(0-mag_range/2-1)
        cbar.set_label('Magnitude [mag]')

        ax1.legend()

        axs = fig.axes

        return fig, axs

#%%definitions
