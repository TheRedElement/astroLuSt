#%%imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

from astroLuSt.physics import photometry as alpp

from typing import Union, Tuple, Callable, Literal

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
        
        amp *= 1E-3
        self.frame[:,:,2] += amp*np.random.randn(*self.frame.shape[:2])

        return
    
    def add_stars(self,
        pos:Union[np.ndarray,Literal['random']],
        f:np.ndarray=None,
        random_config:dict=None,
        ):

        default_random_config = {
            'nstars':1,
            'sizex':self.frame.shape[0],
            'sizey':self.frame.shape[1],
            'fmin':0,
            'fmax':100,
            'apmin':1,
            'apmax':10,
        }
        if random_config is None:
            random_config = default_random_config
        else:
            for k in random_config.keys(): 
                default_random_config[k] = random_config[k]
                random_config = default_random_config



        if pos == 'random':
            posx        = np.random.choice(range(random_config['sizex']), size=(random_config['nstars'],1))
            posy        = np.random.choice(range(random_config['sizey']), size=(random_config['nstars'],1))
            pos         = np.append(posx, posy, axis=1)
            f           = np.random.choice(range(random_config['fmin'],  random_config['fmax'], 1), size=(random_config['nstars']))
            aperture    = np.random.choice(range(random_config['apmin'], random_config['apmax'],1), size=(random_config['nstars']))
        elif pos != 'random' and f is None:
            raise ValueError("`f` has to be provided is `pos` is not `'random'`")
            

        for posi, fi, api in zip(pos,f, aperture):
            star = self.star(pos=posi, f=fi, aperture=api)

            self.frame[:,:,2] += star

        return
    
    def star(self,
        pos:np.ndarray,
        f:float=None, m:float=None,
        aperture:float=1,
        ):

        if f is None:
            if m is not None:
                f =  alpp.mags2fluxes(m=m)
            else:
                raise ValueError(f'At least one of `f` and `m` has to be not `None` but they have values {f} and {m}.')
            

        cov = aperture/2*2.335      #aperture = 2*halfwidth
        star = f*sps.multivariate_normal(
            mean=pos, cov=cov, allow_singular=True
        ).pdf(self.frame[:,:,:2])

        return star
    
    def plot_result(self,
        ):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        mesh = ax1.pcolormesh(self.frame[:,:,0], self.frame[:,:,1], self.frame[:,:,2], cmap='viridis')
        # ax1.scatter(*star_targ[:2]+.5, marker='x', c='r', label='Target')

        ax1.set_xlabel('Pixel')
        ax1.set_ylabel('Pixel')

        cbar = fig.colorbar(mesh, ax=ax1)
        # cbar.ax.invert_yaxis()
        # cbar.ax.set_ylim(0-mag_range/2-1)
        cbar.set_label('Flux [-]')

        ax1.legend()

        axs = fig.axes

        return fig, axs


class TPF_Series:

    def __init__(self) -> None:
        pass


#%%definitions