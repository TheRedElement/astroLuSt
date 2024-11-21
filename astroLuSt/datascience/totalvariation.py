
#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import scipy
from typing import Literal, Tuple

from astroLuSt.monitoring import formatting as almofo

#%%definitions
class TotalVariation:
    """
        - class implementing solving the Total Variation optimization problem (i.e., [Chambolle et al., 2016](https://www.cambridge.org/core/journals/acta-numerica/article/an-introduction-to-continuous-optimization-for-imaging/1115AA7E36FC201E811040D11118F67F))
        - uses the Chambolle-Pock Primal-Dual algorithm for the solution ([Chambolle et al., 2011](https://ui.adsabs.harvard.edu/abs/2011JMIV...40..120C/abstract))

        Attributes
        ----------
            - `lbda`
                - `float`
                - regularization parameter on the reconstruction term
                - high `lbda` will lead to a result very similar to the input
            - `tau`
                - `float`, optional
                - stepsize for updating the primal variable in CPA
                - used to set dual step-size `sigma`
                - the default is `1.0`
            - `theta`
                - `float`, optional
                - stepsize for the extrapolation/overrelaxation step of the primal variables
                - has to be greater than `0`
                - the default is `1.0`
            - `norm`
                - `Literal["l1","l2"]`, optional
                - norm to use in the reconstruction term
                - the default is `None`
                    - will be set to `"l2"`
            - `min_iter`
                - `int`, optional
                - minimum number of iterations to be completed by the algorithm
                - the default is `0`
            - `max_iter`
                - `int`, optional
                - maximum number of iterations the algorithm shall optimize for
                - the default is `100`
            - `eps`
                - `float`, optional
                - tolerance parameter to check for convergence
                - algorithm considered converged if `self.energy` changes less than `eps` for two consecutive iterations
                - the default is `0`
                    - will iterate until `max_iter` is reached
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Infered Attributes
        ------------------
            - `u`
                - `np.ndarray`
                - cleaned version of the original input
            - `energy`
                - `list`
                - energy (i.e. objective) for each iteration

        Methods
        -------
            - `get_energy()`
            - `get_gradient()`
            - `get_nabla()`
            - `apply_proxmap_primal()`
            - `fit()`
            - `transform()`
            - `fit_transform()`
            - `plot_result()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `scipy`
            - `typing`

        Comments
        --------
    """

    def __init__(self,
        lbda:float, tau:float=1.0,
        theta:float=1,
        norm:Literal["l1","l2"]=None,
        min_iter:int=1, max_iter:int=100, eps:float=0.0,
        verbose:int=0,
        ) -> None:

        assert norm in ["l1","l2"], "`norm` has to be on of 'l1', 'l2'!"

        self.lbda       = lbda
        self.tau        = tau
        self.theta      = theta
        if norm is None:    self.norm = "l2"
        else:               self.norm = norm
        self.min_iter   = min_iter
        self.max_iter   = max_iter
        self.eps        = eps
        self.verbose    = verbose

        #operator norm estimate (Chambolle2016, eq. 2.5, https://www.cambridge.org/core/journals/acta-numerica/article/an-introduction-to-continuous-optimization-for-imaging/1115AA7E36FC201E811040D11118F67F)
        K = np.sqrt(8)
        self.sigma = 1/tau/K**2

        #infered attributes
        self.energy = []
        
        return
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    lbda={self.lbda}, tau={self.tau},\n"
            f"    theta={self.theta},\n"
            f"    norm={self.norm},\n"
            f"    min_iter={self.min_iter}, max_iter={self.max_iter}, eps={self.eps},\n"
            f"    verbose={self.verbose},\n"
            f")"
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def get_energy(self,
        u:np.ndarray, g:np.ndarray,
        Du:np.ndarray,
        lbda:float,
        ) -> float:
        """
            - method to compute the Total Variation energy (objective)

        Parameters
        ----------
            - `u`
                - `np.ndarray`
                - result of current iteration
            - `g`
                - `np.ndarray`
                - original (noisy) input
            - `Du`
                - `np.ndarray`
                - gradient applied onto `u`
            - `lbda`
                - `float`
                - penalty parameter for reconstruction term

        Raises
        ------

        Returns
        -------
            - `nrg`
                - `float`
                - energy
                - i.e. evaluated objective
                    - $||Du||_1 + lbda/2*||u - g||_2^2$ for `self.norm == "l2"`
                    - $||Du||_1 + lbda/2*||u - g||_1^2$ for `self.norm == "l1"`

        Comments
        --------            
        """

        #total variation term
        Du = Du.reshape(2,*u.shape)
        t1 = np.sum(np.sqrt(Du[0,:]**2 + Du[1,:]**2))
        
        #reconstruction term
        if self.norm == "l1":
            t2 = lbda/2 * np.sum(np.abs(u - g))
        elif self.norm == "l2":
            t2 = lbda/2 * np.sum((u - g)**2)

        #combine
        nrg = t1 + t2

        return nrg

    def get_gradient(self,
        X:np.ndarray
        ) -> np.ndarray:
        """
            - method to compute the gradient of some 2d array `X` in terms of finite differences
            - follows Eq. 2.4 in [Chambolle et al., 2016](https://www.cambridge.org/core/journals/acta-numerica/article/an-introduction-to-continuous-optimization-for-imaging/1115AA7E36FC201E811040D11118F67F)

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - image to compute the finite differences from
            
            Raises
            ------

            Returns
            -------
                - `DX`
                    - `np.ndarray`
                    - 3d
                    - gradient applied to `X` in x- and y-direction
            
            Comments
            --------
        """
        rows, cols = X.shape
        DX = np.zeros((rows, cols, 2))
        
        Dx = X[:-1,:] - X[1:,:]
        Dy = X[:,:-1] - X[:,1:]

        DX[:-1,:,0] = Dx
        DX[:,:-1,1] = Dy

        return DX

    def get_nabla(self,
        m:int, n:int
        ) -> np.ndarray:
        """
            - method to construct a sparse matrix, the matrix multiplication of which results in a finite difference operation
            - essentially implements the operator to achieve Eq. 2.4 from [Chambolle et al., 2016](https://www.cambridge.org/core/journals/acta-numerica/article/an-introduction-to-continuous-optimization-for-imaging/1115AA7E36FC201E811040D11118F67F)
            - can only be applied to vectorized images of size `(m*n)`
                - `m` rows
                - `n` columns
            
            Parameters
            ----------
                - `m`
                    - `int`
                    - number of rows the image the finite differences shall be applied to has
                - `n`
                    - `int`
                    - number of columns the image the finite differences shall be applied to has

            Raises
            ------

            Returns
            -------
                - `D`
                    - `scipy.sparse.coo_matrix`
                    - sparse representation of the gradient
                    - can be applied to some input `X` like so
                        - `D@X`
                        - where `X` is a vectorized image with shape `(m*n)`

            Comments
            --------
            
        """

        #gradient in x-direction
        vals_x = np.ones(m*n)               #values
        row_x = np.arange(m*n)              #row indices of `vals_x`
        col_x = np.arange(m*n)              #col indices of `vals_x`
        vals_x[range(-1,m*n,n)] = 0         #set last row to 0
        col_x_shift = col_x + 1             #offset columns
        Dx = scipy.sparse.coo_matrix((vals_x[:-1], (row_x[:-1], col_x[:-1])), shape=(m*n,m*n)) \
            - scipy.sparse.coo_matrix((vals_x[:-1], (row_x[:-1], col_x_shift[:-1])), shape=(m*n,m*n))

        #gradient in y-direction
        vals_y = np.ones((m-1)*n)   #values
        row_y = np.arange((m-1)*n)  #row indices of `vals_y`
        col_y = row_y               #col indices of `vals_y`
        col_y_shift = (col_y) + n   #offset columns
        Dy = scipy.sparse.coo_matrix((vals_y, (row_y, col_y)), shape=(m*n,m*n)) \
            - scipy.sparse.coo_matrix((vals_y, (row_y, col_y_shift)), shape=(m*n,m*n))

        #merged
        D = scipy.sparse.vstack([Dx, Dy])
        # print(D.shape)

        return D

    def apply_proxmap_primal(self,
        u:np.ndarray, g:np.ndarray
        ) -> np.ndarray:
        """
            - method to apply the proximal operator (prox) to `u`
            - relevant for updating the primal variable `u`
            - will choose the respective prox depending on `self.norm`

            Parameters
            ----------
                - `u`
                    - `np.ndarray`
                    - primal variable
                        - i.e., cleaned input
                - `g`
                    - `np.ndarray`
                    - original (noisy) input
            
            Raises
            ------
                - `ValueError`
                    - if an invalid `self.norm` is provided

            Returns
            -------
                - `u`
                    - `np.ndarray`
                    - primal variable after applying respective proxmap

        """

        if self.norm == "l1":
            u = g + np.maximum(0.0, np.abs(u-g)-self.tau*self.lbda)*np.sign(u-g)    #proxmap
            # u = f + np.maximum(0.0, np.abs(u-f)-tau*lamb)*np.sign(u-f)
        elif self.norm == "l2":
            u = (u + self.tau*self.lbda*g)/(1+self.tau*self.lbda)
        else:
            raise ValueError("`norm` has to be one of 'l1', 'l2'!")

        return u

    def fit(self,
        X:np.ndarray,
        verbose:int=None,
        *args, **kwargs,
        ) -> None:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - has to be 2d
                    - image onto which Total Variation shall be applied
                - `verbose`
                    - `int`, optional
                    - verbosity level
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

        #default parameters
        if verbose is None: verbose = self.verbose

        m, n = X.shape

        #get nabla (finite differences)
        nabla = self.get_nabla(m, n)

        #init (vectorized)
        g = np.copy(X.flatten())    #noisy image
        u = np.copy(X.flatten())    #primal variable (clean image)
        p = np.zeros(m*n*2)         #dual variable (2 because x- and y-direction)

        #iterate
        for k in range(0, self.max_iter):

            u_bar = u   #old u

            #primal update
            u = u - self.tau*(nabla.T@p)
            u  = self.apply_proxmap_primal(u, g)    #select proxmap based on norm used in the reconstruction term

            #extrapolation step (overrelaxation)
            u_bar = u + self.theta*(u - u_bar)

            #dual update
            p_bar = (p + self.sigma*nabla@u_bar).reshape(2,m*n)
            p_norm = np.sqrt(p_bar[0,:]**2 + p_bar[1,:]**2)
            p = p_bar/np.maximum(1, p_norm)
            p = p.flatten()

            #store energy
            self.energy.append(self.get_energy(u_bar, g, nabla@u_bar, self.lbda))
            delta = abs(self.energy[k]-self.energy[k-1])/abs(self.energy[k-1])  #fractional change of energy

            #logging
            almofo.printf(
                msg=f'Iteration {k+1} with delta={delta:8.1e}, energy={abs(self.energy[k]):9.2e}.',
                context=f'{self.__class__.__name__}.{self.fit.__name__}',
                type='INFO',
                level=0,
                verbose=verbose-1
            )            

            #check convergence
            if (delta < self.eps) and (k > (self.min_iter-1)):
                almofo.printf(
                    msg=f'Converged after {k+1} iterations with delta={delta:7.1e}, energy={self.energy[k]:9.2e}.',
                    context=f'{self.__class__.__name__}.{self.fit.__name__}',
                    type='INFO',
                    level=0,
                    verbose=verbose
                )
                break

        #print finishing message
        almofo.printf(
            msg=f'Finished after {k+1} iterations with delta={delta:7.1e}, energy={self.energy[k]:9.2e}.',
            context=f'{self.__class__.__name__}.{self.fit.__name__}',
            type='INFO',
            level=0,
            verbose=verbose
        )

        #store result
        self.u = u.reshape(m, n)

        return

    def transform(self,
        *args, **kwargs,
        ) -> np.ndarray:
        """
            - method to tranform the input

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `u`
                    - `np.ndarray`
                    - primal variable after application of Total Variation
                    - i.e., denoised image

            Comments
            --------
        """

        u = self.u

        return u
    
    def fit_transform(self,
        X:np.ndarray,
        verbose:int=None,
        *args, **kwargs
        ) -> np.ndarray:
        """
        - method to fit the transformer and transform the input in one go

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - has to be 2d
                    - image onto which Total Variation shall be applied
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`

            Raises
            ------

            Returns
            -------
                - `u`
                    - `np.ndarray`
                    - primal variable after application of Total Variation
                    - i.e., denoised image

            Comments
            --------
        """


        self.fit(X, verbose=verbose)
        u = self.transform()

        return u
    
    def plot_result(self,
        X_in:np.ndarray,
        fig:Figure=None,
        pcolormesh_kwargs:dict=None,
        plot_kwargs:dict=None,
        verbose:int=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to visualize the result after applying the transformer

            Parameters
            ----------
                - `X_in`
                    - `np.ndarray`
                    - original (noisy) input
                - `fig`
                    - `matplotlib.figure.Figure`, optional
                    - figure to plot into
                    - ideally pass an empty figure as 3 new axes will be created
                    - the default is `None`
                        - will create a new figure
                - `verbose` 
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict(cmap='grey')`
                        - will fall back to `self.verbose`
                - `plot_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.plot()`
                    - the default is `None`
                        - will be set to `dict()`

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
        """

        #default parameters
        if verbose is None: verbose = self.verbose
        if pcolormesh_kwargs is None:       pcolormesh_kwargs           = dict()
        if 'cmap' not in pcolormesh_kwargs: pcolormesh_kwargs['cmap']   = 'grey'
        if plot_kwargs is None:             plot_kwargs                 = dict()


        if fig is None:
            fig = plt.figure()
        
        ax1 = fig.add_subplot(132, title=r"$X_\mathrm{in}$")
        ax2 = fig.add_subplot(133, title=r"$u$")
        ax3 = fig.add_subplot(134, title="Energy")
        
        ax1.pcolormesh(X_in, **pcolormesh_kwargs)
        ax2.pcolormesh(self.u, **pcolormesh_kwargs)
        ax3.plot(self.energy, **plot_kwargs)

        ax1.set_xlabel("Pixel")
        ax2.set_xlabel("Pixel")
        ax3.set_xlabel("Iteration")
        ax1.set_ylabel("Pixel")
        ax2.set_ylabel("Pixel")
        ax3.set_ylabel("Energy")

        axs = fig.axes

        return fig, axs
    

    