
#NOTE: resource: https://www.dorukhanserg.in/post/implementing-rpca/
#NOTE: resource CP: https://nbviewer.org/github/gpeyre/numerical-tours/blob/master/python/optim_3_condat_fb_duality.ipynb


#%%imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation as manimation
from typing import Tuple, Literal, Callable, Any, Union

from astroLuSt.monitoring import formatting as almofo

#%%functions
def soft_thresholding(
    xbar:np.ndarray, tau:float=1,
    z:np.ndarray=None,
    ) -> np.ndarray:
    """
        - (element-wise) soft thresholding operator (equivalently soft shrinkage operator)
            - solution to $argmin_x \tau*||x-z||_1 + (1/2)*||x-xbar||_F^2$
            - see https://en.wikipedia.org/wiki/Proximal_operator
                - last accessed: 2024/05/08
                - here: $f(X) = \lambda*||x-z||_1$      
            - implemented following definition in [Parikh et al., 2013, Sec. 6.5.2](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)
                - last accessed: 2024/05/08
        
        Parameters
        ----------
            - `xbar` 
                - `np.ndarray`
                - target matrix
            - `tau`
                - `float`, optional
                - threshold parameter
                - scales width of thresholding region
                - the default is `1`
            - `z`
                - `np.ndarray`, optional
                - same shape as `xbar`
                - offset in the l1 norm
                - the default is `None`
                    - will be set to array of zeros
                    - no offset

        Raises
        ------

        Returns
        -------
            - `prox_l1`
                - `np.ndarray`
                - proximal operator applied element wise to `xbar`
                    - = elementwise soft thresholding operator $prox_{\tau\Vert\cdot\Vert_1}(xbar)$

        Dependencies
        ------------
            - `numpy`

        Comments
        --------

    """
    
    #default parameters
    if z is None: z = np.zeros_like(xbar)

    #NOTE: case for z=0, tau=1
    # prox_l1 = np.sign(xbar) *np.clip(np.abs(xbar) - lbda, a_min=0, a_max=None)
    
    #NOTE: generalized case
    prox_l1 = z + np.sign(xbar-z) * np.clip(np.abs(xbar-z) - tau, a_min=0, a_max=None)
    # prox_l1 = z + np.max([np.zeros_like(xbar), np.abs(xbar-z) - tau], axis=0)*np.sign(xbar-z)  #equivalent but less efficient



    return prox_l1

def svd_shrinkage(
    Y:np.ndarray, tau:float,
    ) -> np.ndarray:
    """
        - SVD shrinkage operator (also SVD thresholding)
            - solution to $argmin_X \tau*||X||_* (1/2)*||X-Y||_F^2$       
        - description/definition in [Cai et al., 2008, Eq. 2.2, Theorem 2.1](https://arxiv.org/pdf/0810.3286)
                - last accessed: 2024/05/08

        Parameters
        ----------
            - `Y`
                - `np.ndarray`
                - target matrix
            - `tau`
                - `float`
                - penalty parameter

        Raises
        ------

        Returns
        -------
            - `D_tau_Y`
                - `np.ndarray`
                - soft-thresholding operator evaluated on `Y`
                - equivalent to $prox_{\tau\Vert\cdot\Vert_\ast}(Y)$
        
        Dependencies
        ------------
            - `numpy`

        Comments
        --------
    """

    #decompose input
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    
    #calculate D_tau evaluated on Sigma (Cai et al., 2008, Eq. 2.2.)
    s_t = soft_thresholding(xbar=S, tau=tau)
    D_tau_S = np.diag(s_t) 

    #Recompose into D_tau evaluated on Y (Cai et al., 2008, Eq. 2.2.)
    D_tau_Y = (U@D_tau_S)@Vh

    return D_tau_Y

def sv_clip(
    xbar:np.ndarray,
    a_min:float=None, a_max:float=None,
    ) -> np.ndarray:
    """
        - function to transform a matrix into a version with singular values clipped to `a_min`  and `a_max`

        Parameters
        ----------
            - `xbar`
                - `np.ndarray`
                - matrix to be clipped
            - `a_min`
                - `float`, optional
                - lower bound for any singular value
                - the default is `None`
                    - no clipping
            - `a_max`
                - `float`, optional
                - upper bound for any singular value
                - the default is `None`
                    - no clipping

        Raises
        ------

        Returns
        -------
            - `x`
                - `np.ndarray`
                - clipped version of `xbar`

        Dependencies
        ------------
            - `numpy`
        
        Comments
        --------
    """
    

    #SVD
    U, S, Vh = np.linalg.svd(xbar, full_matrices=False)
    x_th = np.diag(np.clip(S, a_min=a_min, a_max=a_max))
    x = (U@x_th)@Vh

    return x

def energy(
        L:np.ndarray, S:np.ndarray, lbda:float
    ) -> float:
    """
        - function to compute the RPCA energy (objective)

        Parameters
        ----------
            - `L`
                - `np.ndarray`
                - has to be 2d
                - low-rank part of the original input
                
            - `S`
                - `np.ndarray`
                - has to be 2d
                - sparse part of the original input
            - `lbda`
                - `float`
                - penalty parameter for sparse errors (`S`)

        Raises
        ------

        Returns
        -------
            - `nrg`
                - `float`
                - energy
                - i.e. evaluated objective (||L||_* + lbda*||S||_1)

        Dependencies
        ------------
            - `numpy`
        
        Comments
        --------
    """

    nrg = np.linalg.norm(L, ord='nuc') + lbda*np.sum(np.abs(S))
    return nrg

#%%classes
class RPCA_ADMM:
    """
        - class implementing the solution of a Robust PCA via the Alternating Direction Method of Multipliers (ADMM)

        Attributes
        ----------
            - `lbda`
                - `float`
                - penalty parameter for sparse errors (`S`)
            - `mu`
                - `float`, optional
                - penalty of the augmented Lagrangian
                - similar to `sigma = 1/tau` in Chambolle-Pock algorithm (`RPCA_CP`)
                - has to be > 0
                - the default is `1`
            - `max_iter`
                - `int`, optional
                - maximum number of iterations the algorithm shall optimize for
                - the default is `100`
            - `eps`
                - `float`, optional
                - tolerance parameter to check for convergence
                - algorithm considered converged if `energy` changes less than `eps` for two consecutive iterations
                - the default is `0`
                    - will iterate until `max_iter` is reached
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
        
        Infered Attributes
        ------------------
            - `L`
                - `np.ndarray`
                - low-rank part of the input
            - `S`
                - `np.ndarray`
                - sparse part of the input
            - `energy`
                - `list`
                - energy (i.e. objective) for each iteration

        Methods
        -------
            - `fit()`
            - `transform()`
            - `fit_transform()`

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------

    """

    def __init__(self,
        lbda:float,
        mu:float=1.0,
        max_iter:int=100, eps:float=0,
        verbose:int=0,
        ) -> None:

        #check ranges
        assert lbda     > 0, "`lbda` has to be greater than 0!"
        assert mu       > 0, "`mu` has to be greater than 0!"     #primal stepsize (similar to `sigma=1/tau` in CP)
        assert max_iter > 0, "`max_iter` has to be greater than 0!"
        assert eps      >= 0,"`eps` has to be greater than or equal to 0!"

        #set attributes
        self.lbda       = lbda
        self.mu         = mu
        self.eps        = eps
        self.max_iter   = max_iter
        self.verbose    = verbose

        #output attributes
        self.L = None       #low-rank
        self.S = None       #sparse

        self.energy = []
        
        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    lbda={repr(self.lbda)},\n'
            f'    mu={repr(self.mu)},\n'
            f'    max_iter={repr(self.max_iter)}, eps={repr(self.eps)},\n'
            f'    verbose={repr(self.verbose)},\n'           
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def fit(self,
        X:np.ndarray,
        y:np.ndarray=None,
        L0:np.ndarray=None, S0:np.ndarray=None, Y0:np.ndarray=None,
        verbose:int=None,
        ) -> None:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input matrix
                    - target of optimization constraint
                        - = C in [Boyd et al., 2011, Eq. 3.1](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
                - `y`
                    - `np.ndarray`, optional
                    - targets
                    - not needed
                    - only implemented for consistency
                - `L0`
                    - `np.ndarray`, optional
                    - initialization for the low-rank part (`L`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)`
                - `S0`
                    - `np.ndarray`, optional
                    - initialization for the sparse part (`S`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)`
                - `Y0`
                    - `np.ndarray`, optional
                    - initialization for the dual variable (`Y`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)`
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

        #default params
        if verbose is None:     verbose = self.verbose

        #init
        if L0 is None:  L = np.zeros_like(X)    #low-rank part of X
        else:           L = L0                  
        if S0 is None:  S = np.zeros_like(X)    #sparse part of X
        else:           S = S0
        if Y0 is None:  Y = np.zeros_like(X)    #dual variable
        else:           Y = Y0

        for k in range(self.max_iter):
            #update primal variables
            L_new = svd_shrinkage((X - S + Y/self.mu), 1/self.mu)                   #low-rank part
            S_new = soft_thresholding((X - L_new + Y/self.mu), self.lbda/self.mu)   #sparse part

            #update dual variable
            Y_new = Y + self.mu * (X - L_new - S_new)   #dual variables
            
            #update parameters
            L = L_new.copy()
            S = S_new.copy()
            Y = Y_new.copy()
            
            #track energy
            self.energy.append(energy(L, S, self.lbda))
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
            if (delta < self.eps) and (k > 0):
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
            
        #update result
        self.L = L  #low-rank part of X
        self.S = S  #sparse part of X

        return
    
    def transform(self,
        X:np.ndarray=None, y:np.ndarray=None,
        verbose:int=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to transform the input
            - decomposes input into high-rank-matrix + low-rank-matrix (`L` + `S`)

            Parameters
            ----------
                - `X`
                    - `np.ndarray`, optional
                    - input matrix
                    - not needed
                    - only implemented for consistency
                    - the default is `None`
                - `y`
                    - `np.ndarray`, optional
                    - labels
                    - not needed
                    - only implemented for consistency
                    - the default is `None`
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
                - `L`
                    - `np.ndarray`
                    - low-rank part of `X`
                - `S`
                    - `np.ndarray`
                    - sparse part of `X`

            Comments
            --------
        """
        #default parameters
        if verbose is None: verbose = self.verbose

        #check feasibility
        if self.L is None or self.S is None:
            almofo.printf(
                msg=f'You have to call `self.fit()` before calling `self.transform()`!',
                context=f'{self.__class__.__name__}{self.transform.__name__}',
                type='WARNING',
                level=0,
                verbose=verbose
            )            
        L = self.L
        S = self.S

        return L, S
    
    def fit_transform(self,
        X:np.ndarray, y:np.ndarray=None,
        fit_kwargs:dict=None,
        transform_kwargs:dict=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer and transform the input in one go

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input matrix
                    - target of optimization constraint
                        - = C in [Boyd et al., 2011, Eq. 3.1](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
                - `y`
                    - `np.ndarray`, optional
                    - targets
                    - not needed
                    - only implemented for consistency
                - `fit_kwargs`
                    - `dict`, optional
                    -  kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`
                - `transform_kwargs`
                    - `dict`, optional
                    -  kwargs to pass to `self.transform()`
                    - the default is `None`
                        - will be set to `dict()`
            
            Raises
            ------

            Returns
            -------
                - `L`
                    - `np.ndarray`
                    - low-rank part of `X`
                - `S`
                    - `np.ndarray`
                    - sparse part of `X`

            Comments
            --------
        """
        
        if fit_kwargs is None:          fit_kwargs      = dict()
        if transform_kwargs is None:    transform_kwargs= dict()

        self.fit(X, y, **fit_kwargs)
        L, S = self.transform(X, y, **transform_kwargs)

        return L, S

class RPCA_CP:
    """
        - class implementing the solution of a Robust PCA via the Chambolle-Pock Primal-Dual Algorithm

        
        Attributes
        ----------
            - `lbda`
                - `float`
                - penalty parameter for sparse errors (`S`)
            - `tau`
                - `float`, optional
                - stepsize for updating the primal variable
                - used to set dual step-size `sigma`
                    - `sigma` is similar to `mu` in ADMM (`RPCA_ADMM`)
                - the default is `1.0`
            - `max_iter`
                - `int`, optional
                - maximum number of iterations the algorithm shall optimize for
                - the default is `100`
            - `eps`
                - `float`, optional
                - tolerance parameter to check for convergence
                - algorithm considered converged if `energy` changes less than `eps` for two consecutive iterations
                - the default is `0`
                    - will iterate until `max_iter` is reached
            - `theta`
                - `float`, optional
                - stepsize for the extrapolation/overrelaxation step of the primal variables
                - has to be greater than `0`
                - the default is `1.0`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
        
        Infered Attributes
        ------------------
            - `L`
                - `np.ndarray`
                - low-rank part of the input
            - `S`
                - `np.ndarray`
                - sparse part of the input
            - `energy`
                - `list`
                - energy (i.e. objective) for each iteration

        Methods
        -------
            - `fit()`
            - `transform()`
            - `fit_transform()`

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------

    """
    def __init__(self,
        lbda:float,
        tau:float=1.0,
        max_iter:int=100, eps:float=0.0,
        theta:float=1.0,
        verbose:int=0,
        ) -> None:
        
        #check ranges
        assert lbda     > 0, "`lbda` has to be greater than 0!"
        assert tau      >= 1,"`tau` has to be greater than 0!"      #primal stepsize (`sigma=1/tau` which is similar to `mu` in ADMM)
        assert max_iter > 0, "`max_iter` has to be greater than 0!"
        assert eps      >= 0,"`eps` has to be greater than or equal to 0!"
        assert theta    > 0, "`theta has to be greater than 0!"     #stepsize for extrapolation step

        #set attributes
        self.lbda       = lbda
        self.tau        = tau
        self.max_iter   = max_iter
        self.eps        = eps
        self.theta      = theta
        self.verbose    = verbose

        #get dual stepsize
        self.sigma      = 1/tau
        assert self.tau * self.sigma <= 1, f'`tau` and `sigma` have to fulfill `tau*sigma <= 1.0` but are `{self.tau*self.sigma}`!'
        
        #output attributes
        self.L = None       #low-rank
        self.S = None       #sparse

        self.energy = []

        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    lbda={repr(self.lbda)},\n'
            f'    tau={repr(self.tau)},\n'
            f'    max_iter={repr(self.max_iter)}, eps={repr(self.eps)},\n'
            f'    theta={repr(self.theta)},\n'
            f'    verbose={repr(self.verbose)},\n'           
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
        
    def fit(self,
        X:np.ndarray,
        y:np.ndarray=None,
        L0:np.ndarray=None, Y0:np.ndarray=None,
        verbose:int=None,
        ) -> None:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input matrix
                    - target of optimization constraint
                - `y`
                    - `np.ndarray`, optional
                    - targets
                    - not needed
                    - only implemented for consistency
                - `L0`
                    - `np.ndarray`, optional
                    - initialization for the low-rank part (`L`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)`
                - `Y0`
                    - `np.ndarray`, optional
                    - initialization for the dual variable (`Y`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)`
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

        #init
        if L0 is None:  L = np.zeros_like(X)    #low-rank part of X
        else:           L = L0
        if Y0 is None:  Y = np.zeros_like(X)    #dual variable
        else:           Y = Y0
        L_bar             = L.copy()   #extrapolation of L

        for k in range(self.max_iter):
            #new primal solution
            L_new = soft_thresholding(xbar=L - self.tau*Y, tau=self.lbda*self.tau, z=X)

            #extrapolation step
            L_bar = L_new + self.theta*(L_new - L)
            
            #update dual solution
            Y_new = sv_clip(Y + self.sigma*L_bar, a_min=0, a_max=1)

            #update parameters
            L = L_new.copy()
            Y = Y_new.copy()

            #track energy
            self.energy.append(energy(L, X-L, self.lbda))
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
            if (delta < self.eps) and (k > 0):
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
        self.L = L
        self.S = X - L

        return

    def transform(self,
        X:np.ndarray, y:np.ndarray,
        verbose:int=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to transform the input
            - decomposes input into high-rank-matrix + low-rank-matrix (`L` + `S`)

            Parameters
            ----------
                - `X`
                    - `np.ndarray`, optional
                    - input matrix
                    - not needed
                    - only implemented for consistency
                    - the default is `None`
                - `y`
                    - `np.ndarray`, optional
                    - labels
                    - not needed
                    - only implemented for consistency
                    - the default is `None`
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
                - `L`
                    - `np.ndarray`
                    - low-rank part of `X`
                - `S`
                    - `np.ndarray`
                    - sparse part of `X`

            Comments
            --------
        """
        #default parameters
        if verbose is None: verbose = self.verbose

        #check feasibility
        if self.L is None or self.S is None:
            almofo.printf(
                msg=f'You have to call `self.fit()` before calling `self.transform()`!',
                context=f'{self.__class__.__name__}{self.transform.__name__}',
                type='WARNING',
                level=0,
                verbose=verbose
            )

        L = self.L
        S = self.S

        return L, S

    def fit_transform(self,
        X:np.ndarray, y:np.ndarray,
        fit_kwargs:dict=None,
        transform_kwargs:dict=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer and transform the input in one go

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input matrix
                    - target of optimization constraint
                        - = C in [Boyd et al., 2011, Eq. 3.1](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
                - `y`
                    - `np.ndarray`, optional
                    - targets
                    - not needed
                    - only implemented for consistency
                - `fit_kwargs`
                    - `dict`, optional
                    -  kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`
                - `transform_kwargs`
                    - `dict`, optional
                    -  kwargs to pass to `self.transform()`
                    - the default is `None`
                        - will be set to `dict()`
            
            Raises
            ------

            Returns
            -------
                - `L`
                    - `np.ndarray`
                    - low-rank part of `X`
                - `S`
                    - `np.ndarray`
                    - sparse part of `X`

            Comments
            --------
        """

        if fit_kwargs is None:          fit_kwargs = dict()
        if transform_kwargs is None:    transform_kwargs= dict()

        self.fit(X, y, **fit_kwargs)
        L, S = self.transform(X=X, y=y, **transform_kwargs)

        return L, S

class RobustPCA:
    """
        - class implementing a Robust Principle Component Analysis (RPCA) transformer

        Attributes
        ----------
            - `lbda`
                - `float`
                - penalty parameter for sparse errors (`S`)
            - `tau`
                - `float`, optional
                - stepsize for updating the primal variable
                - only relevant for `method=="chambolle_pock"`
                - also used to set dual step-size `sigma`
                    - `sigma` is similar to `mu` in ADMM
                - the default is `1.0`                
            - `mu`
                - `float`, optional
                - penalty of the augmented Lagrangian
                - only relevant for `method=="admm"`
                - similar to `sigma = 1/tau` in Chambolle-Pock algorithm
                - has to be > 0
                - the default is `1.0`
            - `max_iter`
                - `int`, optional
                - maximum number of iterations the algorithm shall optimize for
                - the default is `100`
            - `eps`
                - `float`, optional
                - tolerance parameter to check for convergence
                - algorithm considered converged if `energy` changes less than `eps` for two consecutive iterations
                - the default is `0.0`
                    - will iterate until `max_iter` is reached
            - `theta`
                - `float`, optional
                - stepsize for the extrapolation step of the primal variables
                - has to be greater than 0
                - the default is `1.0`
            - `method`
                - `Literal['admm','chambolle_pock']`, optional
                - which algorithm to use for fitting the transformer
                - the default is `None`
                    - will be set to `chambolle_pock`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `movie2matrix()`
            - `matrix2movie()`
            - `fit()`
            - `transform()`
            - `fit_transform()`
            - `plot_result()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------
    """
    def __init__(self,
        lbda:float,
        tau:float=1.0, mu:float=1.0,
        max_iter:int=100, eps:float=0.0,
        theta:float=1.0,
        method:Literal['admm','chambolle_pock']=None,
        verbose:int=0,
        ) -> None:

        #check ranges
        assert lbda     > 0, "`lbda` has to be greater than 0!"
        assert tau      >= 1,"`tau` has to be greater than 0!"
        assert mu       > 0, "`mu` has to be greater than 0!"
        assert max_iter > 0, "`max_iter` has to be greater than 0!"
        assert eps      >= 0,"`eps` has to be greater than or equal to 0!"
        assert theta    > 0, "`theta has to be greater than 0!"

        #set attributes
        self.lbda                           = lbda
        self.tau                            = tau
        self.mu                             = mu
        self.max_iter                       = max_iter
        self.eps                            = eps
        self.theta                          = theta
        if method is None:  self.method     = 'chambolle_pock'
        else:               self.method     = method
        self.verbose                        = verbose

        #output attributes
        self.L = None       #low-rank
        self.S = None       #sparse
        
        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    lbda={repr(self.lbda)},\n'
            f'    tau={repr(self.tau)}, mu={repr(self.mu)},\n'
            f'    max_iter={repr(self.max_iter)}, eps={repr(self.eps)},\n'
            f'    theta={repr(self.theta)},\n'
            f'    method={repr(self.method)},\n'
            f'    verbose={repr(self.verbose)},\n'           
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def movie2matrix(self,
        X:np.ndarray,
        ) -> np.ndarray:
        """
            - method to transform a movie (series of frames) into a matrix
            - each row of the matrix contains one vectorized frame

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - series of images to be transformed
                    - has to have shape `(nframes,xpixels,ypixels)`
            
            Raises
            ------

            Returns
            -------
                - `X_mat`
                    - `np.ndarray`
                    - matrix version of `X`
                    - each row contains one vectorized frame

            Comments
            --------
        """

        X_mat = X.reshape(X.shape[0],-1)

        return X_mat

    def matrix2movie(self,
        X:np.ndarray,
        frame_size:Tuple[int],
        ) -> np.ndarray:
        """
            - method to transform a matrix of vectorized frames into a series of frames (movie)
            - inverse transform to `self.movie2matrix()`

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - matrix of vectorized frames
                    - each row contains one vectorized frame
                    - has to have shape `(nframes,frame_size[0]*frame_size[1])`
                - `frame_size`
                    - `Tuple[int,int]`
                    - size each individual frame in the output series shall have
            
            Raises
            ------

            Returns
            -------
                - `X_mov`
                    - `np.ndarray`
                    - transformed version of `X`
                    - has shape `(nframes,frame_size[0],frame_size[1])`
                        - `nframes = X.shape[0]`
                    - series of frames that will form a movie

            Comments
            --------
        """
        X_mov   = X.reshape(X.shape[0], *frame_size)
        return X_mov

    def fit(self,
        X:np.ndarray,
        y:np.ndarray=None,
        L0:np.ndarray=None, S0:np.ndarray=None, Y0:np.ndarray=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to fit the transformer

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input matrix
                        - if 2d array
                            - will be interpreted as actual input
                        - if 3d array
                            - will be interpreted as movie (series of frames)
                            - will transform series of frames into 2d array (calls `self.movie2matrix()`) before fitting
                            - will transform optimized result back to series of frames (calls `self.matrix2movie()`) after fitting
                    - target of optimization constraint
                        - = C in [Boyd et al., 2011, Eq. 3.1](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
                - `y`
                    - `np.ndarray`, optional
                    - targets
                    - not needed
                    - only implemented for consistency
                - `L0`
                    - `np.ndarray`, optional
                    - initialization for the low-rank part (`L`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)` after `X` has been reshaped to be digestible by the algorithm
                - `S0`
                    - `np.ndarray`, optional
                    - initialization for the sparse part (`S`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)` after `X` has been reshaped to be digestible by the algorithm
                - `Y0`
                    - `np.ndarray`, optional
                    - initialization for the dual variable (`Y`)
                    - the default it `None`
                        - will be set to `np.zeros_like(X)` after `X` has been reshaped to be digestible by the algorithm              
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
        
        #default params
        if verbose is None:     verbose = self.verbose

        #reshape input
        if X.ndim == 3:
            orig_size = X.shape
            X = self.movie2matrix(X)
        elif X.ndim == 2:
            orig_size = X.shape
            pass
        else:
            raise ValueError(f'`X` has to be a 2d or 3d array!')

        #execute RPCA
        if self.method == 'admm':
            OPT = RPCA_ADMM(
                lbda=self.lbda,
                mu=self.mu,
                eps=self.eps,
                max_iter=self.max_iter,
                verbose=verbose
            )
            L, S = OPT.fit_transform(X, y=y, fit_kwargs=dict(L0=L0, S0=S0, Y0=Y0))
            self.energy     = OPT.energy
        elif self.method == 'chambolle_pock':
            OPT = RPCA_CP(
                lbda=self.lbda,
                tau=self.tau,
                max_iter=self.max_iter,
                theta=self.theta,
                eps=self.eps,
                verbose=verbose,
            )
            L, S = OPT.fit_transform(X, y=y, fit_kwargs=dict(L0=L0, Y0=Y0))
            self.energy     = OPT.energy

        #reshape to match input shapes
        self.L = self.matrix2movie(L, orig_size[1:])
        self.S = self.matrix2movie(S, orig_size[1:])
        
        return
    
    def transform(self,
        X:np.ndarray=None, y:np.ndarray=None,
        verbose:int=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to transform the input
            - decomposes input into high-rank-matrix + low-rank-matrix (`L` + `S`)

            Parameters
            ----------
                - `X`
                    - `np.ndarray`, optional
                    - input matrix
                    - not needed
                    - only implemented for consistency
                    - the default is `None`
                - `y`
                    - `np.ndarray`, optional
                    - labels
                    - not needed
                    - only implemented for consistency
                    - the default is `None`
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
                - `L`
                    - `np.ndarray`
                    - low-rank part of `X`
                - `S`
                    - `np.ndarray`
                    - sparse part of `X`

            Comments
            --------
        """  
        
        #default parameters
        if verbose is None: verbose = self.verbose

        #check feasibility
        if self.L is None or self.S is None:
            almofo.printf(
                msg=f'You have to call `self.fit()` before calling `self.transform()`!',
                context=f'{self.__class__.__name__}{self.transform.__name__}',
                type='WARNING',
                level=0,
                verbose=verbose
            )

        L = self.L
        S = self.S

        return L, S
    
    def fit_transform(self,
        X:np.ndarray, y:np.ndarray,
        fit_kwargs:dict=None,
        transform_kwargs:dict=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to fit the transformer and transform the input in one go

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - input matrix
                        - if 2d array
                            - will be interpreted as actual input
                        - if 3d array
                            - will be interpreted as movie (series of frames)
                            - will transform series of frames into 2d array (calls `self.movie2matrix()`) before fitting
                            - will transform optimized result back to series of frames (calls `self.matrix2movie()`) after fitting
                    - target of optimization constraint
                        - = C in [Boyd et al., 2011, Eq. 3.1](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
                - `y`
                    - `np.ndarray`, optional
                    - targets
                    - not needed
                    - only implemented for consistency
                - `fit_kwargs`
                    - `dict`, optional
                    -  kwargs to pass to `self.fit()`
                    - the default is `None`
                        - will be set to `dict()`
                - `transform_kwargs`
                    - `dict`, optional
                    -  kwargs to pass to `self.transform()`
                    - the default is `None`
                        - will be set to `dict()`
            
            Raises
            ------

            Returns
            -------
                - `L`
                    - `np.ndarray`
                    - low-rank part of `X`
                - `S`
                    - `np.ndarray`
                    - sparse part of `X`

            Comments
            --------
        """

        #default parameters
        if fit_kwargs is None: fit_kwargs = dict()
        if transform_kwargs is None: transform_kwargs = dict()

        self.fit(X, y, **fit_kwargs)
        L, S = self.transform(X, y, **transform_kwargs)
        
        return L, S
    
    def plot_result(self,
        X:np.ndarray,
        L:np.ndarray, S:np.ndarray,
        fig:Figure=None,
        animate:bool=False,
        verbose:int=None,
        pcolormesh_kwargs:dict=None,
        plot_kwargs:dict=None,
        func_animation_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes,manimation.FuncAnimation]:
        """
            - method to visualize the result after transforming the data

            Parameters
            ----------
                - `X`
                    - `np.ndarray`
                    - original input (complete "image")
                    - has to be 3d array (series of frames)
                - `L`
                    - `np.ndarray`
                    - low-rank part of `X`
                    - has to be 3d array (series of frames)
                - `S`
                    - `np.ndarray`
                    - high-rank part of `X`
                    - has to be 3d array (series of frames)
                - `fig`
                    - `matplotlib.figure.Figure`, optional
                    - figure to plot into
                    - ideally pass an empty figure as 3 new axes will be created
                    - the default is `None`
                        - will create a new figure
                - `animate`
                    - `bool`, optional
                    - whether to generate an animation of the passed frames of images
                    - the default is `False`
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
                - `func_animation_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `matplotlib.animation.FuncAnimation()`
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
                - `anim`
                    - `matplotlib.animation.FuncAnimation`
                    - if `animate==True`
                        - generated animation
                    - else
                        - `None`

            Comments
            --------
        """

        def update(
            frame:int
            ) -> None:
            
            mesh_X.update(dict(array=X[frame]))
            mesh_L.update(dict(array=L[frame]))
            mesh_S.update(dict(array=S[frame]))
            
            return

        #default parameters
        if verbose is None: verbose = self.verbose
        if pcolormesh_kwargs is None:       pcolormesh_kwargs           = dict()
        if 'cmap' not in pcolormesh_kwargs: pcolormesh_kwargs['cmap']   = 'grey'
        if plot_kwargs is None:             plot_kwargs                 = dict()
        if func_animation_kwargs is None:   func_animation_kwargs       = dict()

        #correct shapes
        if X.ndim == 2: X = X.reshape(1,*X.shape)
        if L.ndim == 2: L = L.reshape(1,*L.shape)
        if S.ndim == 2: S = S.reshape(1,*S.shape)

        if fig is None:
            fig = plt.figure()
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(212)
        ax1.set_title('X')
        ax2.set_title('L')
        ax3.set_title('S')
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        
        mesh_X = ax1.pcolormesh(X[0], **pcolormesh_kwargs)
        mesh_L = ax2.pcolormesh(L[0], **pcolormesh_kwargs)
        mesh_S = ax3.pcolormesh(S[0], **pcolormesh_kwargs)
        ax4.plot(self.energy, **plot_kwargs)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Energy")
        if animate:
            anim = manimation.FuncAnimation(fig, update, **func_animation_kwargs)
        else:
            anim = None

        fig.tight_layout()
        
        axs = fig.axes

        return fig, axs, anim
    
