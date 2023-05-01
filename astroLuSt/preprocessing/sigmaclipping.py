

#%%imports
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Union, Callable
import warnings


from astroLuSt.preprocessing.binning import Binning


#%%definitions
class SigmaClipping:
    """
        - class to execute sigma-clipping on x and y
        - creates a mask retaining only values that lie outside an interval of +/- sigma*std_y around a mean curve

        Attributes
        ----------
            - sigma_bottom
                - float, optional
                - multiplier for the bottom boundary
                - i.e. bottom boundary = mean_y - sigma_bottom*std_y
                - the default is 2
                    - i.e. 2*sigma
            - sigma_top
                - float, optional
                - multiplier for the top boundary
                - i.e. top boundary = mean_y + sigma_top*std_y
                - the default is 2
                    - i.e. 2*sigma
            - use_polynomial
                - bool, optional
                - whether to use a polynomial fit instead of binning to estimate the mean curve
                - will use the std of the whole current curve and define upper and lower boundary by
                    - upper_bound := y_mean + sigma_top    * np.nanstd(y_cur)
                    - lower_bound := y_mean - sigma_bottom * np.nanstd(y_cur)
                    - y_cur is the clipped curve after the n-th iteration
                - the default is False
            - use_init_curve_sigma
                - bool, optional
                - whether to use the standard deviation of the initial curve accross all iterations to generate boundaries
                - if False will recalculate the standard deviation for each iteration anew and adjust the boundaries accordingly
                - the default is False 
            - bound_history
                - bool, optional
                - whether to store a history of all upper and lower bounds created during self.fit()
                - the default is False
            - clipmask_history
                - bool, optional
                - whether to store a history of all clip_mask created during self.fit()
                - the default is False
            - verbose
                - int, optional
                - verbosity level
            - binning_kwargs
                - dict, optional
                - kwargs for the Binning class
                - used to generate mean curves if none are provided
                - the default is None
                    - will initialize with {'nintervals':0.1, 'ddof':1}

        Infered Attributes
        ------------------
            - clip_mask
                - np.ndarray
                - final mask for the retained values
                - 1 for every value that got retained
                - 0 for every value that got cut
            - clip_masks
                - list
                    - contains np.ndarrays
                    - every clip_mask created while fitting transformer
                - only will be filled if clipmask_history == True
            - lower_bound
                - np.ndarray
                - traces out the lower bound to be considered for the sigma-clipping
            - lower_bounds
                - list
                    - contains np.ndarrays
                    - every lower_bound created while fitting transormer
                - only will be filled if bound_history == True
            - mean_x
                - np.ndarray
                - x-values of the mean curve created in the last iteration of self.fit()
            - mean_y
                - np.ndarray
                - y-values of the mean curve created in the last iteration of self.fit()
            - std_y
                - np.ndarray
                - standard deviation of the y-values of the mean curve created in the last iteration of self.fit()
            - upper_bound
                - np.ndarray
                - traces out the upper bound to be considered for the sigma-clipping
            - upper_bounds
                - list
                    - contains np.ndarrays
                    - every upper_bound created while fitting transormer
                - only will be filled if bound_history == True
            - x
                - np.ndarray
                - x values of the input data series
            - y
                - np.ndarray
                - y values of the input data series
            - y_mean_interp
                - np.ndarray
                - traces out the interpolated mean representative curve (resulting from binning)
            - y_std_interp
                - np.ndarray
                - traces out the interpolated standard deviation of the mean representative curve

        Methods
        -------
            - get_mean_curve()
            - clip_curve()
            - plot_result()
            - fit()
            - transform()
            - fit_transform()

        Dependencies
        ------------
            - matplotlib
            - numpy
            - re
            - typing

        Comments
        --------
    """


    def __init__(self,
        sigma_bottom:float=2, sigma_top:float=2,
        use_polynomial:bool=True,
        use_init_curve_sigma:bool=True,
        bound_history:bool=False, clipmask_history:bool=False,
        verbose:int=0,
        binning_kwargs:dict=None,
        ) -> None:

        self.sigma_bottom = sigma_bottom
        self.sigma_top = sigma_top

        self.use_polynomial = use_polynomial
        self.use_init_curve_sigma = use_init_curve_sigma

        if binning_kwargs is None:
            self.binning_kwargs = {'nintervals':0.1, 'ddof':1}
        else:
            self.binning_kwargs = binning_kwargs

        self.clipmask_history = clipmask_history
        self.bound_history = bound_history
        self.verbose = verbose

        #init infered attributes
        self.clip_mask     = np.array([])
        self.clip_masks    = []
        self.lower_bound   = np.array([])
        self.lower_bounds  = []
        self.mean_x        = np.array([])
        self.mean_y        = np.array([])
        self.std_y         = np.array([])
        self.upper_bound   = np.array([])
        self.upper_bounds  = []
        self.x             = np.array([])
        self.y             = np.array([])
        self.y_mean_interp = np.array([])
        self.y_std_interp  = np.array([])


        pass
    
    def __repr__(self) -> str:
        return (
            f'SigmaClipping(\n'
            f'    sigma_bottom={self.sigma_bottom}, sigma_top={self.sigma_top},\n'
            f'    use_polynomial={self.use_polynomial},\n'
            f'    use_init_curve_sigma={self.use_init_curve_sigma},\n'
            f'    bound_history={self.bound_history}, clipmask_history={self.clipmask_history},\n'
            f'    verbose={self.verbose},\n'
            f'    binning_kwargs={self.binning_kwargs},\n'
            f')'
        )
    
    def get_mean_curve(self,
        x:np.ndarray, y:np.ndarray,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,
        verbose:int=None,
        legfit_kwargs:dict={'deg':10},
        ) -> None:
        """
            - method to adopt the mean curves if provided and generate some if not

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to generate the mean curve for
                - y
                    - np.ndarray
                    - y-values of the dataseries to generate the mean curve for
                - mean_x
                    - np.ndarray, optional
                    - x-values of a representative mean curve
                    - does not have to have the same shape as x
                    - same shape as mean_y and std_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - mean_y
                    - np.ndarray, optional
                    - y-values of a representative mean curve
                    - does not have to have the same shape as y
                    - same shape as mean_x and std_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - std_y
                    - np.ndarray|Nonw, optional
                    - standard deviation/errorbars of the representative mean curve in y-direction
                    - does not have to have the same shape as y
                    - same shape as mean_x and mean_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None
                - legfit_kwargs
                    - dict, optional
                    - kwargs to pass to np.polynomial.legfit()
                    - the default is {'deg':10}
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        if verbose is None:
            verbose = self.verbose

        if 'ddof' in self.binning_kwargs.keys(): cur_ddof = self.binning_kwargs['ddof']
        else: cur_ddof = 1

        #calculate mean curve if insufficient information is provided
        if mean_x is None or mean_y is None or std_y is None:
            
            if verbose > 0:
                print(
                    f"INFO(SigmaClipping): Calculating mean-curve because one of 'mean_x', 'mean_y', std_y' is None!"
                )
            
            #fit legendre polynomial
            if self.use_polynomial:
                nanbool = (np.isfinite(x)&np.isfinite(y))   #get rid of np.nan for the fit
                coeffs = np.polynomial.legendre.legfit(x[nanbool], y[nanbool], **legfit_kwargs)
                mean_x = x.copy()
                mean_y = np.polynomial.legendre.legval(x, coeffs, tensor=False)
                std_y  = np.zeros_like(mean_y)+np.nanstd(y.copy(), ddof=cur_ddof)

            #use binning in phase to obtain the mean curve
            else:
                binning = Binning(
                    verbose=verbose-1,
                    **self.binning_kwargs
                )

                mean_x, mean_y, std_y = binning.fit_transform(x, y)
        else:
            assert (mean_x.shape == mean_y.shape) and (mean_y.shape == std_y.shape), f"shapes of 'mean_x', 'mean_y' and 'std_y' have to be equal but are {mean_x.shape}, {mean_y.shape}, {std_y.shape}"
        
        #adopt (binned) mean curves
        self.mean_x = mean_x.copy()
        self.mean_y = mean_y.copy()

        #update sigma only if reqested
        if self.use_init_curve_sigma:
            self.std_y = np.zeros_like(self.mean_y)+np.nanstd(self.y.copy(), ddof=cur_ddof)
        else:
            self.std_y  = std_y.copy()


        #get mean curve including error
        self.y_mean_interp = np.interp(self.x, self.mean_x[self.mean_x.argsort()], self.mean_y[self.mean_x.argsort()])
        self.y_std_interp  = np.interp(self.x, self.mean_x[self.mean_x.argsort()], self.std_y[self.mean_x.argsort()])

        return

    def clip_curve(self,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,                    
        sigma_bottom:float=None, sigma_top:float=None,
        prev_clip_mask:np.ndarray=None,
        verbose:int=None,
        legfit_kwargs:dict={'deg':10},
        ) -> None:
        """
            - method to actually execute sigma-clipping on x and y (once)
            - creates a mask retaining only values that lie outside an interval of +/- sigma*std_y around a mean curve

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to generate the mean curve for
                - y
                    - np.ndarray
                    - y-values of the dataseries to generate the mean curve for
                - mean_x
                    - np.ndarray, optional
                    - x-values of a representative mean curve
                    - does not have to have the same shape as x
                    - same shape as mean_y and std_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - mean_y
                    - np.ndarray, optional
                    - y-values of a representative mean curve
                    - does not have to have the same shape as y
                    - same shape as mean_x and std_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - std_y
                    - np.ndarray|Nonw, optional
                    - standard deviation/errorbars of the representative mean curve in y-direction
                    - does not have to have the same shape as y
                    - same shape as mean_x and mean_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - sigma_bottom
                    - float, optional
                    - multiplier for the bottom boundary
                    - i.e. bottom boundary = mean_y - sigma_bottom*std_y
                    - if set will completely overwrite the existing attribute self.sigma_bottom
                        - i.e. self.sigma_bottom will be set as sigma_bottom
                    - the default is 2
                        - i.e. 2*sigma
                - sigma_top
                    - float, optional
                    - multiplier for the top boundary
                    - i.e. top boundary = mean_y + sigma_top*std_y
                    - if set will completely overwrite the existing attribute self.sigma_top
                        - i.e. self.sigma_top will be set as sigma_top
                    - the default is 2
                        - i.e. 2*sigma
                - prev_clip_mask
                    - np.ndarray, optional
                    - boolean array
                    - contains any previously used clip_mask
                    - before the clipping will be applied, x and y will be masked by prev_clip_mask
                    - the default is None
                        - no masking applied
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None                        
                - legfit_kwargs
                    - dict, optional
                    - kwargs to pass to np.polynomial.legfit()
                    - the default is {'deg':10}

            Raises
            ------
        
            Returns
            -------           

            Comments
            --------
        """

        #overwrite original attributes if specified
        if sigma_bottom is None:
            sigma_bottom = self.sigma_bottom
        else:
            self.sigma_bottom = sigma_bottom
        if sigma_top is None:
            sigma_top = self.sigma_top
        else:
            self.sigma_top = sigma_top

        #initialize parameters
        if prev_clip_mask is None: prev_clip_mask = np.ones_like(self.x, dtype=bool)

        #create copy of input arrays to not overwrite them during execution
        x_cur = self.x.copy()
        y_cur = self.y.copy()

        #catching errors
        assert x_cur.shape == y_cur.shape, f"shapes of 'x' and 'y' have to be equal but are {x_cur.shape}, {y_cur.shape}"
        assert x_cur.shape == prev_clip_mask.shape, f"shapes of 'x' and 'prev_clip_mask' have to be equal but are {x_cur.shape}, {prev_clip_mask.shape}"

        #set elements in input array of previous mask to np.nan (but keep them in the array to retain the shape!!)
        x_cur[~prev_clip_mask] = np.nan
        y_cur[~prev_clip_mask] = np.nan


        #obtain mean (binned) curves
        self.get_mean_curve(x_cur, y_cur, mean_x, mean_y, std_y, verbose=verbose, legfit_kwargs=legfit_kwargs)

        #mask of what to retain
        self.lower_bound = self.y_mean_interp-sigma_bottom*self.y_std_interp 
        self.upper_bound = self.y_mean_interp+sigma_top*self.y_std_interp
        
        #store mask
        self.clip_mask = (self.lower_bound<y_cur)&(y_cur<self.upper_bound)

        #store a history of the generated clip_masks if requested
        if self.clipmask_history:
            self.clip_masks.append(self.clip_mask)
        #store a history of the generated bounds if requested
        if self.bound_history:
            self.lower_bounds.append(self.lower_bound)
            self.upper_bounds.append(self.upper_bound)

        return

    def plot_result(self,
        show_cut:bool=True,
        iteration:int=-1,
        ):
        """
            - method to create a plot visualizing the sigma-clipping result

            Parameters
            ----------
                - show_cut
                    - bool, optional
                    - whether to also display the cut datapoints in the summary
                    - the default is True
                - iteration
                    - int, optional
                    - which iteration of the SigmaClipping to display when plotting
                    - only usable if self.clipmask_history AND self.bound_history are True
                        - serves as index to the lists containing the histories
                    - the default is -1
                        - i.e. the last iterations

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure
                    - figure created if verbosity level specified accordingly
                - axs
                    - matplotlib axes
                    - axes corresponding to 'fig'

            Comments
            --------

        """
        ret_color = "tab:blue"
        cut_color = "tab:grey"
        used_bins_color = "tab:orange"
        mean_curve_color = "tab:green"
        ulb_color="k"

        ulb_lab = r"$\bar{y}~\{+%g,-%g\}\sigma$"%(self.sigma_top, self.sigma_bottom)
        
        #if clip_mask history has been stored consider iteration as index to plot this particular iteration
        if self.clipmask_history and self.bound_history:
            clip_mask   = self.clip_masks[iteration]
            lower_bound = self.lower_bounds[iteration]
            upper_bound = self.upper_bounds[iteration]
        else:
            clip_mask   = self.clip_mask
            lower_bound = self.lower_bound
            upper_bound = self.upper_bound

        #sorting-array (needed for plotting)
        sort_array = np.argsort(self.x)

        fig = plt.figure()
        fig.suptitle(f'Iteration: {iteration}')
        ax1 = fig.add_subplot(111)
        if show_cut: ax1.scatter(self.x[~clip_mask], self.y[~clip_mask],color=cut_color,                                                             alpha=0.7, zorder=1, label="Clipped")
        ax1.scatter(self.x[ clip_mask], self.y[ clip_mask],             color=ret_color,                                                             alpha=1.0, zorder=2, label="Retained")
        if not self.use_polynomial: ax1.errorbar(self.mean_x,       self.mean_y, yerr=self.std_y,   color=used_bins_color, linestyle="", marker=".",            zorder=3, label="Used Bins")
        ax1.plot(self.x[sort_array],    self.y_mean_interp[sort_array], color=mean_curve_color,                                                                 zorder=4, label="Mean Curve")
        ax1.plot(self.x[sort_array],    upper_bound[sort_array],        color=ulb_color,       linestyle="--",                                                  zorder=5, label=ulb_lab)
        ax1.plot(self.x[sort_array],    lower_bound[sort_array],        color=ulb_color,       linestyle="--",                                                  zorder=5) #,label=ulb_lab)

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax1.legend()
        plt.show()

        axs = fig.axes

        return fig, axs

    def fit(self,
        x:np.ndarray, y:np.ndarray,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,
        n_iter:int=1, stopping_crit:str=None,
        verbose:int=None,
        clip_curve_kwargs:dict={},
        ):
        """
            - method to apply SigmaClipping n_iter times consecutively
            - similar to scikit-learn scalers

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to clip
                - y
                    - np.ndarray
                    - y-values of the dataseries to clip
                - mean_x
                    - np.ndarray, optional
                    - x-values of a representative mean curve
                    - does not have to have the same shape as x
                    - same shape as mean_y and std_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - mean_y
                    - np.ndarray, optional
                    - y-values of a representative mean curve
                    - does not have to have the same shape as y
                    - same shape as mean_x and std_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - std_y
                    - np.ndarray|Nonw, optional
                    - standard deviation/errorbars of the representative mean curve in y-direction
                    - does not have to have the same shape as y
                    - same shape as mean_x and mean_y
                    - if 'None' will infer a mean curve via data-binning
                    - the default is 'None'
                - n_iter
                    - int, optional
                    - how often to apply SigmaClipping recursively
                    - the default is 1
                - stopping_crit
                    - str, optional
                    - stopping criterion to exit the SigmaClipping loop
                    - will be evaluated
                        - i.e. eval(stopping_crit) will be called
                        - if it evaluates to True, will break the loop
                    - some examples
                        - 'self.clip_mask.sum()<300'
                            - break if the clipped curve contains a maximum of 300 datapoints
                        - 'np.count_nonzero(~self.clip_mask) > 900'
                            - break if more than 900 datapoints get clipped
                        - 'self.clip_mask.sum()/self.x.shape < 0.5'
                            - break if less than 50% of the initial datapoints remain
                        - 'np.nanmean(self.y_std_interp) < 0.3'
                            - break if the mean standard deviation is less than 0.3
                        - 'self.clip_mask.sum()/cur_clip_curve_kwargs["prev_clip_mask"].sum() < 0.6'
                            - break if from one iteration to the next one 60% of the datapoints remain
                        - 'np.nanstd(self.y[self.clip_mask]) < 0.68'
                            - break if the standard deviation of the clipped curve is less than 0.68
                        - 'n==3'
                            - break after 3 iterations
                    - call dir(SigmaClipping) to figure out all the attributes defined
                        - likely any of the attributes (apart from methods) can be used in stopping_crit
                    - the default is None
                        - will continue until n_iter is reached
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None   
                - clip_curve_kwargs
                    - dict, optional
                    - kwargs to pass to self.clip_curve()
                    - the default is {}

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #check types
        if not isinstance(x, np.ndarray):
            x_type = re.sub(r'[<>]', '', str(type(x)))  #substitue angled brackets because TypeError() ignores them
            err_msg = str(f'"x" has to be of type np.ndarray and not {x_type}!')
            raise TypeError(err_msg)
        if not isinstance(y, np.ndarray):
            y_type = re.sub(r'[<>]', '', str(type(y)))  #substitue angled brackets because TypeError() ignores them
            err_msg = str(f'"y" has to be of type np.ndarray and not {y_type}!')
            raise TypeError(err_msg)

        if stopping_crit is None:
            stopping_crit = 'False'

        #assign input as attribute
        self.x = x.copy()
        self.y = y.copy()
        

        #initialize zeroth iteration
        it_zero = np.empty_like(self.x)
        it_zero[:] = np.nan
        self.clip_mask = np.ones_like(self.x, dtype=bool)
        self.clip_masks.append(self.clip_mask)
        self.lower_bounds.append(it_zero)
        self.upper_bounds.append(it_zero)

        if verbose is None:
            verbose = self.verbose


        #initialize if not provided
        cur_clip_curve_kwargs = {**clip_curve_kwargs}  #temporary dict to ensure same results after each call of self.fit()
        if 'prev_clip_mask' not in clip_curve_kwargs.keys():
            cur_clip_curve_kwargs['prev_clip_mask'] = np.ones_like(self.x, dtype=bool)
        else:
            clip_curve_kwargs = clip_curve_kwargs


        for n in range(n_iter):
            if verbose > 0:
                print(f'INFO(SigmaClipping): Executing iteration #{n+1}/{n_iter}')

            self.clip_curve(mean_x, mean_y, std_y, **cur_clip_curve_kwargs)

            #print the output of the stopping criterion
            if verbose > 2: print('INFO(SigmaClipping): stopping_crit evaluated to: %g'%(eval(re.findall(r'^[^<>=!]+', stopping_crit)[0])))
            if eval(stopping_crit):
                if verbose > 0:
                    print(f'INFO(SigmaClipping): stopping_crit fullfilled... Exiting after iteration #{n+1}/{n_iter}')
                
                #restore values of previous iteration
                
                self.clip_mask = self.clip_masks[-2]
                self.clip_masks = self.clip_masks[:-1]
                self.lower_bounds = self.lower_bounds[:-1]
                self.upper_bounds = self.upper_bounds[:-1]

                break

            #update previous clip_mask
            cur_clip_curve_kwargs['prev_clip_mask'] = cur_clip_curve_kwargs['prev_clip_mask']&self.clip_mask
            
        return 

    def transform(self,
        x:np.ndarray, y:np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
            - method to transform the input-dataseries
            - similar to scikit-learn scalers

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to clip
                - y
                    - np.ndarray
                    - y-values of the dataseries to clip
            Raises
            ------

            Returns
            -------
                - x_clipped
                    - np.ndarray
                    - the clipped version of the x-values of the input dataseries
                - y_clipped
                    - np.ndarray
                    - the clipped version of the y-values of the input dataseries

            Comments
            --------
        """

        x_clipped = x[self.clip_mask]
        y_clipped = y[self.clip_mask]

        return x_clipped, y_clipped
    
    def fit_transform(self,
        x:np.ndarray, y:np.ndarray,
        fit_kwargs:dict={},
        ) -> tuple[np.ndarray, np.ndarray]:
        """
            - method to fit the transformer and transform the data in one go
            - similar to scikit-learn scalers
            
            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to clip
                - y
                    - np.ndarray
                    - y-values of the dataseries to clip
                - fit_kwargs
                    - dict, optional
                    - kwargs to pass to self.fit()
            
            Returns
            -------
                - x_clipped
                    - np.ndarray
                    - the clipped version of the x-values of the input dataseries
                - y_clipped
                    - np.ndarray
                    - the clipped version of the y-values of the input dataseries                
        """

        self.fit(
            x, y,
            **fit_kwargs,
        )
        x_clipped, y_clipped = self.transform(x, y)

        return  x_clipped, y_clipped



class StringOfPearls:
    #TODO: loosness (i.e. allow n points of the window_size consecutive ones to not fullfill the condition)
    """
        - class to remove outliers which are at consecutive positions on the input dataseries and the 'metric' of which does not exceed a threshold 'th'
        - follows a similar convention as the scikit-learn library

        Attributes
        ----------
            - window_size, optional
                - int
                - minimum number of consecutive datapoints the 'metric' of which shall not exceed 'th'
                - if 'th' gets exceeded by 'window_size' consecutive points, the respective points get clipped 
                - the default is 3
            - th
                - float, callable, optional
                - threshold defining the lower boundary of the value the consecutive elements shall not exceed
                - if a callable is passed
                    - the callable has to take two arguments (x, and y)
                    - the callable has to return a float
                - the default is 0.01
            - metric
                - callable, optional
                - metric to use for comparison with th
                - the passed callable has to
                    - take two arguments (x, and y)
                    - return an array of the same size as x and y
                - the default is None
                    - will use the absolute gradient in y direction
                    - i.e. np.abs(np.gradient(y)) will be called
            - window
                - np.ndarray, optoinal
                - window to use for the convolution of the boolean
                - if passed, will overwrite window_size
                - the default is None
                    - will result in a window of [1]*window_size
                    - i.e. window_size consecutive entries have to be greater than th to be clipped
            - verbose   
                - int, optional
                - verbosity level
                - the default is 0

        Infered Attributes
        ------------------
            - clip_mask
                - np.ndarray
                - final mask for the retained values
                - 1 for every value that got retained
                - 0 for every value that got cut
            - th_float
                - float
                - floating point representation of the used threshold given 'x' and 'y' used during the call of fit()

        Methods
        -------
            - fit()
            - transform()
            - fit_transform()
            - plot_result()

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing
            - warnings
        
        Comments
        --------


    """
    def __init__(self,
        window_size:int=3,
        th:Union[float,Callable[[np.ndarray, np.ndarray],float]]=0.01,
        metric:Callable[[np.ndarray, np.ndarray],np.ndarray]=None,
        looseness:int=0,
        window:np.ndarray=None,
        verbose:int=0,
        ) -> None:
        

        self.window_size = window_size
        self.th = th
        if metric is None:
            self.metric = lambda x, y: np.abs(np.gradient(y))
        else:
            self.metric = metric
        self.loosenes = looseness
        #overwrite window_size if a custom window is passed
        if window is not None:
            self.window = window
            self.window_size = len(window)
        else:
            self.window = [1]*self.window_size
        self.verbose = verbose
        

        pass

    def __repr__(self) -> str:
        return (
            f'StringOfPearls(\n'
            f'    window_size={self.window_size},\n'
            f'    th={self.th},\n'
            f'    metric={self.metric},\n'
            f'    window={self.window},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )

    def fit(self,
        x:np.ndarray, y:np.ndarray,
        verbose:int=None,
        ):
        """
            - method to apply StringOfPearls
            - similar to scikit-learn scalers

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to clip
                    - preferably sorted
                - y
                    - np.ndarray
                    - y-values of the dataseries to clip
                    - preferably sorted w.r.t. 'x'
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None   

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if verbose is None:
            verbose = self.verbose

        if np.any(np.diff(x)<0) and verbose > 1:
            warnings.warn('The resulting mask might be wrong since "x" is not a sorted array!', category=UserWarning)


        #assign input values
        self.x = x
        self.y = y

        #check what threshold is used
        if isinstance(self.th, (float, int)):
            th = self.th
        elif callable(self.th):
            th = self.th(self.x, self.y)
        else:
            raise ValueError('self.th has to be any of the following: callable, float, int')
        self.th_float = th #floating point representation of self.th


        #convolution of window to get at least window_size consecutive occurences of metric(x,y) >= th 
        conv = np.convolve((self.metric(self.x, self.y) >= th).astype(int), self.window, mode="valid")
        print(conv.shape)
        
        #determine where the convolution resulted in self.window_size consecutive elements
        indexes_start = np.where(conv == self.window_size)[0]

        #generate indices of all
        idxs = np.array([np.arange(idx, idx+self.window_size) for idx in indexes_start])
        idxs = np.unique(idxs)

        #create boolean clip_mask
        self.clip_mask = np.ones_like(self.x, dtype=bool)
        if len(idxs) > 0:
            self.clip_mask[idxs] = False
        
        if verbose > 1:
            print(f'INFO(StringOfPearls): Number of clipped entries: {(~self.clip_mask).sum()}/{len(self.x)}')

        return

    def transform(self,
        x:np.ndarray, y:np.ndarray,        
        ) -> tuple[np.ndarray, np.ndarray]:
        """
            - method to transform the input-dataseries
            - similar to scikit-learn scalers

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to clip
                    - preferably sorted
                - y
                    - np.ndarray
                    - y-values of the dataseries to clip
                    - preferably sorted w.r.t. 'x'
            Raises
            ------

            Returns
            -------
                - x_clipped
                    - np.ndarray
                    - the clipped version of the x-values of the input dataseries
                - y_clipped
                    - np.ndarray
                    - the clipped version of the y-values of the input dataseries 

            Comments
            --------
        """

        x_clipped = x[self.clip_mask]
        y_clipped = y[self.clip_mask]

        return x_clipped, y_clipped
    
    def fit_transform(self,
        x:np.ndarray, y:np.ndarray,
        fit_kwargs:dict={}
        ) -> tuple[np.ndarray, np.ndarray]:
        """
            - method to fit the transformer and transform the data in one go
            - similar to scikit-learn scalers
            
            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to clip
                    - preferably sorted
                - y
                    - np.ndarray
                    - y-values of the dataseries to clip
                    - preferably sorted w.r.t. 'x'            
                - fit_kwargs
                    - dict, optional
                    - kwargs to pass to self.fit()

            Returns
            -------
                - x_clipped
                    - np.ndarray
                    - the clipped version of the x-values of the input dataseries
                - y_clipped
                    - np.ndarray
                    - the clipped version of the y-values of the input dataseries               
        """

        self.fit(x, y, **fit_kwargs)
        x_clipped, y_clipped = self.transform(x, y)

        return x_clipped, y_clipped
    
    def plot_result(self,
        show_cut:bool=True, 
        show_metric:bool=False,   
        ) -> tuple:
        """
            - method to create a plot visualizing the sigma-clipping result

            Parameters
            ----------
                - show_cut
                    - bool, optional
                    - whether to also display the cut datapoints in the summary
                    - the default is True
                - show_metric
                    - bool, optional
                    - whether to also display the metric used to compare against 'th' in the plot
                    - the default is False

            Raises
            ------

            Returns
            -------
                - fig
                    - matplotlib figure|None
                    - figure created if verbosity level specified accordingly
                - axs
                    - matplotlib axes|None
                    - axes corresponding to 'fig'

            Comments
            --------

        """
        
        ret_color = "tab:blue"
        cut_color = "tab:grey"
        met_color = 'tab:green'
        th_color  = 'tab:orange'


        patches = []
        #plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        #sort axis
        ax1.set_zorder(3)
        ax2.set_zorder(2)
        ax1.patch.set_visible(False)


        patches.append(    ax1.scatter(self.x[self.clip_mask],  self.y[self.clip_mask],      color=ret_color, alpha=1.0, zorder=3, label='Retained'))
        if show_cut:
            patches.append(ax1.scatter(self.x[~self.clip_mask], self.y[~self.clip_mask],     color=cut_color, alpha=0.7, zorder=2, label='Clipped'))
        if show_metric:
            patches.append(ax2.scatter(self.x,                  self.metric(self.x, self.y), color=met_color, alpha=0.7, zorder=1, label='Metric'))
            patches.append(ax2.axhline(self.th_float,           linestyle='--',              color=th_color,             zorder=2, label=f'Threshold: {self.th_float:.2f}'))
        
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y', color=ret_color)
        ax2.set_ylabel('metric', color=met_color)

        ax1.spines['right'].set_color(met_color)
        ax2.spines['right'].set_color(met_color)

        ax1.tick_params('y', colors=ret_color)
        ax2.tick_params('y', colors=met_color)

        ax1.legend(patches, [s.get_label() for s in patches])

        plt.tight_layout()
        plt.show()


        axs = fig.axes

        return fig, axs