
#TODO: exit criterion for niter
#TODO: implement use_polynomial
#TODO: option to allow history in plot_result

#%%imports
import matplotlib.pyplot as plt
import numpy as np


from astroLuSt.preprocessing.binning import Binning


#%%definitions
class SigmaClipping:
    """
        - class to execute sigma-clipping on x and y
        - creates a mask retaining only values that lie outside an interval of +/- sigma*std_y around a mean curve

        Attributes
        ----------'
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

        Comments
        --------
    """


    def __init__(self,                
        sigma_bottom:float=2, sigma_top:float=2,
        bound_history:bool=False, clipmask_history:bool=False,
        verbose:int=0,
        binning_kwargs:dict=None,
        ) -> None:

        self.sigma_bottom = sigma_bottom
        self.sigma_top = sigma_top

        if binning_kwargs is None:
            self.binning_kwargs = {'nintervals':0.1}
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
            f'    bound_history={self.bound_history}, clipmask_history={self.clipmask_history},\n'
            f'    verbose={self.verbose},\n'
            f'    binning_kwargs={self.binning_kwargs},\n'
            f')'
        )
    
    def get_mean_curve(self,
        x:np.ndarray, y:np.ndarray,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,
        verbose:int=None,
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
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        if verbose is None:
            verbose = self.verbose

        #calculate mean curve if insufficient information is provided
        if mean_x is None or mean_y is None or std_y is None:
            
            if verbose > 0:
                print(
                    f"INFO(SigmaClipping): Calculating mean-curve because one of 'mean_x', 'mean_y', std_y' is None!"
                )
            
            binning = Binning(
                verbose=verbose-1,
                **self.binning_kwargs
            )

            mean_x, mean_y, std_y = binning.fit_transform(x, y)
        else:
            assert (mean_x.shape == mean_y.shape) and (mean_y.shape == std_y.shape), f"shapes of 'mean_x', 'mean_y' and 'std_y' have to be equal but are {mean_x.shape}, {mean_y.shape}, {std_y.shape}"
        
        #adopt mean curves
        self.mean_x = mean_x.copy()
        self.mean_y = mean_y.copy()
        self.std_y  = std_y.copy()

        return

    def clip_curve(self,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,                    
        sigma_bottom:float=None, sigma_top:float=None,
        prev_clip_mask:np.ndarray=None,
        verbose:int=None,
        ) -> np.ndarray:
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
        self.get_mean_curve(x_cur, y_cur, mean_x, mean_y, std_y, verbose=verbose)


        #get mean curve including error
        self.y_mean_interp = np.interp(self.x, self.mean_x, self.mean_y)
        self.y_std_interp  = np.interp(self.x, self.mean_x, self.std_y)

        #mask of what to retain
        self.lower_bound = self.y_mean_interp-sigma_bottom*self.y_std_interp 
        self.upper_bound = self.y_mean_interp+sigma_top*self.y_std_interp
        
        #store mask
        self.clip_mask = (self.lower_bound<y_cur)&(y_cur<self.upper_bound)

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
        if show_cut: ax1.scatter(self.x[~clip_mask], self.y[~clip_mask],             color=cut_color,                                 alpha=0.7, zorder=1, label="Clipped")
        ax1.scatter(self.x[ clip_mask], self.y[ clip_mask],             color=ret_color,                                 alpha=1.0, zorder=2, label="Retained")
        ax1.errorbar(self.mean_x,       self.mean_y, yerr=self.std_y,   color=used_bins_color, linestyle="", marker=".",            zorder=3, label="Used Bins")
        ax1.plot(self.x[sort_array],    self.y_mean_interp[sort_array], color=mean_curve_color,                                     zorder=4, label="Mean Curve")
        ax1.plot(self.x[sort_array],    upper_bound[sort_array],        color=ulb_color,       linestyle="--",                      zorder=5, label=ulb_lab)
        ax1.plot(self.x[sort_array],    lower_bound[sort_array],        color=ulb_color,       linestyle="--",                      zorder=5) #,label=ulb_lab)

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax1.legend()
        plt.show()

        axs = fig.axes

        return fig, axs

    def fit(self,
        x:np.ndarray, y:np.ndarray,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,                        
        n_iter:int=1,
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
                - n_iter
                    - int, optional
                    - how often to apply SigmaClipping recursively
                    - the default is 1
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

        #assign input as attribute
        self.x = x.copy()
        self.y = y.copy()

        if verbose is None:
            verbose = self.verbose


        #initialize if not provided
        cur_clip_curve_kwargs = {}  #temporary dict to ensure same results after each call of self.fit()
        if 'prev_clip_mask' not in clip_curve_kwargs.keys():
            print('IN IF')
            cur_clip_curve_kwargs['prev_clip_mask'] = np.ones_like(self.x, dtype=bool)
        else:
            cur_clip_curve_kwargs = clip_curve_kwargs
            print('IN ELSE')


        for n in range(n_iter):
            if verbose > 0:
                print(f'INFO(SigmaClipping): Executing iteration #{n+1}/{n_iter}')

            self.clip_curve(mean_x, mean_y, std_y, **cur_clip_curve_kwargs)
            cur_clip_curve_kwargs['prev_clip_mask'] = cur_clip_curve_kwargs['prev_clip_mask']&self.clip_mask

            #store a history of the generated clip_masks if requested
            if self.clipmask_history:
                self.clip_masks.append(self.clip_mask)
            #store a history of the generated bounds if requested
            if self.bound_history:
                self.lower_bounds.append(self.lower_bound)
                self.upper_bounds.append(self.upper_bound)
    
        return 

    def transform(self,
        x:np.ndarray, y:np.ndarray,
        ):
        """
            - method to transform the input-dataseries
            - similar to scikit-learn scalers

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values of the dataseries to generate the mean curve for
                - y
                    - np.ndarray
                    - y-values of the dataseries to generate the mean curve for
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        x_clipped = x[self.clip_mask]
        y_clipped = y[self.clip_mask]

        return x_clipped, y_clipped
    
    def fit_transform(self,
        x:np.ndarray, y:np.ndarray,
        fit_kwargs:dict={},
        ):
        """
            - method to fit the transformer and transform the data in one go
            - similar to scikit-learn scalers
            
            Parameters
            ----------
                - fit_kwargs
                    - dict, optional
                    - kwargs to pass to self.fit()        
        """

        self.fit(
            x, y,
            **fit_kwargs,
        )
        x_clipped, y_clipped = self.transform(x, y)

        return  x_clipped, y_clipped


