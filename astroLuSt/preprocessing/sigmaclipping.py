
#TODO: implement n-iter (i.e. execute SigmaClipping n-iter times consecutively)

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
        ----------
            - x
                - np.ndarray
                - x-values of the dataseries to clip
                - same shape as y
            - y
                - np.ndarray
                - y-values of the dataseries to clip
                - same shape as x
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
                - the default is 2
                    - i.e. 2*sigma
            - sigma_top
                - float, optional
                - multiplier for the top boundary
                - i.e. top boundary = mean_y + sigma_top*std_y
                - the default is 2
                    - i.e. 2*sigma
            - verbose
                - int, optional
                - verbosity level
            - binning_kwargs
                - dict, optional
                - kwargs for the Binning class
                - used to generate mean curves if none are provided
                - the default is None

        Derived Attributes
        ------------------
            - clip_mask
                        - np.ndarray
                        - mask for the retained values
                        - 1 for every value that got retained
                        - 0 for every value that got cut     
            - lower_bound
                - np.ndarray
                - traces out the lower bound to be considered for the sigma-clipping
            - upper_bound
                - np.ndarray
                - traces out the upper bound to be considered for the sigma-clipping
            - sort_array
                - np.ndarray
                - indices to sort self.x in ascending order
                - only needed for plotting y_mean_interp, upper_bound, lower_bound
            - y_mean_interp
                - np.array
                - traces out the interpolated mean representative curve (resulting from binning)
            - y_std_interp
                - np.array
                - traces out the interpolated standard deviation of the mean representative curve

        Methods
        -------
            - get_mean_curve()
            - clip_curve
            - plot_result()

        Dependencies
        ------------
            - matplotlib
            - numpy

        Comments
        --------
    """


    def __init__(self,
        x:np.ndarray, y:np.ndarray,
        mean_x:np.ndarray=None, mean_y:np.ndarray=None, std_y:np.ndarray=None,                 
        sigma_bottom:float=2, sigma_top:float=2,
        verbose:int=0,
        binning_kwargs:dict=None,
        ) -> None:

        self.x = x
        self.y = y

        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_y  = std_y
        
        self.sigma_bottom = sigma_bottom
        self.sigma_top = sigma_top

        if binning_kwargs is None:
            self.binning_kwargs = {'nintervals':0.1}
        else:
            self.binning_kwargs = binning_kwargs


        self.verbose = verbose

        pass
    
    def __repr__(self) -> str:

        return (
        f'SigmaClipping(\n'
        f'    x={self.x}, y={self.y},\n'
        f'    mean_x:={self.mean_x}, mean_y={self.mean_y}, std_y={self.std_y},\n'
        f'    sigma_bottom={self.sigma_bottom}, sigma_top={self.sigma_top},\n'
        f'    verbose={self.verbose},\n'
        f'    binning_kwargs={self.binning_kwargs},\n'
        f')'
        )
    
    def get_mean_curve(self,
        verbose:int=None,
        ) -> None:
        """
            - method to adopt the mean curves if provided and generate some if not

            Parameters
            ----------
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
        if self.mean_x is None or self.mean_y is None or self.std_y is None:
            
            if verbose > 0:
                print(
                    f"INFO: Calculating mean-curve because one of 'mean_x', 'mean_y', std_y' is None!"
                )
            
            binning = Binning(
                verbose=verbose-1,
                **self.binning_kwargs
            )

            self.mean_x, self.mean_y, self.std_y = binning.bin_curve(self.x, self.y)
        else:
            assert (self.mean_x.shape == self.mean_y.shape) and (self.mean_y.shape == self.std_y.shape), f"shapes of 'mean_x', 'mean_y' and 'std_y' have to be equal but are {self.mean_x.shape}, {self.mean_y.shape}, {self.std_y.shape}"
        
        return 

    def clip_curve(self,
        sigma_bottom:float=None, sigma_top:float=None,
        verbose:int=None,
        ):
        """
            - method to actually execute sigma-clipping on x and y
            - creates a mask retaining only values that lie outside an interval of +/- sigma*std_y around a mean curve

            Parameters
            ----------
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
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None                        

            Raises
            ------
        
            Returns
            -------
                - clip_mask
                    - np.ndarray
                    - mask for the retained values
                    - 1 for every value that got retained
                    - 0 for every value that got cut            

            Dependencies
            ------------

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


        #catching errors
        assert self.x.shape == self.y.shape, f"shapes of 'x' and 'y' have to be equal but are {self.x.shape}, {self.y.shape}"

        self.get_mean_curve(verbose=verbose)

        #sorting-array
        self.sort_array = np.argsort(self.x)

        #get mean curve including error
        self.y_mean_interp = np.interp(self.x, self.mean_x, self.mean_y)
        self.y_std_interp  = np.interp(self.x, self.mean_x, self.std_y)

        #mask of what to retain
        self.lower_bound = self.y_mean_interp-sigma_bottom*self.y_std_interp 
        self.upper_bound = self.y_mean_interp+sigma_top*self.y_std_interp
        clip_mask = (self.lower_bound<self.y)&(self.y<self.upper_bound)

        self.clip_mask = clip_mask

        return clip_mask
    
    def plot_result(self):
        """
            - method to create a plot visualizing the sigma-clipping result

            Parameters
            ----------

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
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.x[~self.clip_mask], self.y[~self.clip_mask],             color=cut_color,                                 alpha=0.7, zorder=1, label="Clipped")
        ax1.scatter(self.x[self.clip_mask],  self.y[self.clip_mask],              color=ret_color,                                 alpha=1.0, zorder=2, label="Retained")
        ax1.errorbar(self.mean_x,            self.mean_y, yerr=self.std_y,        color=used_bins_color, linestyle="", marker=".",            zorder=3, label="Used Bins")
        ax1.plot(self.x[self.sort_array],    self.y_mean_interp[self.sort_array], color=mean_curve_color,                                     zorder=4, label="Mean Curve")
        ax1.plot(self.x[self.sort_array],    self.upper_bound[self.sort_array],   color=ulb_color,       linestyle="--",                      zorder=5, label=ulb_lab)
        ax1.plot(self.x[self.sort_array],    self.lower_bound[self.sort_array],   color=ulb_color,       linestyle="--",                      zorder=5) #,label=ulb_lab)

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax1.legend()
        plt.show()

        axs = fig.axes

        return fig, axs

