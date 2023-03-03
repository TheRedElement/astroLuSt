
#%%imports
import matplotlib.pyplot as plt
import numpy as np

#%%definitions
class Binning:
    """
        - class to execute data-binning on a given input series
        - essentially calculates a mean representative curve
        - the scatter of the original data and thus certainty of the representation is captured by the standard deviation of 'y' in an interval


        Attributes
        ----------
            - nintervals
                - int, optional
                - nuber of intervals/bins to generate
                - the default is 100
            - npoints_per_interval
                - int, optional
                - generate intervals/bins automatically such that each bin contains 'npoints_per_interval' datapoints
                    - the last interval will contain all datapoints until the end of the dataseries
                - if set will overwrite nintervals
                - the default is None
                    - will use 'nintervals' to generate the bins
            - xmin
                - float, optional
                - the minimum value to consider for the interval/bin creation
                - the default is None
                    - will use the minimum of the input-series x-values
            - xmax
                - float, optional
                - the maximum value to consider for the interval/bin creation
                - the default is None
                    - will use the maximum of the input-series x-values
            - ddof
                - int, optional
                - Delta Degrees of Freedom used in np.nanstd()
                - the default is 0
            - verbose
                - int, optional
                - verbosity level
        
        Derived Attributes
        ------------------
            - generated_bins
                - np.ndarray
                - boundaries of the generated intervals/bin

        Methods
        -------
            - generate_bins()
            - bin_curve()
            - plot_result()

        Dependencies
        ------------
            - matplotlib
            - numpy

        Comments
        --------
    """

    def __init__(self,
        nintervals:int=100, npoints_per_interval:int=None,
        xmin:float=None, xmax:float=None,
        ddof:int=0,
        verbose:int=0,     
        ):
    
        self.nintervals = nintervals
        self.npoints_per_interval= npoints_per_interval
        self.xmin= xmin
        self.xmax= xmax
        self.ddof= ddof
        self.verbose= verbose

        pass

    def __repr__(self):

        return (
            f'Binning(\n'
            f'    nintervals={self.nintervals}, npoints_per_interval={self.npoints_per_interval},\n'
            f'    xmin={self.xmin}, xmax={self.xmax},\n'
            f'    ddof={self.ddof},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )

    def generate_bins(self,
        x:np.ndarray, y:np.ndarray,
        nintervals:int=None, npoints_per_interval:int=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to generate the requested bins

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values w.r.t. which the binning shall be executed
                - y
                    - np.ndarray
                    - y-values to be binned
                - nintervals
                    - int, optional
                    - nuber of intervals/bins to generate
                    - overwrites self.nintervals
                    - the default is None
                        - uses self.nintervals
                - npoints_per_interval
                    - int, optional
                    - generate intervals/bins automatically such that each bin contains 'npoints_per_interval' datapoints
                        - the last interval will contain all datapoints until the end of the dataseries
                    - overwrites self.npoints_per_interval
                    - if set will overwrite nintervals
                    - the default is None
                        - will use 'nintervals' to generate the bins
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose if set
                    - the default is None                
            
            Raises
            ------

            Returns
            -------
                - self.bins
                    - np.ndarray
                    - boundaries of the generated bins
            
            Comments
            --------
        """

        #set/overwrite internal attributes
        if npoints_per_interval is not None:
            nintervals = None
        elif npoints_per_interval is None and nintervals is None:
            nintervals = self.nintervals
            npoints_per_interval = self.npoints_per_interval
        elif npoints_per_interval is None and nintervals is not None:
            npoints_per_interval = None
        elif npoints_per_interval is not None and nintervals is None:
            nintervals = None

        if verbose is None: verbose = self.verbose

        #dynamically calculate bins if npoints_per_interval is specified 
        if npoints_per_interval is not None:
            sortidx = np.argsort(x)
            x_ = x[sortidx]
            y_ = y[sortidx]
            chunck_idxs = np.arange(0, x_.shape[0], npoints_per_interval)
            
            bins = x_[chunck_idxs]
            bins = np.append(bins, np.nanmax(x_)+1E-4)

        #interpret nintervals
        else:
            if 0 < nintervals and nintervals <= 1:
                #calculate nintervals as fraction of the shape of x and y 
                nintervals = int(self.nintervals*x.shape[0])
            elif nintervals > 1:
                nintervals = int(nintervals)
            else:
                raise ValueError("'nintervals' has to greater than 0!")


            if self.xmin is None: self.xmin = np.nanmin(x)
            if self.xmax is None: self.xmax = np.nanmax(x)

            bins = np.linspace(self.xmin, self.xmax, nintervals+1)
            bins[-1] += 1E-4

        #assign as attribute
        self.bins = bins


        if self.verbose > 0:
            print(f"INFO: Generated {len(self.bins)-1} bins")

        return self.bins
    
    def bin_curve(self,
        x:np.ndarray, y:np.ndarray,
        bins:np.ndarray=None,
        ddof:int=None,
        verbose:int=None,
        **generate_bins_kwargs:dict,
        ):
        """
            - method to execute the binning of y w.r.t. x

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values w.r.t. which the binning shall be executed
                - y
                    - np.ndarray
                    - y-values to be binned
                - bins
                    - np.ndarray
                    - array containing the boundaries of the intervals/bins to use for binning the curve
                    - will overwrite the autogeneration-process
                - ddof
                    - int, optional
                    - Delta Degrees of Freedom used in np.nanstd()
                    - overwrites self.ddof if set
                    - the default is None
                - verbose
                    - int, optional
                    - overwrites self.verbosity if set
                    - verbosity level
                - **generate_bins_kwargs
                    - kwargs of generate_bins()
                    
            Raises
            ------

            Returns
            -------
                - x_binned
                    - np.ndarray
                    - binned values for input 'x'
                    - has shape (1, nintervals)
                - y_binned
                    - np.ndarray
                    - binned values for input 'y'
                    - has shape (1, nintervals)
                - y_std
                    - np.ndarray
                    - standard deviation of 'y' for each interval
                    - characterizes the scattering of the input curve
                    - has shape (1, nintervals)            
        """

        #set/overwrite class attributes
        if ddof is None: ddof = self.ddof
        if verbose is None: verbose = self.verbose
        
        if bins is None:
            bins = self.generate_bins(x, y, verbose=verbose, **generate_bins_kwargs)
        else:
            self.bins = bins

        #init result arrays
        x_binned = np.array([])
        y_binned = np.array([])
        y_std = np.array([])
        self.n_per_bin = np.array([])    #number of samples per bin

        for b1, b2 in zip(bins[:-1], bins[1:]):

            iv_bool = (b1 <= x)&(x < b2)

            x_binned       = np.append(x_binned,       np.nanmean(x[iv_bool]))
            y_binned       = np.append(y_binned,       np.nanmean(y[iv_bool]))
            y_std          = np.append(y_std,          np.nanstd(y[iv_bool], ddof=ddof))
            self.n_per_bin = np.append(self.n_per_bin, np.count_nonzero(iv_bool))

        return x_binned, y_binned, y_std

    def plot_result(self,
        x, y,
        x_binned, y_binned, y_std,
        ):
        """
            - function to plot the result of the binning in phase

            Parameters
            ----------
                - x
                    - np.ndarray
                    - x-values w.r.t. which the binning shall be executed
                - y
                    - np.ndarray
                    - y-values to be binned
                - x_binned
                    - np.ndarray
                    - binned values for input 'x'
                    - has shape (1, nintervals)
                    - output of bin_curve()
                - y_binned
                    - np.ndarray
                    - binned values for input 'y'
                    - has shape (1, nintervals)
                    - output of bin_curve()
                - y_std
                    - np.ndarray
                    - standard deviation of 'y' for each interval
                    - characterizes the scattering of the input curve
                    - has shape (1, nintervals)                        
                    - output of bin_curve()

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
    
        verbose = self.verbose

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y, label="Input", zorder=1, color="tab:blue", alpha=0.7)
        ax1.errorbar(x_binned, y_binned, yerr=y_std, linestyle="", marker=".", label="Binned", zorder=2, color="tab:orange", alpha=1)

        if verbose > 2:
            ax1.vlines(self.bins, ymin=np.nanmin(y), ymax=np.nanmax(y), color='tab:grey', zorder=3, label='Bin Boundaries')

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        fig.tight_layout()
        plt.show()

        axs = fig.axes

        return fig, axs

