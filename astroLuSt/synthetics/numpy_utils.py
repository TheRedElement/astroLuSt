


#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import scipy.stats as sps
from typing import Union, Tuple, Callable



#%%definitions
class VariableLinspace:
    """
        - class to generate a datapoints where number of datapoints is not equidistant but defined by mutivariate gaussian distributions
        - will generate 'num' datapoints (or as close as possible to that) from 'start' to 'stop' (inclusive)

        Attributes
        ----------
            - start
                - int
                - starting value of the datapoints to generate
            - stop
                - int
                - stopping value of the datapoints to generate
            - num
                - int, optional
                - number of datapoints to generate
                - might not be exact
                    - if you want to have the exact number of datapoints set 'go_exact' to True
                - the default is 100
            - nintervals
                - int, optional
                - number of intervals to divide the whole range [start, stop] into
                - for each of those intervals the amount of datapoints equal to the value of a multivariate gaussian will be generated
                - the generated datapoints for all intervals will then be combined to one dataseries
                - the default is 70
            - go_exact
                - bool, optional
                - whether to cut datapoints from the final generated dataseries until the num datapoints are reached
                    - datapoints will be cut at random, but the start and stop will still be part of the series
                - the default is False
            - maxiter
                - int, optional
                - maximum number of iterations to try and find a scaling for the underlying distribution that results in a linspace of shape as close to 'num' as possible
                - the default is None
                    - will test values from num/2 until 3/2*num
                    -i.e. maxiter = num
            - centers
                - np.ndarray, optional
                - array of centers of the gaussians to use for the underlying data distribution
                    - the gaussians will get superpsitioned to create the final multivariate gaussian data distribution
                - the default is None
                    - will set the center at (stop-start)/2
            - widths
                - np.ndarray, optional
                - array of the standard deviations corresponding to 'centers'
                - standard deviations of the gaussians to use for the underlying data distribution
                    - the gaussians will get superpsitioned to create the final multivariate gaussian data distribution
                - the default is None
                    - will be an array of ones with the same shape as 'centers'
            - verbose
                - int, optional
                - verbosity level
                - the default is 0
                
        Infered Attributes
        ------------------

        Methods
        -------

        Dependencies
        ------------
            - matplotlib
            - numpy
            - scipy
            - typing

        Comments
        --------

    """

    def __init__(self,
        start:float, stop:float, num:int=100, 
        nintervals:int=70,
        centers:np.ndarray=None, widths:np.ndarray=None,
        go_exact:bool=False, maxiter:int=None,
        verbose:int=0
        ) -> None:
        
        self.start      = start
        self.stop       = stop
        self.num        = num
        self.nintervals = nintervals
        self.go_exact   = go_exact
        if maxiter is None:
            self.maxiter = self.num
        else:
            self.maxiter    = maxiter
        if centers is None:
            self.centers = np.array([(stop-start)/2])
        else:
            self.centers = np.array(centers)
        if widths is None:
            self.widths = np.ones_like(self.centers)
        else:
            self.widths = np.array(widths)

        self.verbose    = verbose

        self.check_shapes

        return
    
    def __repr__(self) -> str:
        return (
            f'VariableLinspace(\n'
            f'    start={self.start}, stop={self.stop}, num={self.num},\n'
            f'    nintervals={self.nintervals},\n'
            f'    centers={self.centers}, widths={self.widths},\n'
            f'    go_exact={self.go_exact}, maxiter={self.maxiter},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )
    
    @property
    def check_shapes(self):
        if self.centers.shape != self.widths.shape:
            raise ValueError(f'"self.centers" and "self.widths" have to be np.ndarrays of the same shape but have shapes {self.centers.shapes} and {self.widths.shapes}, respectively!')
        if self.num < self.nintervals:
            raise ValueError(f'"self.num" has to be greater than "self.nintervals"!')
        return

    def get_datapoint_distribution(self,
        centers:np.ndarray, widths:np.ndarray,
        intervals:np.ndarray,
        ):

        #generate datapoint distribution
        dist = 0
        for c, w in zip(centers, widths):
            gauss = sps.norm.pdf(intervals, loc=c, scale=w)
            gauss /= gauss.max()        #to give all gaussians the same weight
            dist += gauss               #superposition all gaussians to get distribution

        return dist

    def make_combined_linspace(self,
        start:float, stop:float,
        intervals:np.ndarray,
        dist:np.ndarray,
        ):

        #create combined linspace
        combined_linspace = np.array([])
        for i1, i2, bins in zip(intervals[:-1], intervals[1:], dist):
            ls = np.linspace(i1,i2,int(bins), endpoint=False)                       #don't include endpoint to ensure non-overlapping
            combined_linspace = np.append(combined_linspace, ls)
        
        #add border points
        combined_linspace = np.insert(combined_linspace,0,start)
        combined_linspace = np.append(combined_linspace,stop)

        return combined_linspace      

    def make_exact(self,
        x, num,
        verbose
        ):

        #cut random points but not the borders to get exactly to requested nbins
        n_to_cut = x.shape[0] - num
        remove = np.random.randint(1, x.shape[0]-1, size=n_to_cut)
        x = np.delete(x, remove)
        
        if verbose > 1:
            print("    Number of cut datapoints: %s"%(n_to_cut))

        return x

    def generate(self,
        start:float=None, stop:float=None, num:int=None, 
        nintervals:int=None,
        centers:np.ndarray=None, widths:np.ndarray=None,
        go_exact:bool=None, maxiter:int=None,
        verbose:int=None
        ):
        
        if start      is None: start      = self.start
        if stop       is None: stop       = self.stop
        if num        is None: num        = self.num
        if nintervals is None: nintervals = self.nintervals
        if centers    is None: centers    = self.centers
        if widths     is None: widths     = self.widths
        if go_exact   is None: go_exact   = self.go_exact
        if maxiter    is None: maxiter    = self.maxiter
        if verbose    is None: verbose    = self.verbose

        linspace_range = np.array([start, stop])
        self.intervals = np.linspace(linspace_range.min(), linspace_range.max(), nintervals+1)

        dist = self.get_datapoint_distribution(centers=centers, widths=widths, intervals=self.intervals)


        #test various bins to get as close to nbins as possible
        iteration = 0
        testbin = num/2
        best_delta = 1E18
        best_combined_linspace = None
        while best_delta > 1E-6 and iteration < maxiter:
            #normalize
            self.testdist = dist/np.sum(dist)
            self.testdist *= testbin

            #create linspace for testdistribution        
            combined_linspace = self.make_combined_linspace(start=start, stop=stop, intervals=self.intervals, dist=self.testdist)

            delta = (num-combined_linspace.shape[0])
            if delta < best_delta:
                best_delta = delta
                best_combined_linspace = combined_linspace
            testbin += 1
            iteration += 1
            
        #make sure to get sorted array
        best_combined_linspace = np.sort(best_combined_linspace)

        if go_exact:
            best_combined_linspace = self.make_exact(best_combined_linspace, num, verbose)
        
        self.best_combined_linspace = best_combined_linspace

        if verbose > 0:
            print(
                f'INFO(VariableLinspace):\n'
                f'    Number of iterations:       {iteration}\n'
                f'    Shape of combined_linspace: {combined_linspace.shape}\n'
                f'    Desired shape:              {num}\n'
                f'    Range of linspace:          [{combined_linspace.min()},{combined_linspace.max()}]\n'
            )

        return best_combined_linspace
    
    def plot_result(self,
        ) -> Tuple[Figure, plt.Axes]:

        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.hist(self.best_combined_linspace, histtype='step', bins=self.nintervals, zorder=4, label='Generated Data')
        ax1.plot(self.intervals, self.testdist,                                      zorder=1, label='Underlying Distribution')
        for c, w in zip(self.centers, self.widths):
            ax1.axvline(c,   color="tab:purple")
            ax1.axvline(c+w, color="tab:red")
            ax1.axvline(c-w, color="tab:red")
        ax1.set_xlabel("x", fontsize=16)
        ax1.set_ylabel("Number Of Points")

        ax1.legend()
        plt.tight_layout()
        plt.show()

        axs = fig.axes

        return fig, axs
    

