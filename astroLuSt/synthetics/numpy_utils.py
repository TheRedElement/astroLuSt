


#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import re
from typing import Union, Tuple, Callable
import warnings



#%%definitions
class VariableLinspace:

    def __init__(self) -> None:
        
        return

    def linspace_def(self,
        centers, widths=None, linspace_range=[0,1] ,
        nintervals=50, nbins=100, maxiter=100000,
        go_exact=True, testplot=False, verbose=False, timeit=False
        ):
        """
            - method to generate a linspace in the range of linspace_range with higher resolved areas around given centers
            - the final array has ideally nbins entries, but mostly it is more than that
                - the user can get to the exact solution by seting go_exact = True
            - the resolution follows a gaussian for each given center
            - the size of these high-res-areas is defined by widths, wich is equal to
              the standard deviation in the gaussian

            Parameters
            ----------
                - centers
                    - np.array/list
                    - Defines the centers of the areas that will be higher resolved
                - widths
                    - np.array/list, optional
                    - Defines the widths (in terms of standard deviation for a gaussian)
                    - Those widths define the interval which will be higher resolved
                    - The default is None
                        - will result in all ones
                - linspace_range
                    - np.array/list, optional
                    - Range on which the linspace will be defined
                    - The default is [0,1]
                - nintervals
                    - int, optional
                    - Number of intervals to use for computing the distribution of nbins
                    - Basically irrelevant in the endresult, but crutial for computation
                    - i.e. for 50 intervals all nbins get distributed over 50 intervals
                    - The default is 50
                - nbins
                    - int, optional
                    - total number of bins that are wished in the final array
                    - due to rounding and the use of a distribution the exact number is very unlikely to be reached.
                    - because of this the number of bins is in general lower than requested.
                    - just play until the best result is reached, or set go_exact ;)
                    - the default is 100
                - maxiter
                    - int, optional
                    - parameter to define the maximum number of iterations to take to get as close to the desired nbins as possible
                    - the default is 100000
                - go_exact
                    - bool, optional
                    - if True random points will be cut from the final result to achive  exactly the amount of requested nbins
                    - the default is True
                - testplot
                    - bool, optional
                    - if True will produce a test-plot that shows the underlying distribution as well as the resulting array
                    - the default is False.
                - verbose
                    - bool, optional
                    - if True will show messages defined by the creator of the function
                        - e.g. Length of the output, number of iterations, ...
                    - the default is False
                - timeit
                    - bool, optional
                    - specify wether to time the task and return the information or not.
                    - the default is False
                
            Raises
            ------
                - TypeError
                    - if the parametersprovided are of the wrong type

            Returns
            -------
                - combined_linspace
                    - np.array
                    - a linspace that has higher resolved areas where the user requests them
                        - those areas are defined by centers and widths, respectively

            Dependencies
            ------------
                - numpy
                - matplotlib.pyplot
                - scipy.stats
 
            Comments
            --------
                - If you don't get the right length of your array right away, vary nbins and nintervals until you get close enough, or set go_exact ;)
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.stats as sps


        ##########################################
        #initialize not provided data accordingly#
        ##########################################
        
        if widths ==  None:
            widths = np.ones_like(centers)

        #################################
        #check if all shapes are correct#
        #################################
        
        if nintervals > nbins:
            raise ValueError("nintervals has to be SMALLER than nbins!")
        
        shape_check1_name = ["centers", "widths"]
        shape_check1 = [centers, widths]
        for sciidx, sci in enumerate(shape_check1):
            for scjidx, scj in enumerate(shape_check1):
                var1 = shape_check1_name[sciidx]
                var2 = shape_check1_name[scjidx]
                if len(sci) != len(scj):
                    raise ValueError("Shape of %s has to be shape of %s"%(var1, var2))

        ####################################
        #check if all datatypes are correct#
        ####################################
        
        if type(linspace_range) != np.ndarray and type(linspace_range) != list:
            raise TypeError("input_array has to be of type np.array or list!")
        if type(centers) != np.ndarray and type(centers) != list:
            raise TypeError("centers has to be of type np.array or list!")
        if type(widths) != np.ndarray and type(widths) != list:
            raise TypeError("widths has to be of type np.array or list!")        
        if type(go_exact) != bool:
            raise TypeError("go_exact has to be of type bool!")
        if type(testplot) != bool:
            raise TypeError("testplot has to be of type bool!")
        if type(verbose) != bool:
            raise TypeError("verbose has to be of type bool!")
        if type(timeit) != bool:
            raise TypeError("timeit has to be of type bool!")
        
        #initial definitions and conversions
        centers   = np.array(centers)
        widths    = np.array(widths)
        linspace_range = np.array(linspace_range)
        intervals = np.linspace(linspace_range.min(), linspace_range.max(), nintervals+1)
        
        #generate datapoint distribution
        dist = 0
        for c, w in zip(centers, widths):
            gauss = sps.norm.pdf(intervals, loc=c, scale=w)
            gauss /= gauss.max()        #to give all gaussians the same weight
            dist += gauss               #superposition all gaussians to get distribution


        #test various bins to get as close to nbins as possible
        iteration = 0
        testbin = nbins/2
        delta = 1E18
        while delta > 1E-6 and iteration < maxiter:
                    
            testdist = dist/np.sum(dist)            #normalize so all values add up to 1
            testdist *= testbin
            # dist *= nbins                   #rescale to add up to the number of bins
        
            
            #create combined linspace
            combined_linspace = np.array([])
            for i1, i2, bins in zip(intervals[:-1], intervals[1:], testdist):
                ls = np.linspace(i1,i2,int(bins), endpoint=False)                       #don't include endpoint to ensure non-overlapping
                combined_linspace = np.append(combined_linspace, ls)
            
            #add border points
            combined_linspace = np.insert(combined_linspace,0,linspace_range.min())
            combined_linspace = np.append(combined_linspace,linspace_range.max())
        
            delta = (nbins-combined_linspace.shape[0])
            testbin += 1
            iteration += 1
            
        #make sure to get sorted array
        combined_linspace = np.sort(combined_linspace)
        
        #cut random points but not the borders to get exactly to requested nbins
        if go_exact:
            n_to_cut = combined_linspace.shape[0] - nbins
            remove = np.random.randint(1, combined_linspace.shape[0]-1, size = n_to_cut)
            combined_linspace = np.delete(combined_linspace, remove)
        
        if verbose:
            print("\n"+50*"-"+"\n",
                  "verbose, linspace_def:\n",
                  "--> Number of iterations         : %s\n"%(iteration),
                  "--> Shape of combined_linspace   : %s\n"%(combined_linspace.shape),
                  "--> Desired shape                : %s\n"%(nbins),
                  "--> Range of linspace            : [%g, %g]"%(combined_linspace.min(), combined_linspace.max())
                  )
            if go_exact:
                print("--> Number of cut datapoints: %s"%(n_to_cut))
            print(50*"-"+"\n")
        
        if testplot:
            y_test = np.ones_like(combined_linspace)*testdist.max()
            
            fig = plt.figure()
            plt.suptitle("Testplot to visualize generated linspace", fontsize=18)
            plt.plot(combined_linspace, y_test, color="k", marker=".", alpha=0.5, linestyle="", zorder=4)
            plt.scatter(intervals, testdist, color="gainsboro", zorder=1, marker=".", figure=fig)
            plt.vlines(centers, testdist.min(), testdist.max(), colors="b")
            plt.vlines(centers+widths, testdist.min(), testdist.max(), colors="r")
            plt.vlines(centers-widths, testdist.min(), testdist.max(), colors="r")
            plt.xlabel("x", fontsize=16)
            plt.ylabel("Number of points", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
            
        
        return combined_linspace