
    ###################
    #Steinwender Lukas#
    ###################



#______________________________________________________________________________
#Class containing useful stuff for data analysis
#TODO: add attributes?
#TODO: Add progress bar?
#TODO: implement PDM
class Data_LuSt:
    """
            Class to quickly print nice tables and save them to a text-file if need be.
        
        Methods
        -------
            - linspace_def
                --> creates an array of points with high resolution in some regions
            - lc_error
                --> estimates the error of a lightcurve given the respective time series and a
                    time-difference condition
            - pdm TODO: implement
            - fold
                --> folds an array (time series) onto a given phase
            - periodic shift
                --> shifts an array with regards to periodic boundaries
            - phase2time
                --> converts an array of phases to the repective times, given a period
            - phase_binning
                --> executes binning in phase on a dataseries
            - sigmaclipping
                --> executes sigma clipping in a dataseries
        
        Attributes
        ----------
            - 
                
        Dependencies
        ------------
            - numpy
            - matplotlib.pyplot
            - scipy.stats

        Comments
        --------
    """

    def __init__(self):
        pass

    def linspace_def(centers, widths=None, linspace_range=[0,1] ,
                    nintervals=50, nbins=100, spreads=None, maxiter=100000,
                    go_exact=True, testplot=False, verbose=False, timeit=False):
        """
            - function to generate a linspace in the range of linspace_range with higher 
              resolved areas around given centers
            - the final array has ideally nbins entries, but mostly it is more than that
                --> the user can get to the exact solution by seting go_exact = True
            - the resolution follows a gaussian for each given center
            - the size of these high-res-areas is defined by widths, wich is equal to
              the standard deviation in the gaussian
            - the spilling over the border of the chosen high-res areas can be varied by
              changing the parameter spreads

            Parameters
            ----------
                - centers
                    --> np.array/list
                    --> Defines the centers of the areas that will be higher resolved
                - widths
                    --> np.array/list, optional
                    --> Defines the widths (in terms of standard deviation for a gaussian)
                    --> Those widths define the interval which will be higher resolved
                    --> The default is None
                        ~~> will result in all ones
                - linspace_range
                    --> np.array/list, optional
                    --> Range on which the linspace will be defined
                    --> The default is [0,1]
                - nintervals
                    --> int, optional
                    --> Number of intervals to use for computing the distribution of nbins
                    --> Basically irrelevant in the endresult, but crutial for computation
                    --> i.e. for 50 intervals all nbins get distributed over 50 intervals
                    --> The default is 50
                - nbins
                    --> int, optional
                    --> total number of bins that are wished in the final array
                    --> due to rounding and the use of a distribution the exact number
                        is very unlikely to be reached.
                    --> because of this the number of bins is in general lower than requested.
                    --> just play until the best result is reached, or set go_exact ;)
                    --> the default is 100
                - spreads
                    --> np.array/list, optional
                    --> parameter to control the amount of overflowing of the distribution over
                        the given high-resolution parts
                    --> the default is None
                        ~~> will result in no additional overflow.
                - maxiter
                    --> int, optional
                    --> parameter to define the maximum number of iterations to take to get as
                        close to the desired nbins as possible
                    --> the default is 100000
                - go_exact
                    --> bool, optional
                    --> if True random points will be cut from the final result to achive 
                        exactly the amount of requested nbins
                    --> the default is True
                - testplot
                    --> bool, optional
                    --> if True will produce a test-plot that shows the underlying distribution
                        as well as the resulting array
                    --> the default is False.
                - verbose
                    --> bool, optional
                    --> if True will show messages defined by the creator of the function
                        ~~> e.g. Length of the output, number of iterations, ...
                    --> the default is False
                - timeit
                    --> bool, optional
                    --> specify wether to time the task and return the information or not.
                    --> the default is False
                
            Raises
            ------
                - TypeError
                    --> if the parametersprovided are of the wrong type

            Returns
            -------
                - combined_linspace
                    --> np.array
                    --> a linspace that has higher resolved areas where the user requests them
                        ~~> those areas are defined by centers and widths, respectively

            Dependencies
            ------------
                - numpy
                - matplotlib.pyplot
                - scipy.stats
 
            Comments
            --------
                - If you don't get the right length of your array right away, vary nbins
                  and nintervals until you get close enough, or set go_exact ;)
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.stats as sps
        from module_parts.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("linspace_def")
            task.start_task()


        ##########################################
        #initialize not provided data accordingly#
        ##########################################
        
        if widths ==  None:
            widths = np.ones_like(centers)
        if spreads == None:
            #initialize all spreads with "1"
            spreads = np.ones_like(centers)

        #################################
        #check if all shapes are correct#
        #################################
        
        if nintervals > nbins:
            raise ValueError("nintervals has to be SMALLER than nbins!")
        
        shape_check1_name = ["centers", "widths", "spreads"]
        shape_check1 = [centers, widths, spreads]
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
        if type(spreads) != np.ndarray and type(spreads) != list:
            raise TypeError("spreads has to be of type np.array or list!")
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
        for c, w, s in zip(centers, widths, spreads):
            gauss = sps.norm.pdf(intervals, loc=c, scale=s*w)
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
            plt.ylabel("number of points", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
            
        #time execution
        if timeit:
            task.end_task()

        
        return combined_linspace

    def lc_error(fluxes, times, delta_t_points, timeit=False):
        """    
            - estimates the error of a lightcurve given the respective time series and a
              time-difference condition (delta_t_points - maximum difference between 2 data points
              of same cluster)
            - the LC will be divided into clusters of datapoints, which are nearer to their
              respective neighbours, than the delta_t_points
            - returns an array, which assigns errors to the LC
                --> those errors are the standard deviation of the respective intervals
                --> all values of the LC in the same interval get the same errors assigned
            - returns mean values of the intervals and the belonging standard deviations
            
            Parameters
            ----------
                - fluxes
                    --> np.array
                    --> fluxes to extimate the error of
                - times
                    --> np.array
                    --> times corresponding to fluxes
                - delta_t_points
                    --> float
                    --> time-difference condition
                        ~~> delta_t_points - maximum difference between 2 data points
                            of same cluster
                - timeit 
                    --> bool, optional
                    --> specify wether to time the task and return the information or not
                    --> the default is False
                
            Raises
            ------

            Returns
            -------
                - LC_errors
                    --> estimated errors of flux
                - means
                    --> mean values of flux
                - stabws
                    --> standard deviations of flux
        
            Dependencies
            ------------
                - numpy

            Comments
            --------
        """
        
        import numpy as np
        from module_parts.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("lc_error")
            task.start_task()

        
        
        times_shifted = np.roll(times, 1)  #shifts array by one entry to the left, last element gets reintroduced at idx=0
        times_bool = times-times_shifted > delta_t_points 
        cond_fulfilled = np.where(times_bool)[0]    #returns an array of the indizes of the times, which fulfill condition in times_bool
        cond_fulfilled_shift = np.roll(cond_fulfilled,1)

        means  = np.array([]) #array to save mean values of intervals to
        stabws = np.array([]) #array to save stabws of intevals to
        LC_errors = np.empty_like(times)        #assign same errors to whole interval of LC
        
        for idxf, idxfs in zip(cond_fulfilled, cond_fulfilled_shift):
            
            if idxf == cond_fulfilled[0]:
                mean_iv  = np.mean(fluxes[0:idxf])
                stabw_iv = np.std(fluxes[0:idxf])
                LC_errors[0:idxf] = stabw_iv        #assign same errors to whole interval of LC         

            elif idxf == cond_fulfilled[-1]:
                mean_iv  = np.mean(fluxes[idxf:])
                stabw_iv = np.std(fluxes[idxf:])
                LC_errors[idxf:] = stabw_iv
            else:
                mean_iv  = np.mean(fluxes[idxfs:idxf])
                stabw_iv = np.std(fluxes[idxfs:idxf])
                LC_errors[idxfs:idxf] = stabw_iv    #assign same errors to whole interval of LC         
            
            means  = np.append(means,  mean_iv)
            stabws = np.append(stabws, stabw_iv) 
            
        #time execution
        if timeit:
            task.end_task()
        
        return LC_errors, means, stabws

    def pdm():
        #TODO: implement
        """

            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------

            Comments
            --------
        """ 
        raise NotImplementedError("Not implemented yet. But on the TODO-list ;)")   

    def fold(time, period, timeit=False):
        """
            - takes an array of times
                --> folds it with a specified period
                --> returns folded array of phases    

            Parameters
            ----------
                - time
                    --> np.array
                    --> times to be folded with the specified period
                - period 
                    --> int
                    --> period to fold the times onto
                - timeit
                    --> bool, optional
                    --> specify wether to time the task and return the information or not

            Raises
            ------

            Returns
            -------
            phases_folded : np.array
                phases corresponding to the given time folded with period.

            Dependencies
            ------------
                - numpy

            Comments
            --------
        """

        import numpy as np
        from module_parts.utility_astroLuSt import Time_stuff
        
        #time execution
        if timeit:
            task = Time_stuff("fold")
            task.start_task()

        delta_t = time-time.min()
        phases = delta_t/period
        
        #fold phases by getting the remainder of the division by the ones-value 
        #this equals getting the decimal numbers of that specific value
        #+1 because else a division by 0 would occur
        #floor always rounds down a value to the ones (returns everything before decimal point)
        phases_folded = (phases)-np.floor(phases) - 0.5

        #time execution
        if timeit:
            task.end_task()

        return phases_folded

    def periodic_shift(input_array, shift, borders, timeit=False, testplot=False, verbose=False):
        """
            - function to shift an array considering periodic boundaries

            Parameters
            ----------
                - input_array
                    --> np.array
                    --> array to be shifted along an interval with periodic boundaries
                - shift
                    --> float/int
                    --> magnizude of the shift to apply to the array
                - borders
                    --> list/np.array
                    --> upper and lower boundary of the periodic interval
                - timeit 
                    --> bool, optional
                    --> wether to time the execution
                    --> the default is False
                - testplot
                    --> bool, optional
                    --> wether to show a testplot
                    --> the default is False
                - verbose
                    --> bool, optional
                    --> wether to output information about the result
                    --> the default is False

            Raises
            ------
                - TypeError
                    --> if the provided parameters are of the wrong type

            Returns
            -------
            shifted : np.arra
                array shifted by shift along the periodic interval in borders.

            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
        """        
        
        import numpy as np
        import matplotlib.pyplot as plt
        from module_parts.utility_astroLuSt import Time_stuff
        
        
        #time execution
        if timeit:
            task = Time_stuff("periodic_shift")
            task.start_task()



        ################################
        #check if all types are correct#
        ################################
        
        if type(input_array) != np.ndarray:
            raise TypeError("input_array has to be of type np.ndarray!")
        if (type(borders) != np.ndarray) and (type(borders) != list):
            raise TypeError("borders has to be of type np.array or list!")
        if (type(timeit) != bool):
            raise TypeError("timeit has to be of type bool")
        if (type(testplot) != bool):
            raise TypeError("testplot has to be of type bool")
        if (type(verbose) != bool):
            raise TypeError("verbose has to be of type bool")


        #############
        #shift array#
        #############
        lower_bound = np.min(borders)
        upper_bound = np.max(borders)

        
        #apply shift
        shifted = input_array+shift
        
        #resproject into interval
        out_of_lower_bounds = (shifted < lower_bound)
        out_of_upper_bounds = (shifted > upper_bound)

        lower_deltas = lower_bound-shifted[out_of_lower_bounds]
        shifted[out_of_lower_bounds] = upper_bound - lower_deltas    
        upper_deltas = shifted[out_of_upper_bounds]-upper_bound
        shifted[out_of_upper_bounds] =lower_bound + upper_deltas
        

        if verbose:
            print("input_array: %a"%input_array)
            print("shifted_array: %a"%shifted)
            print("shift: %g"%shift)
            print("boundaries: %g, %g"%(lower_bound, upper_bound))
        
        if testplot:
            y_test = np.ones_like(shifted)
            
            fig = plt.figure()
            plt.suptitle("Testplot to visualize shift")
            # plt.plot(shifted[out_of_lower_bounds], y_test[out_of_lower_bounds]-1, color="b", marker=".", alpha=1, linestyle="", zorder=4)
            # plt.plot(shifted[out_of_upper_bounds], y_test[out_of_upper_bounds]+1, color="r", marker=".", alpha=1, linestyle="", zorder=4)
            plt.plot(shifted, y_test, color="r", marker="x", alpha=1, linestyle="", zorder=4, label="shifted array")
            plt.plot(input_array, y_test, color="k", marker=".", alpha=1, linestyle="", zorder=4, label="original array")
            plt.vlines([lower_bound, upper_bound], ymin=y_test.min()-1, ymax=y_test.max()+1, color="g", linestyle="--", label="boundaries")
            plt.xlabel("x", fontsize=16)
            plt.ylabel("y", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend()
            plt.show()

        #time execution
        if timeit:
            task.end_task()
        
        return shifted

    def phase2time(phases, period, timeit=False):
        """
            - converts a given array of phases into its respective time equivalent

            Parameters
            ----------
                - phases
                    --> np.array, float
                    --> the phases to convert to times
                - period
                    --> float
                    --> the given period the phase describes

            Raises
            ------

            Returns
            -------
                - time
                    --> np.array, float
                    --> the resulting time array, when the phases are converted
                - timeit
                    --> bool, optional
                    --> specify wether to time the task and return the information or not
                    --> the default is False    

            Dependencies
            ------------

            Comments
            --------
        """
   
        from module_parts.utility_astroLuSt import Time_stuff
        
        #time execution
        if timeit:
            task = Time_stuff("phase2time")
            task.start_task()

        time = phases*period
        
        #time execution
        if timeit:
            task.end_task()

        return time

    def phase_binning(fluxes, phases, nintervals, nbins, centers, widths, spreads=None, go_exact=True, verbose=False, testplot=False, timeit=False):
        """
            - function to execute binning in phase on some given timeseries
            - additional parameters allow the user to define different resolutions in 
              in different areas of the timeseries
            - returns the mean flux, phase and variance of each interval

            Parameters
            ----------
                - fluxes
                    --> np.array
                    --> fluxes to be binned
                - phases
                    --> np.array
                    --> phases to be binned
                - nintervals
                    --> int
                    --> number of intervals to distribute nbins on
                - nbins
                    --> int
                    --> number of bins you wish to have in the final array
                    --> is most of the time not fullfilled exactly
                        ~~> due to rounding
                    --> play with widths and spreads to adjust the number
                - centers
                    --> np.array/list
                    --> centers of areas one wants a higher resolution on
                    --> will become the centers of a gaussian for this area representing the
                        distribution of nbins
                - widths
                    --> np.array/list
                    --> widths of the areas one wants a higher resolution on
                    --> will become the standard deviation in the gaussian for this area
                        representing the distribution of nbins
                - spreads
                    --> np.array/list, optional
                    --> used to define the spill over the boundaries of centers +/- widths
                    --> the default is None
                        ~~> will result in no spill of the gaussian
                - go_exact
                    --> bool, optional
                    --> set to True if you want to use the exact number of requested bins
                    --> serves as input for linspace_def()
                - verbose
                    --> bool, optional
                    --> wether to show messages integrated in the function or not
                    --> the default is False
                - testplot
                    --> bool, optinoal
                    --> wether to show a testplot of the chosen distribution for the higher resolved area
                    --> the default is False
                - timeit
                    --> bool, optional
                    --> Specify wether to time the task and return the information or not
                    --> the default is False

            Raises
            ------

            Returns
            -------
                - phases_mean
                    --> np.array
                    --> mean phase of each interval
                        ~~> serve as representative for the repective intervals
                - fluxes_mean
                    --> np.array
                    --> the mean flux of each interval
                        ~~> serve as representative for the repective intervals
                - fluxes_sigm
                    --> np.array
                    --> the variance of the flux of each inteval
                - intervals
                    --> the intervals used for the calculation

            Dependencies
            ------------
                - numpy

            Comments
            --------
                - Plot_LuSt.linspace_def() will be called inside this function
        """
        
        import numpy as np
        from module_parts.utility_astroLuSt import Time_stuff
        
        #time execution
        if timeit:
            task = Time_stuff("phase_binning")
            task.start_task()

        intervals = Data_LuSt.linspace_def(centers,
                                           widths,
                                           linspace_range=[phases.min(),phases.max()],
                                           nintervals=nintervals,
                                           nbins=nbins+1,
                                           go_exact=go_exact,
                                           spreads=spreads,
                                           testplot=testplot,
                                           verbose=verbose
                                           )
        
        #saving arrays for mean LC    
        phases_mean = np.array([])    
        fluxes_mean = np.array([])
        fluxes_sigm = np.array([])
        
        #calculate mean flux for each interval
        for iv1, iv2 in zip(intervals[1:], intervals[:-1]):  
            bool_iv = ((phases <= iv1) & (phases > iv2))
            
            #calc mean phase, mean flux and standard deviation of flux for each interval        for pidx, p in enumerate(phases):
            mean_phase = np.mean(phases[bool_iv])
            mean_flux  = np.mean(fluxes[bool_iv])
            sigm_flux  = np.std(fluxes[bool_iv])

            phases_mean = np.append(phases_mean, mean_phase)
            fluxes_mean = np.append(fluxes_mean, mean_flux)
            fluxes_sigm = np.append(fluxes_sigm, sigm_flux)    
        
        if verbose:
            print("\n"+50*"-"+"\n",
                  "verbose, phase_binning: \n",
                  "--> requested shape            : %s\n"%nbins,
                  "--> shape of binned phases     : %s\n"%phases_mean.shape,
                  "--> shape of binned fluxes     : %s\n"%fluxes_mean.shape,
                  "--> shape of binned flux errors: %s\n"%fluxes_sigm.shape,
                  "--> shape of intervals used    : %s\n"%intervals.shape,
                  50*"-"+"\n"
                  )
            
        #time execution
        if timeit:
            task.end_task()

        
        return phases_mean, fluxes_mean, fluxes_sigm, intervals

    def sigma_clipping(fluxes, fluxes_mean, phases, phases_mean, intervals, clip_value_top, clip_value_bottom, times=[], timeit=False):
        """
            - cuts out all datapoints of fluxes (and phases and times) array which are outside of the interval
              [clip_value_bottom, clip_value_top] and returns the remaining array
            - used to get rid of outliers
            - clip_value_bottom and clip_value_top are usually defined as n*sigma, with 
              n = 1,2,3,... and sigma the standard deviation or Variance
            - if times is not specified, it will return an array of None with same size as fluxes    

            Parameters
            ----------
                - fluxes
                    --> np.array
                    --> fluxes to be sigma-clipped
                - fluxes_mean
                    --> np.array
                    --> mean values of fluxes to use as reference for clipping
                - phases
                    --> np.array
                    --> phases to be sigma-clipped
                - phases_mean
                    --> np.array 
                    --> phases corresponding to fluxes_mean
                - intervals
                    --> np.array
                    --> intervals to calculate the mean flux
                        ~~> this will be used as reference for cutting datapoints
                - clip_value_top
                    --> np.array
                    --> top border of clipping
                - clip_value_bottom
                    --> np.array
                    --> bottom border of clipping
                - times
                    --> np.array/list, optional
                    --> times corresponding to fluxes
                        ~~> only if existent
                    --> the default is []
                - timeit
                    --> bool, optional
                    --> specify wether to time the task and return the information or not.
                    --> the default is False
                

            Returns
            -------
                - fluxes_sigcut
                    --> fluxes after cutting out all values above clip_value_top and below clip_value bottom
                - phases_sigcut
                    --> phases after cutting out all values above clip_value_top and below clip_value bottom
                - times_sigcut
                    --> times after cutting out all values above clip_value_top and below clip_value bottom
                    --> None if not provided
                - cut_f
                    --> all cut-out values of fluxes
                - cut_p
                    --> all cut-out values of phases
                - cut_t
                    --> all cut-out values of times
        """

        import numpy as np
        from module_parts.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("sigma_clipping")
            task.start_task()

        times = np.array(times)

        #initiate saving arrays
        if len(times) == 0:
            times = np.array([None]*len(fluxes))
        elif len(times) != len(fluxes):
            raise ValueError("fluxes and times have to be of same legth!")
        elif len(fluxes) != len(phases):
            raise ValueError("fluxes and phases have to be of same length!")

        times_sigcut = np.array([])    
        fluxes_sigcut = np.array([])
        phases_sigcut = np.array([])
        cut_p = np.array([])
        cut_f = np.array([])
        cut_t = np.array([])

        intervalsp = np.roll(intervals,1)    

        for iv, ivp in zip(intervals[1:], intervalsp[1:]):
            
            bool_iv = ((phases <= iv) & (phases > ivp))
            bool_mean = ((phases_mean <= iv) & (phases_mean > ivp))
            upper_flux = fluxes_mean[bool_mean] + clip_value_top[bool_mean]
            lower_flux = fluxes_mean[bool_mean] - clip_value_bottom[bool_mean]
            
            fluxes_iv = fluxes[bool_iv]
            phases_iv = phases[bool_iv]
            times_iv  = times[bool_iv]
            
            bool_fluxcut = ((fluxes_iv < upper_flux) & (fluxes_iv > lower_flux))
            fluxes_cut = fluxes_iv[bool_fluxcut]
            phases_cut = phases_iv[bool_fluxcut]
            times_cut  = times_iv[bool_fluxcut]
            cut_fluxes = fluxes_iv[~(bool_fluxcut)] 
            cut_phases = phases_iv[~(bool_fluxcut)]  
            cut_times  = times_iv[~(bool_fluxcut)]
                    
            fluxes_sigcut = np.append(fluxes_sigcut, fluxes_cut)
            phases_sigcut = np.append(phases_sigcut, phases_cut)
            times_sigcut  = np.append(times_sigcut, times_cut)
            cut_f = np.append(cut_f, cut_fluxes)
            cut_p = np.append(cut_p, cut_phases)
            cut_t = np.append(cut_t, cut_times)
        
        #time execution
        if timeit:
            task.end_task()
                        
        return fluxes_sigcut, phases_sigcut, times_sigcut, cut_f, cut_p, cut_t

