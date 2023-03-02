

    ###################
    #Steinwender Lukas#
    ###################
    
#______________________________________________________________________________
#Class containing useful stuff for data analysis
#TODO: add attributes?
#TODO: Add progress bar? - https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#TODO: fold(): Add option to fold into any desired interval


class Data_LuSt:
    """
        - Class to execute data processing
            - especially designed for astronomy
        
        Methods
        -------
            - linspace_def
                - creates an array of points with high resolution in some regions
            - lc_error
                - estimates the error of a lightcurve given the respective time series and a time-difference condition
            - pdm
                - runs a Phase Dispersion Minimization to estimate the period of a periodic time series
            - fold
                - folds an array (time series) onto a given phase
            - periodic shift
                - shifts an array with regards to periodic boundaries
            - phase2time
                - converts an array of phases to the repective times, given a period
            - phase_binning
                - executes binning in phase on a dataseries
            - sigmaclipping
                - executes sigma clipping in a dataseries
        
        Attributes
        ----------
                
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
                    nintervals=50, nbins=100, maxiter=100000,
                    go_exact=True, testplot=False, verbose=False, timeit=False):
        """
            - function to generate a linspace in the range of linspace_range with higher resolved areas around given centers
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
        from astroLuSt.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("linspace_def")
            task.start_task()


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
            
        #time execution
        if timeit:
            task.end_task()

        
        return combined_linspace

    def lc_error(fluxes, times, delta_t_points, timeit=False, verbose=False, testplot=False):
        """    
            - estimates the error of a lightcurve given the respective time series and a time-difference condition (delta_t_points - maximum difference between 2 data points of same cluster)
            - the LC will be divided into clusters of datapoints, which are nearer to their respective neighbours, than the delta_t_points
            - returns an array, which assigns errors to the LC
                - those errors are the standard deviation of the respective intervals
                - all values of the LC in the same interval get the same errors assigned
            - returns mean values of the intervals and the belonging standard deviations
            
            Parameters
            ----------
                - fluxes
                    - np.array
                    - fluxes to estimate the error of
                - times
                    - np.array
                    - times corresponding to fluxes
                - delta_t_points
                    - float
                    - time-difference condition
                        - maximum difference between 2 data points of same cluster
                - timeit 
                    - bool, optional
                    - specify wether to time the task and return the information or not
                    - the default is False
                - verbose
                    - bool, optional
                    - wether to show additional information implemented by the creator
                    - the default is False
                - testplot
                    - bool, optional
                    - whether to show a testplot
                    - the default is False
            Raises
            ------

            Returns
            -------
                - LC_errors
                    - np.array
                    - estimated errors of flux
                - means_fluxes
                    - np.array
                    - mean flux-values of the used intervals
                - mean_times
                    - np.array
                    - mean of the used time-intervals
                - sigmas
                    - np.array
                    - standard deviations of flux inside the used intervals
                - intervals
                    - np.array
                    - used intervals

            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        from astroLuSt.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("lc_error")
            task.start_task()

        #check where the datapoints are more delta_t_points apart from each other
        indices =  np.where(np.diff(times) > delta_t_points)[0]

        #used intervals
        mean_delta = np.mean(np.diff(times[indices]))
        intervals = times[indices] - (times[indices] - times[indices+1])/2
        intervals = np.insert(intervals, 0, times.min()-mean_delta/2)          #add initial point
        intervals = np.append(intervals, times.max()+mean_delta/2)             #add final point

        #initiate saveing arrays
        mean_fluxes  = np.array([]) #array to save mean values of flux of the intervals
        mean_times = np.array([])   #array to save mean values of the time-intervals to
        sigmas = np.array([])       #array to save the standard deviation to
        LC_errors = np.empty_like(times)    #array of errors that get assigned to the whole LC

        # calculate mean and standard deviation for each interval inbetween bigger gaps
        for iv, ivs in zip(intervals[:-1], intervals[1:]):

            iv_bool = (iv < times) & (times < ivs)
            times_iv = times[iv_bool]      #times of current interval
            fluxes_iv = fluxes[iv_bool]    #fluxes of current interval

            #calculate needed values
            mean_flux_iv = np.mean(fluxes_iv)
            sigma_iv = np.std(fluxes_iv)
            mean_time_iv = np.mean(times_iv)


            LC_errors[iv_bool] = sigma_iv                       #assign error to whole interval
            mean_fluxes = np.append(mean_fluxes, mean_flux_iv)  #store mean of the interval
            mean_times = np.append(mean_times, mean_time_iv)    #store mean time of the interval
            sigmas = np.append(sigmas, sigma_iv)
            
        if verbose:
            print("NO verbose IMPLEMENTED YET!")
            
        if testplot:
            fig = plt.figure(figsize=(10,10))
            fig.suptitle("Result of lc_error()", fontsize=18)
            ax = fig.add_subplot(111)
            ax.vlines(intervals, ymax=fluxes.max(), ymin=fluxes.min(), color="tab:grey", linestyle="--", label="intervals", zorder=1)
            ax.errorbar(times, fluxes, yerr=LC_errors, color="tab:blue", marker=".", linestyle="", capsize=3, label="estimated errors", zorder=2)
            ax.errorbar(mean_times, mean_fluxes, yerr=sigmas, color="tab:orange", marker=".", linestyle="", capsize=3, label="means of the interval", zorder=3)
            ax.set_xlabel(r"$x$", fontsize=16)
            ax.set_ylabel(r"y$(x)$", fontsize=16)
            ax.tick_params("both", labelsize=16)
            fig.legend(fontsize=16)
            plt.show()
            
        #time execution
        if timeit:
            task.end_task()
        
        return LC_errors, mean_fluxes, mean_times, sigmas, intervals


    def pdm(times, fluxes, period_start=1, period_stop=100, nperiods=100, nintervals=100, normalize=True, testplot=False, verbose=False, timeit=True):
        #TODO: add option to ignore nan-values and still get an output - to make choice of nperiods and nintervals not so hard (especially for short-period signals)
        """
            - function to execute a Phase Dispersion Minimization on a given timeseries.
            - not only limited to lightcurve-anaylsis but any time series with periodic behaiviour actually
                - e.g.: radial velocity curves

            Parameters
            ----------
                - times
                    - np.array
                    - times of the time series to run phase dispersion minimization on
                - fluxes
                    - np.array
                    - fluxes of the time series to run phase dispersion minimization on
                    - can also be any other function of time
                - period_start
                    - float, optional
                    - the period to consider as starting point for the analysis
                    - the default is 1
                - period_stop
                    - float, optional
                    - the period to consider as stopping point for the analysis
                    - the default is 100
                - nperiods
                    - int, optional
                    - how many trial periods to consider during the analysis
                    - the default is 100
                - ninteravls
                    - int, optional
                    - the number of itervals to consider for the evaluation of the total variance in the folded curve
                        - if nintervals is too big, only one datapoint might avalilable per interval
                            - it is therefore not possible to calculate the variance in that interval and hence the curve
                        - if nintervals is too small, all the datapoints might lie in the intervals
                            - therefore to many datapoints are considered to estimate the total variance hence leading to a variance which is too high
                        - just play around until you think you get a reasonable result (use testplot to check)
                    - the default is 100
                - normalize
                    - bool, optional
                    - wether to normalize the calculated variances
                    - the default is True
                - testplot
                    - bool, optional
                    - wether to show a testplot of the result or not
                    - the default is False
                - verbose
                    - bool, optional
                    - wether to show additional information implemented by the creator
                    - the default is False
                - timeit
                    - bool, optional
                    - wether to time the execution
                    - the default is True

            Raises
            ------

            Returns
            -------
                - best_period
                    - float
                    - the period yielding the lowest variance in the whole curve
                - best_sigma2
                    - float
                    - the lowest variance calculated
                - periods_sorted
                    - np.array
                    - the periods sorted after the the variance they yielded in the curve
                - sigma2s_sorted
                    - np.array
                    - the variances sorted from low to high
                    - corresponding to periods_sorted
                - best_fold
                    - np.array
                    - the resulting phases of the times folded with best_period
                - errestimate
                    - float
                    - an estiamte of the uncertainty of the result
                    - estimated to be 2* the maximum distance between two trial periods
                        - because the best period is certain to lie within the trial interval
                        - but where exactly is not sure

            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
            """ 

        import numpy as np
        import matplotlib.pyplot as plt
        from astroLuSt.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("pdm")
            task.start_task()

        #set up trial periods
        trial_periods = np.linspace(period_start, period_stop, nperiods+1)

        sigma2s_sums = np.array([])  #array to save the total variance
        all_intervals = []

        #calculate variance of folded curve for each trial period
        for p in trial_periods:
            folded = Data_LuSt.fold(times, p)                                #fold curve on period
            intervals = np.linspace(folded.min(), folded.max(), nintervals+1)   #setup intervals to use for variance calculation
            all_intervals.append(intervals)
            sigma2s = np.array([])                                              #array to save variance of each interval to
            #calculate variance of each interval
            for iv, ivs in zip(intervals[:-1], intervals[1:]):
                iv_bool = (iv <= folded) & (folded < ivs)
                sigma2_iv = np.nanvar(fluxes[iv_bool])
                sigma2s = np.append(sigma2s, sigma2_iv)

            #save all calculate values accordingly and determine period of minimum variance
            sigma2s_sums = np.append(sigma2s_sums, np.sum(sigma2s))
        
        #normalize sigma2s if specified
        if normalize:
            # sigma2s_sums /= np.linalg.norm(sigma2s_sums)    #actual normalization
            # sigma2s_sums /= np.sum(sigma2s_sums)            #normalize so everything adds up to 1
            sigma2s_sums /= sigma2s_sums.max()              #normalize so the maximum is 1

        #calculated desired parameters
        sortidx = np.argsort(sigma2s_sums)
        periods_sorted = trial_periods[sortidx]
        sigma2s_sorted = sigma2s_sums[sortidx]
        best_period = periods_sorted[0]
        best_sigma2 = sigma2s_sorted[0]
        best_fold = Data_LuSt.fold(times, best_period)
        best_intervals = all_intervals[sortidx[0]]

        errestimate = 2*np.diff(np.sort(periods_sorted)).max()  #error estimate as 2*maximum difference between periods

        if verbose:
            print("No verbose impemented yet.")

        if testplot:
            fig = plt.figure(figsize=(12,12))
            ax1 = fig.add_subplot(211)
            ax1.set_title("PDM-result", fontsize=18)
            ax1.plot(trial_periods, sigma2s_sums, color="tab:blue", marker=".", linestyle="", zorder=1)
            ax1.vlines(best_period, ymin=sigma2s_sums.min(), ymax=sigma2s_sums.max(), color="tab:orange", linestyle="-", label=r"$P_{\mathrm{PDM}} =$" + f"{best_period:.3f}", zorder=2)
            ax1.hlines(best_sigma2, xmin=trial_periods.min(), xmax=trial_periods.max(), color="tab:orange", linestyle="-", zorder=2)
            ax1.tick_params("both", labelsize=16)
            ax1.set_xlabel("Period", fontsize=16)
            ax1.set_ylabel("Variance", fontsize=16)
            ax1.legend(fontsize=16)
            ax2 = fig.add_subplot(212)
            ax2.set_title("Resulting lightcurve", fontsize=18)
            ax2.plot(best_fold, fluxes, color="tab:blue", marker=".", linestyle="", label="Final resulting lightcurve")
            ax2.vlines(best_intervals, ymax=fluxes.max(), ymin=fluxes.min(), color="tab:orange", linestyle="--", label="Used intervals")
            ax2.tick_params("both", labelsize=16)
            ax2.set_xlabel("Phase", fontsize=16)
            ax2.set_ylabel("Flux", fontsize=16)
            ax2.legend(fontsize=16)        
            plt.show()

        #time execution
        if timeit:
            task.end_task()

        return best_period, best_sigma2, periods_sorted, sigma2s_sorted, best_fold, errestimate


    def fold(time, period, tref=None, timeit=False):
        """
            - takes an array of times
                - folds it with a specified period
                - returns folded array of phases    

            Parameters
            ----------
                - time
                    - np.array
                    - times to be folded with the specified period
                - period 
                    - int
                    - period to fold the times onto
                - tref
                    - float, optional
                    - reference time to consider when folding the lightcurve
                    - the default is None
                        - will take min(time) as reference
                - timeit
                    - bool, optional
                    - specify wether to time the task and return the information or not

            Raises
            ------

            Returns
            -------
                - phases_folded
                    - np.array
                    - phases corresponding to the given time folded onto the period

            Dependencies
            ------------
                - numpy

            Comments
            --------
        """

        import numpy as np
        from astroLuSt.utility_astroLuSt import Time_stuff
        
        #time execution
        if timeit:
            task = Time_stuff("fold")
            task.start_task()

        if tref is None:
            tref = time.min()

        delta_t = time-tref
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

    def periodic_expansion(phases, fluxes, phase_ref_min=0, phase_ref_max=0, minmax="max", testplot=False, timeit=False):
        """
            - function to expand a periodic timeseries on either side
                - takes all datapoints up to a reference phase
                - appends them to the original array according to specification  

            Parameters
            ----------
                - phases
                    - np.array
                    - phases of the datapoints to be expanded
                - fluxes 
                    - np.array
                    - fluxes of the datapoints to be expanded
                - phase_ref_min
                    - float, optional
                    - reference phase
                        - will be used in order to determine which phases to consider for appending
                    - used in the case of appending to the minimum and both ends
                    - the default is 0
                - phase_ref_max
                    - float, optional
                    - reference phase
                        - will be used in order to determine which phases to consider for appending
                    - used in the case of appending to the minimum and both ends
                    - the default is 0
                - minmax
                    - str, optional
                    - wether to append to the maximum or minimum of the dataseries
                    - can take either
                        - 'min'
                            - will expand on the minimum side
                            - will consider all phases from phase_ref to the maximum phase
                        - 'max'
                            - will expand on the maximum side
                            - will consider all phases the minimum phase up to phase_ref
                        - 'both'
                            - will expand on both ends of the curve
                            - requires 'phase_ref' to be a list with two entries
                    - the default is 'max'
                -testplot
                    - bool, optional
                    - whether to show a testplot of the result
                    - the default is False
                - timeit
                    - bool, optional
                    - specify whether to time the task and return the information or not
                    - the default is False

            Raises
            ------
                - ValueError
                    - if 'minmax' gets passed a wrong argument
                - TypeError
                    - if 'phase_ref' gets passed a wrong type

            Returns
            -------
                - expanded_phases
                    - np.array
                    - phases including the expanded part
                - expanded_fluxes
                    - np.array
                    - fluxes including the expanded part

            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
        """
        

        import numpy as np
        import matplotlib.pyplot as plt
        from astroLuSt.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("periodic_expansion")
            task.start_task()

        #sort to get correct appendix in the end
        sortidx = np.argsort(phases)
        phases = phases[sortidx]
        fluxes = fluxes[sortidx]

        #append to maximum
        if minmax == "max":
            phase_bool = (phases < phase_ref_max)
            appendix_phases = phases.max() + (phases[phase_bool] - phases.min())
            phase_ref = phase_ref_max
        #append to minimum
        elif minmax == "min":
            phase_bool = (phases > phase_ref_min)
            appendix_phases = phases.min() - (phases.max() - phases[phase_bool])
            phase_ref = phase_ref_min
        elif minmax == "both":
            if phase_ref_min < phase_ref_max:
                raise ValueError("'phase_ref_min' has to be greater or equal than 'phase_ref_max' ")
            phase_ref = [phase_ref_min, phase_ref_max]
            phase_bool_max = (phases < phase_ref_max)
            phase_bool_min = (phases > phase_ref_min)
            phase_bool = phase_bool_max|phase_bool_min
            appendix_phases_max = phases.max() + (phases[phase_bool_max] - phases.min())
            appendix_phases_min = phases.min() - (phases.max() - phases[phase_bool_min])
            appendix_phases = np.append(appendix_phases_max, appendix_phases_min)
        else:
            raise ValueError("'minmax' has to bei either 'min', 'max' or 'both'!")
        
        appendix_fluxes = fluxes[phase_bool]
        expanded_phases = np.append(phases, appendix_phases)
        expanded_fluxes = np.append(fluxes, appendix_fluxes)

        if testplot:
            fig = plt.figure()
            plt.suptitle("Testplot to visualize shift")
            plt.plot(phases, fluxes, color="tab:grey", marker=".", linestyle="", alpha=0.5, zorder=2, label="original input")
            plt.plot(expanded_phases, expanded_fluxes, color="tab:blue", marker=".", linestyle="", alpha=1, zorder=1, label="expanded input")
            plt.vlines(phase_ref, ymin=fluxes.min(), ymax=fluxes.max(), color="g", linestyle="--", label="reference phase")
            plt.xlabel("x", fontsize=16)
            plt.ylabel("y", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend()
            plt.show()
        
        #time execution
        if timeit:
            task.end_task()

        return expanded_phases, expanded_fluxes


    def periodic_shift(input_array, shift, borders, timeit=False, testplot=False, verbose=False):
        """
            - function to shift an array considering periodic boundaries

            Parameters
            ----------
                - input_array
                    - np.array
                    - array to be shifted along an interval with periodic boundaries
                - shift
                    - float/int
                    - magnizude of the shift to apply to the array
                - borders
                    - list/np.array
                    - upper and lower boundary of the periodic interval
                - timeit 
                    - bool, optional
                    - wether to time the execution
                    - the default is False
                - testplot
                    - bool, optional
                    - wether to show a testplot
                    - the default is False
                - verbose
                    - bool, optional
                    - wether to output information about the result
                    - the default is False

            Raises
            ------
                - TypeError
                    - if the provided parameters are of the wrong type

            Returns
            -------
                - shifted
                    - np.array
                    - array shifted by shift along the periodic interval in borders

            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
        """        
        
        import numpy as np
        import matplotlib.pyplot as plt
        

        ################################
        #check if all types are correct#
        ################################
        
        if type(input_array) != np.ndarray:
            raise TypeError("input_array has to be of type np.ndarray! If you want to shift a scalar, simply convert it to an array and acess outputarray[0]")
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

        return shifted

    def phase2time(phases, period, timeit=False):
        """
            - converts a given array of phases into its respective time equivalent

            Parameters
            ----------
                - phases
                    - np.array, float
                    - the phases to convert to times
                - period
                    - float
                    - the given period the phase describes

            Raises
            ------

            Returns
            -------
                - time
                    - np.array, float
                    - the resulting time array, when the phases are converted 

            Dependencies
            ------------

            Comments
            --------
                - operates with phases in the interval [-0.5,0.5]
                - if you wish to convert phases from the interval [0,1], simply pass phases-0.5 to the function
        """
   
        time = (phases+0.5)*period
        
        return time

    def phase_binning(fluxes, phases, nintervals, nbins, centers, widths, go_exact=True, verbose=False, testplot=False, timeit=False):
        """
            - function to execute binning in phase on some given timeseries
            - additional parameters allow the user to define different resolutions in  in different areas of the timeseries
            - returns the mean flux, phase and variance of each interval

            Parameters
            ----------
                - fluxes
                    - np.array
                    - fluxes to be binned
                - phases
                    - np.array
                    - phases to be binned
                - nintervals
                    - int
                    - number of intervals to distribute nbins on
                - nbins
                    - int
                    - number of bins you wish to have in the final array
                    - is most of the time not fullfilled exactly
                        - due to rounding
                    - play with widths to adjust the number
                - centers
                    - np.array/list
                    - centers of areas one wants a higher resolution on
                    - will become the centers of a gaussian for this area representing the distribution of nbins
                - widths
                    - np.array/list
                    - widths of the areas one wants a higher resolution on
                    - will become the standard deviation in the gaussian for this area representing the distribution of nbins
                - go_exact
                    - bool, optional
                    - set to True if you want to use the exact number of requested bins
                    - serves as input for linspace_def()
                - verbose
                    - bool, optional
                    - wether to show messages integrated in the function or not
                    - the default is False
                - testplot
                    - bool, optinoal
                    - wether to show a testplot of the chosen distribution for the higher resolved area
                    - the default is False
                - timeit
                    - bool, optional
                    - Specify wether to time the task and return the information or not
                    - the default is False

            Raises
            ------

            Returns
            -------
                - phases_mean
                    - np.array
                    - mean phase of each interval
                        - serve as representative for the repective intervals
                - fluxes_mean
                    - np.array
                    - the mean flux of each interval
                        - serve as representative for the repective intervals
                - fluxes_sigm
                    - np.array
                    - the variance of the flux of each inteval
                - intervals
                    - the intervals used for the calculation

            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
                - Plot_LuSt.linspace_def() will be called inside this function
                - contains a subfunction (execute_binning())
                    - used to correct if nan occur in the result
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        from astroLuSt.utility_astroLuSt import Time_stuff

        #time execution
        if timeit:
            task = Time_stuff("phase_binning")
            task.start_task()

        def execute_binning(correctnan=0):
            """
                - Code snippet to execute the binning procedure
                - Needed to correct if the result contains nan values
                    - in that case it is necessary to run the procedure twice
                
                Parameters
                ----------
                    - correctnan
                        - int, optional
                        - parameter to correct for the nan values in the result
            """

            intervals = Data_LuSt.linspace_def(centers,
                                               widths,
                                               linspace_range=[phases.min(),phases.max()],
                                               nintervals=nintervals,
                                               nbins=nbins+1+correctnan,
                                               go_exact=go_exact,
                                               testplot=testplot,
                                               verbose=verbose
                                               )

            #saving arrays for mean LC    
            phases_mean = np.array([])    
            fluxes_mean = np.array([])
            fluxes_sigm = np.array([])

            #calculate mean flux for each interval
            for iv1, iv2 in zip(intervals[:-1], intervals[1:]):  
                bool_iv = ((iv1 <= phases) & (phases < iv2))

                #calc mean phase, mean flux and standard deviation of flux for each interval        for pidx, p in enumerate(phases):
                mean_phase = np.mean(phases[bool_iv])
                mean_flux  = np.mean(fluxes[bool_iv])
                sigm_flux  = np.std(fluxes[bool_iv])

                phases_mean = np.append(phases_mean, mean_phase)
                fluxes_mean = np.append(fluxes_mean, mean_flux)
                fluxes_sigm = np.append(fluxes_sigm, sigm_flux)    
            return  phases_mean, fluxes_mean, fluxes_sigm, intervals

        phases_mean, fluxes_mean, fluxes_sigm, intervals = execute_binning()

        nans = np.count_nonzero(np.isnan(phases_mean))
        if nans > 0:
            
            print("Found nan in the result. Will execute procedure once more to correct for the nan.")
            phases_mean, fluxes_mean, fluxes_sigm, intervals = execute_binning(correctnan=nans)
            phases_mean = phases_mean[~np.isnan(phases_mean)]
            fluxes_mean = fluxes_mean[~np.isnan(fluxes_mean)]
            fluxes_sigm = fluxes_sigm[~np.isnan(fluxes_sigm)]

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
            
        if testplot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(phases, fluxes, color="gainsboro", marker=".", linestyle="", zorder=1, label="original data")
            ax.errorbar(phases_mean, fluxes_mean, yerr=fluxes_sigm, color="tab:blue", marker=".", linestyle="", zorder=3, label="representative data")
            ax.legend(fontsize=16)
            ax.set_xlabel("x", fontsize=14)
            ax.set_ylabel("y", fontsize=14)
            ax.tick_params(axis="both", labelsize=14)
            plt.show()
            
        
        #time execution
        if timeit:
            task.end_task()

        
        return phases_mean, fluxes_mean, fluxes_sigm, intervals

    def sigma_clipping(fluxes, fluxes_mean, phases, phases_mean, intervals, clip_value_top, clip_value_bottom, times=[], testplot=False, timeit=False):
        """
            - cuts out all datapoints of fluxes (and phases and times) array which are outside of the interval [clip_value_bottom, clip_value_top] and returns the remaining array
            - used to get rid of outliers
            - clip_value_bottom and clip_value_top are usually defined as n*sigma, with  n = 1,2,3,... and sigma the standard deviation or Variance
            - if times is not specified, it will return an array of None with same size as fluxes    

            Parameters
            ----------
                - fluxes
                    - np.array
                    - fluxes to be sigma-clipped
                - fluxes_mean
                    - np.array
                    - mean values of fluxes to use as reference for clipping
                - phases
                    - np.array
                    - phases to be sigma-clipped
                - phases_mean
                    - np.array 
                    - phases corresponding to fluxes_mean
                - intervals
                    - np.array
                    - intervals to calculate the mean flux
                        - this will be used as reference for cutting datapoints
                - clip_value_top
                    - np.array
                    - top border of clipping
                - clip_value_bottom
                    - np.array
                    - bottom border of clipping
                - times
                    - np.array/list, optional
                    - times corresponding to fluxes
                        - only if existent
                    - the default is []
                - testplot
                    - bool, optinonal
                    - whether to show a testplot
                    - the default is False
                - timeit
                    - bool, optional
                    - specify wether to time the task and return the information or not.
                    - the default is False

            Raises
            ------
                - ValueError
                    - if the shapes of the inputs are not correct

            Returns
            -------
                - fluxes_sigcut
                    - fluxes after cutting out all values above clip_value_top and below clip_value bottom
                - phases_sigcut
                    - phases after cutting out all values above clip_value_top and below clip_value bottom
                - times_sigcut
                    - times after cutting out all values above clip_value_top and below clip_value bottom
                    - None if not provided
                - cut_f
                    - all cut-out values of fluxes
                - cut_p
                    - all cut-out values of phases
                - cut_t
                    - all cut-out values of times

            Dependencies
            ------------
                - numpy
                - matplotlib
            
            Comments
            --------

        """

        import numpy as np
        import matplotlib.pyplot as plt
        from astroLuSt.utility_astroLuSt import Time_stuff

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
            
        if testplot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(phases_sigcut, fluxes_sigcut, color="tab:blue", marker=".", linestyle="", zorder=2, label="Kept data")
            ax.plot(cut_p, cut_f, color="gainsboro", marker=".", linestyle="", zorder=1, label="Removed data")
            ax.legend(fontsize=16)
            ax.set_xlabel("x", fontsize=14)
            ax.set_ylabel("y", fontsize=14)
            ax.tick_params(axis="both", labelsize=14)
            plt.show()

        #time execution
        if timeit:
            task.end_task()
                        
        return fluxes_sigcut, phases_sigcut, times_sigcut, cut_f, cut_p, cut_t

class Units:
    """
        - Class to execute common uit conversions
        
        Methods
        -------
            - mags2fluxes
                - converts magnitudes to flux
            - flux2mags
                - converts flux to magnitudes
        
        Attributes
        ----------
                
        Dependencies
        ------------

        Comments
        --------
    """


    def __init__(self):
        pass
    
    def mags2fluxes(mags, m_ref):
        """
            - function to convert magnitudes to flux
            
            Parameters
            ----------
                - mags
                    - float, np.array
                    - magnitudes to be converted
                - m_ref
                    - float
                    - reference magnitude for the conversion
                        - this value is dependent on the passband filter in use

            Raises
            ------

            Returns
            -------
                - fluxes
                    - float, np.array
                    - fluxes corresponding to mags

            Dependencies
            ------------

            Comments
            --------

        """
        flux = 10**(0.4*(m_ref - mags))
        return flux


    def fluxes2mags(flux, m_ref):
        """
            - function to convert photon flux to magnitudes

            Parameters
            ----------
                - flux
                    - float, np.array
                    - fluxes to be converted
                - m_ref
                    - float
                    - reference magnitude for the conversion
                        - this value is dependent on the passband filter in use

            Raises
            ------

            Returns
            -------
                - mags
                    - float, np.array
                    - magnitudes corresponding to fluxes

            Dependencies
            ------------

            Comments
            --------
        """
        import numpy as np
        mags = -2.5*np.log10(flux) + m_ref
        return mags

class Synthetic_Curves:
    """
        - Class to generate synthetic curves
        - useful to create synthetic training data for AI
        
        Methods
        -------
            - 
        
        Attributes
        ----------
                
        Dependencies
        ------------

        Comments
        --------
    """
    def __init__(self):
        pass

    def synth_eb(mu1=-0.3, mu2=0.3, mu3=0, sigma1=0.01, sigma2=0.045, frequ3=2, scale1=0.04, scale2=0.04, scale3=0.1,
                 dip1_add=0.01, dip2_add=0.01, noise_scale=0, total_shift=0, fluxmin=0.8, resolution=100, dip_border_factor=4.5,
                 total_eclipse1=False, total_eclipse2=False, testplot=True, timeit=False):
        """
            - function to create synthetic lightcurves of eclipsing binaries in phase space
                - on the interval [-0.5,0.5]
            - the resulting function is by far close to real observations or physics
                - it is however useful e.g.:
                    - getting an initialization of weigths for NNs
            - creation done by superpositioning in total 6 gaussians and one sin
                - 3 gaussians for each dip
                - the sin for a basline flux variation
            - if you need higher quality data I strongly suggest you check out PHOEBE
                - PHysics Of Eclipsing BinariEs
                - http://phoebe-project.org/

            Parameters
            ----------
                - mu1
                    - float, optional
                    - mean of the first dip
                    - the default is -0.3
                - mu2
                    - float, optional
                    - mean of the second dip
                    - the default is 0.3 
                - mu3
                    - float, optional
                    - zeropoint of the continuum flux
                    - the default is 0
                - sigma1
                    - float, optional
                    - standard deviation of the first dip
                    - the default is 0.01
                - sigma2
                    - float, optional
                    - standard deviation of the second dip
                    - the default is 0.045
                - frequ3
                    - float, optional
                    - some frequency measure
                    - the default is 2
                - scale1
                    - float, optional
                    - scaling factor to increase/decreace depth of first dip
                    - the default is 0.04
                - scale2
                    - float, optional
                    - scaling factor to increase/decreace depth of second dip
                    - the default is 0.04
                - scale3
                    - float, optional
                    - scaling factor to increase/decreace depth of continuum flux
                    - the default is 0.01
                - dip1_add
                    - float, optional
                    - offset of the side-parts of first dip
                    - the default is 0.01
                - dip2_add
                    - float, optional
                    - offset of the side-parts of second dip
                    - the default is 0.01
                - noise_scale
                    - float, optional
                    - how much noise to add to the generated curve
                    - the default is 0
                - total_shift
                    - float, optional
                    - some total shift of the curve
                        - to eneable dips reaching over the boundaries
                    - the default is 0
                - fluxmin
                    - float, optional
                    - the lowst fluxvalue of the smooth curve
                    - the whole curve will get scaled to be in the interval [1, fluxmin]
                    - the default 0.8
                - resolution
                    - int, optional
                    - number of datapoints to resolve the generated curve
                    - the defalut is 100
                - dip_border_factor
                    - float, optional
                    - factor on how much of the area around each dip to consider as part of the dip
                        - will be the factor the standard deviations (sigma) will get multiplied with
                    - the default is 4.5
                - total_eclipse1
                    - bool, optional
                    - wheter the generated first dip has to contain a total eclipse or not
                        - otherwise the dip shape will be a random gaussian
                    - the default is False
                - total_eclipse2
                    - bool, optional
                    - wheter the generated second dip has to contain a total eclipse or not
                        - otherwise the dip shape will be a random gaussian
                    - the default is False
                - testplot
                    - bool, optional
                    - wheter to create a test plot of the created lightcurve
                    - the default is True
                -timeit
                    - bool, optional
                    - whether to time the execution of the generation
                    - the default is False

            Raises
            ------
                - ValueError
                    - if parameters out of bounds

            Returns
            -------
                - dict containing
                    - fluxes
                        - np.array
                        - generated fluxes in some (non-scientific) flux-unit
                    - relative_fluxes
                        - np.array
                        - generated normalized relative fluxes 
                    - phases
                        - np.array
                        - generated phases
                        - with total_shift = 0
                    - shifted_phases
                        - np.array
                        - generated phases
                        - with total_shift != 0
                    - dip_positions
                        - list
                        - positions of the dips
                        - with total_shift != 0
                    - dip_borders
                        - list
                        - estimated borders of the dips
                        - with total_shift != 0
                    - dip1_type
                        - str
                        - classification of the first dip
                    - dip2_type
                        - str
                        - classification of the second dip
                    - dip1
                        - np.array
                        - underlying curve of first dip
                    - dip1_center
                        - np.array
                        - center gaussian of first dip
                    - dip1_pos
                        - np.array
                        - positive offset-part of first dip
                    - dip1_neg
                        - np.array
                        - negative offset-part of first dip
                    - dip2
                        - np.array
                        - underlying curve of first dip
                    - dip2_center
                        - np.array
                        - center gaussian of first dip
                    - dip2_pos
                        - np.array
                        - positive offset-part of first dip
                    - dip2_neg
                        - np.array
                        - negative offset-part of first dip
                    - basline_flux
                        - np.array
                        - sinusoidal baseline flux variation
                    - noise
                        - np.array
                        - artificially added noise
            
            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
                - Check out PHOEBE for more accurate synthetic LCs
                    - http://phoebe-project.org/
                - calls astroLuSts periodic_shift()

        """

        import numpy as np
        import matplotlib.pyplot as plt
        import astroLuSt.preprocessing.data_astroLuSt as ald

        def gaussian(times, mu, sigma):
            """
                - function to create a gaussian with center mu and stadard-deviation sigma
            """
            Q1 = sigma*np.sqrt(2*np.pi)
            exp1 = -0.5*(times-mu)**2 / sigma**2
            return np.exp(exp1)/Q1

        #check if input values are correct
        warningmessage1 = "%s should be in the interval [%g, %g] in order to avoid overlapping dips!"
        warningmessage2 = "%s should be in the interval [%g, %g]!"
        warningmessage4 = "%s has to be in the interval [%g, %g], in order to prevent empty masks!"
        if mu1 < -0.3 or -0.2 < mu1:
            print(warningmessage1%("mu1", -0.3, -0.2))
        if mu2 < 0.2 or 0.3 < mu2:
            print(warningmessage1%("mu2", 0.2, 0.3))
        if mu3 < -0.5 or 0.5 < mu3:
            print(warningmessage1%("mu3", -0.5, -0.5))
        if sigma1 < 0.01 or 0.045 < sigma1:
            print(warningmessage1%("sigma1", 0.01, 0.045))
        if sigma2 < 0.01 or 0.045 < sigma2:
            print(warningmessage1%("sigma2", 0.01, 0.045))
        if frequ3 < 0 or 6 < frequ3:
            print(warningmessage2%("frequ3", 0, 4))
        if scale1 < 0 or 0.04 < scale1:
            print(warningmessage1%("scale1", 0, 0.04))
        if scale2 < 0 or 0.04 < scale2:
            print(warningmessage1%("scale2", 0, 0.04))
        if scale3 < 0 or 0.15 < scale3:
            print(warningmessage2%("scale3", 0, 0.03))
        if dip1_add < 1e-3 or 0.01 < dip1_add:
            raise ValueError(warningmessage4%("dip1_add", 0, 0.01))
        if dip2_add < 1e-3 or 0.01 < dip2_add:
            raise ValueError(warningmessage4%("dip2_add", 0, 0.01))
        if total_shift < -0.5 or 0.5 < total_shift:
            raise ValueError("total_shift has to be in the interval [-0.5,0.5]")

        #set some initialization variables
        phases = np.linspace(-0.5, 0.5, resolution)

        #shift x_vals to enable dip reaching over boundary
        phases_shifted = ald.Data_LuSt.periodic_shift(phases, total_shift, [-0.5,0.5])
        mu1_shifted = ald.Data_LuSt.periodic_shift(np.array([mu1]), total_shift, [-0.5,0.5])[0]
        mu2_shifted = ald.Data_LuSt.periodic_shift(np.array([mu2]), total_shift, [-0.5,0.5])[0]
        
        #first dip
        if total_eclipse1:
            sigma1 = 0.5*np.abs((mu1+dip1_add) - (mu1-dip1_add))
        dip1_center = -scale1*gaussian(phases, mu1, sigma1)*(not total_eclipse1) #0 if total eclipse
        dip1_pos    = -scale1*gaussian(phases, mu1+dip1_add, sigma1)
        dip1_neg    = -scale1*gaussian(phases, mu1-dip1_add, sigma1)
        dip1        = dip1_neg + dip1_center + dip1_pos
            
        #second dip
        if total_eclipse2:
            sigma2 = 0.5*np.abs((mu2+dip2_add) - (mu2-dip2_add))
        dip2_center = -scale2*gaussian(phases, mu2, sigma2)*(not total_eclipse2) #0 if total eclipse
        dip2_pos    = -scale2*gaussian(phases, mu2+dip2_add, sigma2) 
        dip2_neg    = -scale2*gaussian(phases, mu2-dip2_add, sigma2)
        dip2        = dip2_neg + dip2_center + dip2_pos

        #basline flux
        # f_bl = scale3*gaussian(phases, mu3, sigma3)
        f_bl = scale3*np.sin(2*frequ3*np.pi*(phases + mu3))

        #put everything together to get synthetic LC
        fluxes = f_bl + dip1 + dip2
        noise = np.random.normal(size=np.shape(fluxes))*noise_scale
        relative_fluxes = np.interp(fluxes, (fluxes.min(), fluxes.max()), (fluxmin,1))  #normalize to 1 (such that sinusoidal baseline is centered around 1)
        relative_fluxes += noise   #add some noise

        #classify dip borders
        dip1_border1 = ald.Data_LuSt.periodic_shift(np.array([mu1-np.abs(dip_border_factor*sigma1)]), total_shift, [-0.5,0.5])[0]
        dip1_border2 = ald.Data_LuSt.periodic_shift(np.array([mu1+np.abs(dip_border_factor*sigma1)]), total_shift, [-0.5,0.5])[0]
        dip2_border1 = ald.Data_LuSt.periodic_shift(np.array([mu2-np.abs(dip_border_factor*sigma2)]), total_shift, [-0.5,0.5])[0]
        dip2_border2 = ald.Data_LuSt.periodic_shift(np.array([mu2+np.abs(dip_border_factor*sigma2)]), total_shift, [-0.5,0.5])[0]

        #masks to separate dips 
        if dip1_border1 < dip1_border2:
            mask1 = (dip1_border1<phases_shifted)&(phases_shifted<dip1_border2) #mask of dip1
        else:
            mask1 = (dip1_border1<phases_shifted)|(phases_shifted<dip1_border2) #mask of dip1 if reaching over border
        if dip2_border1 < dip2_border2:
            mask2 = (dip2_border1<phases_shifted)&(phases_shifted<dip2_border2) #mask of dip2
        else:
            mask2 = (dip2_border1<phases_shifted)|(phases_shifted<dip2_border2) #mask of dip2 if reaching over border


        #classify if total/partial eclipse
        std_crit = 2e-2
        d1_std = np.std(relative_fluxes[mask1])
        d2_std = np.std(relative_fluxes[mask2])
        
        if total_eclipse1:
            #total if specified
            d1_type = "total"
        elif d1_std <= std_crit:
            #non-eclipsing if indistinguishable from baseline
            d1_type = "non-eclipsing"
        else:
            #partial otherwise
            d1_type = "partial"

        if total_eclipse2:
            #total if specified
            d2_type = "total"
        elif d2_std <= std_crit:
            #non-eclipsing if indistinguishable from baseline
            d2_type = "non-eclipsing"
        else:
            #partial otherwise
            d2_type = "partial"
        
        #assign classification corretly
        if np.min(dip1) < np.min(dip2):
            d1_lab = f"Primary eclipse ({d1_type})"
            d2_lab = f"Secondary eclipse ({d2_type})"
        else:
            d1_lab = f"Secondary eclipse ({d1_type})"
            d2_lab = f"Primary eclipse ({d2_type})"    

        if testplot:
            
            dip1_color = "tab:blue"
            dip2_color = "tab:orange"
            bl_color = "tab:grey"
            d1_color = "tab:blue"
            d2_color = "tab:orange"
            
            #insert nan to avoid connecting start and end
            insidx = np.argmin(phases_shifted)
            phases_shifted_plt = np.insert(phases_shifted, insidx, np.NaN)
            relative_fluxes_plt = np.insert(relative_fluxes, insidx, np.NaN)
            noise_plt = np.insert(noise, insidx, np.NaN)

            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax1.set_title("Generation parts", fontsize=16)
            ax1.plot(phases, dip1, color=dip1_color, alpha=1)#, label=d1_lab)
            ax1.plot(phases, dip1_center, color=dip1_color, alpha=.5)
            ax1.plot(phases, dip1_pos, color=dip1_color, alpha=.5)
            ax1.plot(phases, dip1_neg, color=dip1_color, alpha=.5)
            ax1.plot(phases, dip2, color=dip2_color, alpha=1)#, label=d2_lab)
            ax1.plot(phases, dip2_center, color=dip2_color, alpha=.5)
            ax1.plot(phases, dip2_pos, color=dip2_color, alpha=.5)
            ax1.plot(phases, dip2_neg, color=dip2_color, alpha=.5)
            ax1.plot(phases, f_bl, color=bl_color, alpha=.5, label="Baseline flux")
            # ax1.plot(phases, fluxes, color=whole_color, alpha=.7, label="Synthetic lightcurve")
            ax1.set_xticks([])
            ax1.set_ylabel("Flux-like", fontsize=16)
            ax2 = fig.add_subplot(212)
            ax2.set_title("Generated LC", fontsize=16)
            ax2.plot(phases_shifted[~mask1&~mask2], relative_fluxes[~mask1&~mask2], color="tab:green", marker=".", linestyle="", alpha=1, label="Continuum flux")
            ax2.plot(phases_shifted[mask1], relative_fluxes[mask1], color=d1_color, marker=".", linestyle="", alpha=1, label=d1_lab)
            ax2.plot(phases_shifted[mask2], relative_fluxes[mask2], color=d2_color, marker=".", linestyle="", alpha=1, label=d2_lab)
            ax2.plot(phases_shifted_plt, relative_fluxes_plt-noise_plt, color="k", alpha=1, label="Non-noisy LC")
            ax2.vlines((mu1_shifted, mu2_shifted), ymin=relative_fluxes.min(), ymax=relative_fluxes.max(), color="grey", linestyles="--")#, label="Dippositions")
            ax2.vlines((dip1_border1, dip1_border2, dip2_border1, dip2_border2), ymin=relative_fluxes.min(), ymax=relative_fluxes.max(), color="gainsboro", linestyles="--")#, label="Dipborders")
            ax2.set_xlabel("Phase", fontsize=16)
            ax2.set_ylabel("Relative-flux", fontsize=16)
            fig.legend()
            plt.tight_layout()
            plt.show()


        return {
            "fluxes":fluxes,
            "relative_fluxes":relative_fluxes,
            "phases":phases,
            "shifted_phases":phases_shifted,
            "dip_positions":[mu1_shifted, mu2_shifted],
            "dip_borders":[dip1_border1,dip1_border2,dip2_border1,dip2_border2],
            "dip1_type":d1_type,
            "dip2_type":d2_type,
            "dip1":dip1,
            "dip1_center":dip1_center,
            "dip1_pos":dip1_pos,
            "dip1_neg":dip1_neg,
            "dip2":dip2,
            "dip2_center":dip2_center,
            "dip2_pos":dip2_pos,
            "dip2_neg":dip2_neg,
            "basline_flux":f_bl,
            "noise":noise,
        }

    def periodize(phases, fluxes, period, repetitions, testplot=True, timeit=False):
        """
            - function to create a periodic signal out of a time-series given in phase space
            - works for phases given in the interval [-0.5, 0.5]

            Parameters
            ----------
                - phases
                    - np.array
                    - array of phases to be converted
                - fluxes
                    - np.array
                    - array of the corresponding fluxes
                - period
                    - float
                    - period to use for the repetition
                - testplot
                    - bool, optional
                    - whether to show a test-plot
                    - the default is True
                - timeit
                    - bool, optional
                    - whether to time the execution
                    - the default is False

            Raises
            ------

            Returns
            -------
                - times_periodized
                    - times of the periodized signal
                - fluxes_periodized
                    - fluxes of the periodized signal

            Dependencies
            ------------
                - numpy
                - matplotlib

            Comments
            --------
                - if you want to use a phase space defined on [0,1], simply call pass phases-0.5 to the function
        
        """
        import astroLuSt.preprocessing.data_astroLuSt as ald
        import numpy as np
        import matplotlib.pyplot as plt

        times = ald.Data_LuSt.phase2time(phases+0.5, period, timeit=False)
        times_periodized = np.array([])
        fluxes_periodized = np.array([])
        for r in range(repetitions):
            times_add = times + (r*period)
            times_periodized = np.append(times_periodized, times_add)
            fluxes_periodized = np.append(fluxes_periodized, fluxes)
        
        sortidx = np.argsort(times_periodized)
        times_periodized = times_periodized[sortidx]
        fluxes_periodized = fluxes_periodized[sortidx]

        if testplot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(times_periodized, fluxes_periodized, color="tab:blue", linestyle="", marker=".", label="Periodized signal")
            ax1.set_xlabel("Time [Period-unit]", fontsize=16)
            ax1.set_ylabel("Relative flux", fontsize=16)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return times_periodized, fluxes_periodized, 
