

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


