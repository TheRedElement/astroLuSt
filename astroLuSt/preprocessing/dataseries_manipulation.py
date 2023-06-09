
#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Tuple, Callable

#%%definitions

def phase2time(
    phase:Union[np.ndarray,float],
    period:Union[np.ndarray,float],
    tref:Union[np.ndarray,float]=0,
    verbose:int=0,
    ):
    """
        - converts a given array of phases into its respective time equivalent

        Parameters
        ----------
            - phases
                - np.ndarray, float
                - the phases to convert to times
            - period
                - np.ndarray, float
                - the given period(s) the phase describes
            - tref
                - np.ndarray, float, optional
                - reference time
                    - i.e. offset from time=0
                - the default is 0
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

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

    time = phase*period + tref
    
    return time

def fold(
    time:np.ndarray,
    period:float, tref:float=None,
    verbose=0) -> Tuple[np.ndarray, np.ndarray]:
    """
        - takes an array of times
            - folds it onto a specified period into phase space
            - returns folded array of phases (in interval 0 to 1)

        Parameters
        ----------
            - time
                - np.array
                - times to be folded with the specified period
            - period 
                - float
                - period to fold the times onto
            - tref
                - float, optional
                - reference time to consider when folding the lightcurve
                - the default is None
                    - will take min(time) as reference
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------

        Returns
        -------
            - phases_folded
                - np.array
                - phases corresponding to the given time folded onto the period
            - periods_folded
                - np.array
                - phases_folded in time domain

        Dependencies
        ------------
            - numpy

        Comments
        --------
    """

    if tref is None:
        tref = time.min()

    delta_t = time-tref
    phases = delta_t/period
    
    #fold phases by getting the remainder of the division by the ones-value 
    #this equals getting the decimal numbers of that specific value
    #+1 because else a division by 0 would occur
    #floor always rounds down a value to the ones (returns everything before decimal point)
    phases_folded = np.mod(phases,1)

    periods_folded = phases_folded * period

    return phases_folded, periods_folded

def resample(
    x:np.ndarray, y:np.ndarray,
    ndatapoints:int=50,
    sort_before:bool=True,
    verbose:int=0
    ) -> Tuple[np.ndarray, np.ndarray, Figure, plt.Axes]:
    """
        - function to resample a dataseries y(x) to nfeatures new datapoints via interpolation

        Parameters
        ----------
            - x
                - np.ndarray
                - independent input variable x
            - y
                - np.ndarray
                - dependent variable (y(x))
            - ndatapoints
                - int, optional
                - number of datapoints of the resampled dataseries
                - the default is 50
            - sort_before
                - bool, optional
                - whether to sort the input arrays x and y with regards to x before resampling
                - the default is True
            - verbose
                -  int optional
                - verbosity level
                - the default is 0
            
        Raises
        ------

        Returns
        -------
            - interp_x
                - np.ndarray
                - resamples array of x
            - interp_y
                - np.ndarray
                - resamples array of y

        Dependencies
        ------------
            - numpy
            - matplotlib
        
        Comments
        --------

    """

    #try converting to numpy (otherwise bin generation might fail)
    if not isinstance(x, np.ndarray):
        try:
            x = x.to_numpy()
        except:
            raise TypeError(f'"x" hat to be of type np.ndarray and not {type(x)}')
    if not isinstance(y, np.ndarray):
        try:
            y = y.to_numpy()
        except:
            raise TypeError(f'"y" hat to be of type np.ndarray and not {type(y)}')
            

    if sort_before:
        idxs = np.argsort(x)
        x_, y_ = x[idxs], y[idxs]

    interp_x = np.linspace(0, 1, ndatapoints)

    interp_y =  np.interp(interp_x, x_, y_)

    if verbose > 1:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y, label="Input")
        ax1.scatter(interp_x, interp_y, label="Resampled")

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        plt.tight_layout()
        plt.show()

        axs = fig.axes
    else:
        fig = None
        axs = None
    

    return interp_x, interp_y, fig, axs

def periodic_shift(
    input_array:np.ndarray,
    shift:float, borders:Union[list,np.ndarray],
    testplot:bool=False,
    verbose:int=0
    ) -> np.ndarray:
    """
        - function to shift an array considering periodic boundaries

        Parameters
        ----------
            - input_array
                - np.ndarray
                - array to be shifted along an interval with periodic boundaries
            - shift
                - float
                - magnitude of the shift to apply to the array
            - borders
                - list/np.ndarray
                - upper and lower boundary of the periodic interval
            - testplot
                - bool, optional
                - wether to show a testplot
                - the default is False
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

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
    if (type(testplot) != bool):
        raise TypeError("testplot has to be of type bool")



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
    

    if verbose > 0:
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

def periodize(
    y:np.ndarray, period:Tuple[float,np.ndarray],
    repetitions:int=2,
    x:np.ndarray=None, 
    testplot:bool=False,
    ) -> Tuple[np.ndarray,np.ndarray]:
    """
        - function to create a periodic signal out of a time-series given in phase space
        - works for phases given in the interval [0, 1]

        Parameters
        ----------
            - y
                - np.ndarray
                - has to be at least 2d
                - y-values to be periodized
            - period
                - np.ndarray, float
                - if np.ndarray, has to be of same length as y
                - period to use for the repetition
            - repetitions
                - int, optional
                - number of times the signal (y) shall be repeated
                - the default is 2
            - x
                - np.ndarray, optional
                - x-values to be periodized
                - has to be of same shape as y
                - usually phases
                    - i.e. an array from 0 to 1
                - the default is None
                    - will generate phases for every samples in y
            - testplot
                - bool, optional
                - whether to show a test-plot
                - the default is False

        Raises
        ------

        Returns
        -------
            - x_periodized
                - np.ndarray
                - same shape as y
                - times of the periodized signal
            - y_periodized
                - np.ndarray
                - same shape as y
                - y-values of the periodized signal

        Dependencies
        ------------
            - numpy
            - matplotlib

        Comments
        --------
    
    """

    #initializie x if not passed
    if x is None:
        if len(y.shape) == 1:
            nsamples = 1
            npoints  = y.shape[0]
        else:
            nsamples = y.shape[0]
            npoints  = y.shape[1]

        x = np.array([np.linspace(0, 1, npoints) for i in range(nsamples)])

    #reshape if 1d array was passed
    if len(x.shape) == 1: x = x.reshape(1,x.shape[0])
    if len(y.shape) == 1: y = y.reshape(1,y.shape[0])

    x_times = phase2time(x, period)

    x_periodized = np.empty((x_times.shape[0], x_times.shape[1]*repetitions))
    y_periodized = np.empty((x_times.shape[0], x_times.shape[1]*repetitions))

    for r in range(repetitions):
        times_add = x_times + (r*period)

        startidx = r*x_times.shape[1]
        endidx   = startidx+times_add.shape[1]

        x_periodized[:,startidx:endidx] = times_add
        y_periodized[:,startidx:endidx] = y


    if testplot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for xp, yp in zip(x_periodized, y_periodized):
            ax1.scatter(xp, yp, label="Periodized Signal")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return x_periodized, y_periodized, 

def periodic_expansion(
    x:np.ndarray, y:np.ndarray,
    x_ref_min:float=0, x_ref_max:float=0,
    minmax:str="max",
    testplot=False,
    ):
    """
        - function to expand a periodic timeseries on either side
            - takes all datapoints up to a reference phase
            - appends them to the original array according to specification  

        Parameters
        ----------
            - x
                - np.ndarray
                - x-values of the datapoints to be expanded
            - y 
                - np.ndarray
                - y-values of the datapoints to be expanded
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
            - testplot
                - bool, optional
                - whether to show a testplot of the result
                - the default is False

        Raises
        ------
            - ValueError
                - if 'minmax' gets passed a wrong argument
            - TypeError
                - if 'phase_ref' gets passed a wrong type

        Returns
        -------
            - expanded_x
                - np.ndarray
                - x-values including the expanded part
            - expanded_y
                - np.ndarray
                - y-values including the expanded part

        Dependencies
        ------------
            - numpy
            - matplotlib

        Comments
        --------
    """
    

    import numpy as np
    import matplotlib.pyplot as plt

    #sort to get correct appendix in the end
    sortidx = np.argsort(x)
    x = x[sortidx]
    y = y[sortidx]

    #append to maximum
    if minmax == "max":
        x_bool = (x < x_ref_max)
        appendix_phases = np.nanmax(x) + (x[phase_bool] - np.nanmin(x))
        x_ref = x_ref_max
    #append to minimum
    elif minmax == "min":
        phase_bool = (x > x_ref_min)
        appendix_phases = np.nanmin(x) - (np.nanmax(x) - x[phase_bool])
        x_ref = x_ref_min
    elif minmax == "both":
        if x_ref_min < x_ref_max:
            raise ValueError("'x_ref_min' has to be greater or equal than 'x_ref_max' ")
        x_ref = [x_ref_min, x_ref_max]
        x_bool_max = (x < x_ref_max)
        x_bool_min = (x > x_ref_min)
        x_bool = x_bool_max|x_bool_min
        appendix_x_max = np.nanmax(x) + (x[x_bool_max] - np.nanmin(x))
        appendix_x_min = np.nanmin(x) - (np.nanmax(x) - x[x_bool_min])
        appendix_x = np.append(appendix_x_max, appendix_x_min)
    else:
        raise ValueError("'minmax' has to bei either 'min', 'max' or 'both'!")
    
    appendix_y = y[phase_bool]


    expanded_x = np.append(x, appendix_x)
    expanded_y = np.append(y, appendix_y)

    if testplot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x,          y,          color="tab:grey", alpha=0.5, zorder=2, label="Original Input")
        ax1.scatter(expanded_x, expanded_y, color="tab:blue", alpha=1,   zorder=1, label="Expanded Input")
    
        if minmax == 'both':
            ax1.axvline(x_ref[0], color="g", linestyle="--", label="Reference Phase")
            ax1.axvline(x_ref[1], color="g", linestyle="--")
        else:
            ax1.axvline(x_ref,    color="g", linestyle="--", label="Reference Phase")
        
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        
        ax1.legend()

        plt.tight_layout()
        plt.show()
    
    return expanded_x, expanded_y