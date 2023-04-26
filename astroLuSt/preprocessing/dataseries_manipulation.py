
#%%imports
import matplotlib.pyplot as plt
import numpy as np

#%%definitions

def fold(
    time:np.ndarray,
    period:float, tref:float=None,
    verbose=0) -> np.ndarray:
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
    ) -> tuple:
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

def periodic_shift(input_array:np.ndarray, shift:float, borders:list, testplot:bool=False, verbose:int=0):
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
    
    #time execution



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
