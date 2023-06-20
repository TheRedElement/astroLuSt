

#%%imports
import numpy as np
from typing import Union

#%%definitions
def phase2time(
    phase:Union[np.ndarray,float],
    period:Union[np.ndarray,float],
    tref:Union[np.ndarray,float]=0,
    verbose:int=0,
    ) -> Union[np.ndarray,float]:
    """
        - converts a given array of phases into its respective time equivalent

        Parameters
        ----------
            - `phases`
                - np.ndarray, float
                - the phases to convert to times
            - `period`
                - np.ndarray, float
                - the given period(s) the phase describes
            - `tref`
                - np.ndarray, float, optional
                - reference time
                    - i.e. offset from `time==0`
                - the default is 0
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Raises
        ------

        Returns
        -------
            - `time`
                - np.array, float
                - the resulting time array, when the phases are converted 

        Dependencies
        ------------
            - typing

        Comments
        --------
            - operates with phases in the interval [0,1]
    """

    time = phase*period + tref
    
    return time