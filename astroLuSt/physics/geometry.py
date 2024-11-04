

#%%imports
import numpy as np

#%%definitions

def angular_distance(
    ra:np.ndarray, dec:np.ndarray,
    approx:str=None,
    ) -> np.ndarray:
    """
        - function to calcualte the angular distance between two objects on the night-sky

        Parameters
        ----------
            - `ra`
                - `np.ndarray`
                    - 2d
                    - each row contains ra for target1 and target2, respectively
                - right ascension of the object(s)
                - has to be between `0` and `2*pi`
            - `dec`
                - `np.ndarray`
                    - 2d
                    - each row contains dec for target1 and target2, respectively
                - declination of the object(s)
                - has to be between `-pi/2` and `pi/2`
            - `approx`
                - `str`, optional
                - which approximation to use
                - the default is `None`
                    - will use the general case

        Raises
        ------

        Returns
        -------
            - `theta`
                - `np.ndarray`
                - same shape as `ra` and `dec` along the first axis
                - angular distance between each sample in `ra` and `dec`

        Dependencies
        ------------
            - `numpy`

        Comments
        --------
    """
    if len(ra.shape) > 1 and len(dec.shape) > 1:
        if ra.shape[1] != 2 or dec.shape[1] != 2:
            raise ValueError(f'"ra" and "dec" have to be two dimensional with 2 entries in the second axis (i.e. shape of (-1, 2)) but have shape {ra.shape} and {dec.shape}. Try reshaping them with np.ndarray.reshape(-1,2).')
    else:
        raise ValueError(f'"ra" and "dec" have to be two-dimensional')

    if np.any((ra < 0)|(ra>2*np.pi)):
        raise ValueError('"ra" has to be between 0 and 2*pi')
    if np.any((dec < -np.pi/2)|(dec>np.pi/2)):
        raise ValueError('"dec" has to be between -pi/2 and pi/2')

    if approx == 'smallangle':
        theta = np.sqrt(((ra[:,0]-ra[:,1])*np.cos(dec[:,0]))**2 + (dec[:,0] - dec[:,1])**2)
    elif approx == 'planar':
        theta = np.sqrt((ra[:,0]-ra[:,1])**2 + (dec[:,0]-dec[:,1])**2)
    else:
        theta = np.arccos(np.sin(dec[:,0]*np.sin(dec[:,1])) + np.cos(dec[:,0])*np.cos(dec[:,1])*np.cos(ra[:,0]-ra[:,1]))

    return theta

