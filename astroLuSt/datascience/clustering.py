
#%%imports
import numpy as np

#%%definitions
def kdist(X:np.ndarray, k:int=4) -> np.ndarray:
    """
        - function to return the distance of every datapoint in "X" to its k-th nearest neighbour
        - useful for generating kdist-graphs ([Ester et al., 1996](https://ui.adsabs.harvard.edu/abs/1996kddm.conf..226E/abstract))
            - kdist-plots are used to determing the epsilon environment for DBSCAN

        Parameters
        ----------
            - `X`
                - `np.ndarray`
                - some dataset
                - dataset to determine the distance of every point to its k-th neighbor
            - `k`
                - `int`, optional
                - which neighbor to calculate the distance to
                - the default is `4`
                    - that is the standard value of `"npoints"` used for DBSCAN [Ester et al., 1996](https://ui.adsabs.harvard.edu/abs/1996kddm.conf..226E/abstract)

        Raises
        ------

        Returns
        -------
            - 'kth_dist'
                - `np.ndarray`
                - array of sorted kth-distances
        
        Dependencies
        ------------
            - `numpy`

        Comments
        --------
    """

    #all distance kombinations
    alldist = np.linalg.norm(X - X[:,None], axis=-1)

    kth_dist = np.sort(np.sort(alldist, axis=1)[:,k])[::-1]  #first element is the distance to the point itself
    
    return kth_dist
