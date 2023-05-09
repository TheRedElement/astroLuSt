

#%%imports
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Callable

#%%definitions
class SamplewiseScaler:
    """
        - class to scale some data not feature-wise but sample wise

        Attributes
        ----------
            - scaler
                - class
                - some instance implementing a fit and transform method
                - the default is sklearn.preprocessing.MinMaxScaler()
                    - will scale each sample to have its fatures exactly between 0 and 1

        Methods
        -------
            - fit()
            - transform()
            - fit_transform()

        Dependencies
        ------------
            - numpy
            - sklearn

        Comments
        --------

    """

    def __init__(self,
        scaler:Callable=MinMaxScaler()
        ) -> None:
        
        self.scaler = scaler

        return
    
    def fit(self,
        X:np.ndarray, y:np.ndarray=None,
        ) -> None:
        """
            - fit method for the scaler
            - for compatibility with the sklearn module
            
            Parameters
            ----------
                - X
                    - np.ndarray, pd.DataFrame
                    - input data that shall be transformed
                    - contains samples as the rows and features as the columns
                - y
                    - np.ndarray
                    - does not affect the transformation
                    - included for compatibility

            Raises
            ------

            Returns
            -------

        """

        self.X_scaled = self.scaler.fit_transform(X.T).T

        return
    
    def transform(self,    
        ) -> np.ndarray:
        """
            - transform method for the scaler
            - for compatibility with the sklearn module
            
            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - self.X_scaled
                    - np.ndarray
                    - the scaled version of the input data X
            
        """
        return self.X_scaled
    
    def fit_transform(self,
        X:np.ndarray, y:np.ndarray=None,
        ) -> np.ndarray:
        """
            - method for the scaler that first fits and consecutively transforms the input data
            - for compatibility with the sklearn module
            
            Parameters
            ----------
                - X
                    - np.ndarray, pd.DataFrame
                    - input data that shall be transformed
                    - contains samples as the rows and features as the columns
                - y
                    - np.ndarray
                    - does not affect the transformation
                    - included for compatibility
            Raises
            ------

            Returns
            -------
                - self.X_scaled
                    - np.ndarray
                    - the scaled version of the input data X
            
        """

        self.fit(X, y)
        self.transform()

        return self.X_scaled

