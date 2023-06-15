

#%%imports
import numpy as np
from typing import Callable, Union

#%%definitions
class AxisScaler:
    """
        - class to scale some data not feature-wise but along given axes

        Attributes
        ----------
            - `scaler`
                - callable
                - some function
                    - has to have least at the following (positional) parameter
                        - `X`
                            - np.ndarray
                            - input array
                        - `axis`
                            - tuple, int
                            - denoting the axis to operate on
                    - returning one value `X_new`
                - if a string is passed one of the class methods will be used
                    - currently the following strings are allowed
                        - 'range_scaler'
                - the default is `None`
                    - will fallback to `self.range_scale()`
                        - scales featurewise into the interval [0,1]
            - `axis`
                - tuple, int
                - axis along which to apply the scaling
                - the default is `None`
                    - defaults to `0`
                    - will scale feature-wise
            - `scaler_kwargs`
                - dict, optional
                - any kwargs that will be passed to `scaler`
                - the default is `None`
                    - will be set to `{}`

        Infered Attributes
        ------------------
            - `internal_scalers`
                - list
                - contains all scalers that are internally implemented
            - `X_new`
                - np.ndarray
                - the scaled version of the input data `X`

        Methods
        -------
            - `range_scale()`
            - `fit()`
            - `transform()`
            - `fit_transform()`

        Raises
        ------
            - ValueError
                - if the string passed to `scaler` is not valid

        Dependencies
        ------------
            - numpy
            - typing

        Comments
        --------

    """

    def __init__(self,
        scaler:Union[str,Callable]=None,
        axis:Union[tuple,int]=None,
        scaler_kwargs:dict=None,
        ) -> None:
        

        #list of internally available scalers
        self.internal_scalers = ['range_scaler']

        if isinstance(scaler,str):
            if scaler not in self.internal_scalers:
                raise ValueError(f'`{scaler}` is not internally available. Try one of {self.internal_scalers}')
            elif scaler == 'range_scaler':
                self.scaler = self.range_scaler
        elif scaler is None:
                self.scaler = self.range_scaler
        else:
            self.scaler = scaler
        
        if axis is None:
            self.axis = 0
        else:
            self.axis = axis

        if scaler_kwargs is None:
            self.scaler_kwargs = {}
        else:
            self.scaler_kwargs = scaler_kwargs

        return
    
    def __repr__(self) -> str:
        
        return (
            f'AxisScaler(\n'
            f'    scaler={self.scaler.__name__},\n'
            f'    scaler_kwargs={self.scaler_kwargs},\n'
            f')'
        )
    
    def range_scaler(self,
        X:np.ndarray, axis:Union[tuple,int]=None,
        feature_range:tuple=(0,1), 
        ) -> np.ndarray:
        """
            - method that scales the input X along axis into feature-range

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - input data that shall be scaled
                - `feature_range`
                    - tuple, optional
                    - feature range into which to scale `X`
                    - the maximum and minimum of `X` w.r.t. `axis` will lye in that range 
                    - the default is `(0,1)`
                - `axis`
                    - tuple, int
                    - axis along which to apply the scaling
                    - will override `self.axis`
                    - the default is `None`
                        - will fallback to `self.axis`
            
            Raises
            ------

            Returns
            -------
                - `X_new`
                    - np.ndarray
                    - scaled version of `X`

            Comments
            --------
        """
        if axis is None:
            axis = self.axis

        X_min = np.nanmin(X, axis=axis, keepdims=True)
        X_max = np.nanmax(X, axis=axis, keepdims=True)
        fmin = np.nanmin(feature_range)
        fmax = np.nanmax(feature_range)

        X_new = (X-X_min)/(X_max - X_min) * (fmax-fmin) + fmin

        return X_new    
    
    def fit(self,
        X:np.ndarray, y:np.ndarray=None,
        scaler_kwargs:dict=None,
        ) -> None:
        """
            - fit method for the scaler
            - for compatibility with the sklearn module
            
            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - input data that shall be transformed
                    - contains samples as the rows and features as the columns
                - `y`
                    - np.ndarray, optional
                    - does not affect the transformation
                    - included for compatibility
                    - the default is `None`
                - `scaler_kwargs`
                    - dict, optional
                    - any kwargs that will be passed to `scaler`
                    - will override `self.scaler_kwargs`
                    - the default is `None`
                        - will fallback to `self.scaler_kwargs`         

            Raises
            ------

            Returns
            -------

            Comments
            --------

        """
        if scaler_kwargs is None:
            scaler_kwargs = self.scaler_kwargs
        
        self.X_new = self.scaler(X, **scaler_kwargs)

        return
    
    def transform(self,
        X:np.ndarray=None,
        ) -> np.ndarray:
        """
            - transform method for the scaler
            - for compatibility with the sklearn module
            
            Parameters
            ----------
                - `X`
                    - np.ndarray, optional
                    - input data that shall be transformed
                    - contains samples as the rows and features as the columns
                    - the default is `None`

            Raises
            ------

            Returns
            -------
                - `X_new`
                    - np.ndarray
                    - the scaled version of the input data `X`

            Comments
            --------
            
        """

        X_new = self.X_new
        
        return X_new
    
    def fit_transform(self,
        X:np.ndarray, y:np.ndarray=None,
        scaler_kwargs:dict=None
        ) -> np.ndarray:
        """
            - method for the scaler that first fits and consecutively transforms the input data
            - for compatibility with the sklearn module
            
            Parameters
            ----------
                - `X`
                    - np.ndarray, pd.DataFrame
                    - input data that shall be transformed
                    - contains samples as the rows and features as the columns
                - `y`
                    - np.ndarray
                    - does not affect the transformation
                    - included for compatibility
                    - the default is `None`
                - `scaler_kwargs`
                    - dict, optional
                    - any kwargs that will be passed to `scaler`
                    - will override `self.scaler_kwargs`
                    - the default is `None`
                        - will fallback to `self.scaler_kwargs`

            Raises
            ------

            Returns
            -------
                - `X_new`
                    - np.ndarray
                    - the scaled version of the input data `X`
            
            Comments
            --------
            
        """
        if scaler_kwargs is None:
            scaler_kwargs = self.scaler_kwargs

        self.fit(X, y, scaler_kwargs=scaler_kwargs)
        self.transform()

        X_new = self.X_new

        return X_new

