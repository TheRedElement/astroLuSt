
#%%imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Union, Tuple, Callable

#%%definitions

def polyfit2d(
    x:np.ndarray, y:np.ndarray, z:np.ndarray,
    deg:int=1,
    verbose:int=0
    ) -> Tuple[PolynomialFeatures, LinearRegression, np.ndarray, np.ndarray]:
    """
        - function to make a polynomial fit to a dataset of 3 variables

        Parameters
        ----------
            - x
                - np.ndarray
                - independent variable 1
            - y
                - np.ndarray
                - independent variable 2
            - z
                - np.ndarray
                - dependent variable 
                    - z(x,y)
            - deg
                - int, optional
                - degree of the polynomial to fit
            - verbose
                - int, optional
                - verbosity level
                - the default is 0
        
        Raises
        ------

        Returns
        -------
            - poly
                - sklearn.preprocessing.PolynomialFeatures instance
                - fitted model to generate all polynomial features of two inputs in dependence in deg
            - poly_reg_model
                - sklearn.linear_model.LinearRegression instance
                - the fitted model
            - coeffs
                - np.ndarray
                - coefficients of the fitted model
            - intercept
                - float
                - intercept of the fitted model
        
        Dependencies
        ------------
            - matplotlib
            - numpy
            - sklearn
    """

    poly = PolynomialFeatures(degree=deg, include_bias=False)
    poly.fit(np.array([x, y]).T)
    poly_features = poly.fit_transform(np.array([x, y]).T)

    # print(poly_features)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, z)
    
    coeffs = poly_reg_model.coef_
    intercept = poly_reg_model.intercept_

    if verbose > 2:
        z_pred = poly_reg_model.predict(poly_features)
        # print(z_pred)
        # print(coeffs)
        # print(intercept)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(221, projection=None)
        ax2 = fig.add_subplot(223, projection=None)
        ax3 = fig.add_subplot(122, projection='3d')
        
        ax1.scatter(x, z, label='Data')
        ax2.scatter(y, z, label='Data')
        ax3.scatter(x, y, z, label='Data')
        
        ax1.scatter(x, z_pred, label='Model')
        ax2.scatter(y, z_pred, label='Model')
        ax3.scatter(x, y, z_pred, color='r', label='Model')

        ax1.set_xlabel('x')
        ax2.set_xlabel('y')
        ax3.set_xlabel('x')
        ax1.set_ylabel('z')
        ax2.set_ylabel('z')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')

        ax1.legend()
        plt.tight_layout()
        plt.show()


    return poly, poly_reg_model, coeffs, intercept