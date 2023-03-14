
#%%imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#%%definitions

def polyfit2d(
    x:np.ndarray, y:np.ndarray, z:np.ndarray,
    deg:int=1,
    verbose:int=0
    ):
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
        print(z_pred)
        print(coeffs)
        print(intercept)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, label='Data')
        ax1.scatter(x, y, z_pred, color='r', label='Model')

        ax1.legend()

        plt.show()


    return poly_reg_model, coeffs, intercept