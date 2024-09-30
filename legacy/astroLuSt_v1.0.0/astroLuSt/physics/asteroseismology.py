

#%%imports
import numpy as np
import pandas as pd


#%%definitions
class ScalingRelations:
    """
        - class to apply the asteroseismic scaling relations
            - i.e. Gaulme et al., (2016) and reference therein
                - https://ui.adsabs.harvard.edu/abs/2016ApJ...832..121G/abstract

        Attributes
        ----------
            - `nu_max`
                - np.ndarray
                - frequency of maximum power
                - use same units as `delta_nu`
            - `delta_nu`
                - np.ndarray
                - large frequency separation
                - use same units as `nu_max`
            - `t_eff`
                - np.ndarray
                - effective temperature
                - use same units as `t_eff_sun`
            - `e_nu_max`
                - np.ndarray, optional
                - error corresponding to `nu_max`
                - the default is `None`
                    - no error considered
            - `e_delta_nu`
                - np.ndarray, optional
                - error to `delta_nu`
                - the default is `None`
                    - no error considered
            - `e_t_eff`
                - np.ndarray, optional
                - error to `t_eff`
                - the default is `None`
                    - no error considered
            - `nu_max_sun`
                - float, optional
                - solar value of the frequency at maximum power
                - the default is 3100 microHz
            - `delta_nu_sun`
                - float, optional
                - solar value of the large frequency separation
                - the default is 135.2 microHz
            - `t_eff_sun`
                - float
                - solar effective temperature
                - the default is 5777 K
            - `logg_sun`
                - float, optional
                - log solar surface gravity
                - the default is 4.44 cm/s^2 (cgs system)
            - `zeta_corr`
                - bool, optional
                - correction factor for the large frequency separation
                - the default is `False`
        
        Inferred Attributes
        -------------------
            - `e_radius_seism`
                - np.ndarray
                - error estimate for `radius_seism`
            - `e_logg_seism`
                - np.ndarray
                - error estimate for `logg_seism`
            - `e_luminosity_seism`
                - np.ndarray
                - error estimate for `luminosity_seism`
            - `e_mass_seism`
                - np.ndarray
                - error estimate for `mass_seism`
            - `radius_seism`
                - np.ndarray
                - seismic solution for the radius
            - `logg_seism`
                - np.ndarray
                - seismic solution for the surface gravity
            - `luminosity_seism`
                - np.ndarray
                - seismic solution for the luminosity
            - `mass_seism`
                - np.ndarray
                - seismic solution for the seismic mass
            - `nu_max_hom`
                - np.ndarray
                - homologically scaled nu max
            - `delta_nu_hom`
                - np.ndarray
                - homologically scaled delta nu
            - `t_eff_hom`
                - np.ndarray
                - homologically scaled effective temperature
        
        Methods
        -------
            - `homological_principle()`
            - `get_radius()`
            - `get_mass()`
            - `get_logg()`
            - `get_luminosity()`
            - `results2pandas()`

        Dependencies
        ------------
            - numpy
            - pandas

        Comments
        --------
            - uncertainties are estimated according to gaussian error-propagation
    
    """

    def __init__(self,
        nu_max:np.ndarray,
        delta_nu:np.ndarray,
        t_eff:np.ndarray,
        #errors
        e_nu_max:np.ndarray=None, e_delta_nu:np.ndarray=None, e_t_eff:np.ndarray=None,
        #solar values
        nu_max_sun:float=3100, delta_nu_sun:float=135.2, t_eff_sun:float=5777, logg_sun:float=4.44, 
        #corrections
        zeta_corr:bool=True
        ) -> None:
        
        self.nu_max = np.array(nu_max).reshape(-1)
        self.delta_nu = np.array(delta_nu).reshape(-1)
        self.t_eff = np.array(t_eff).reshape(-1)
        if e_nu_max is None:
            self.e_nu_max = np.array([np.nan]*len(self.nu_max))
        else:
            self.e_nu_max = e_nu_max
        if e_delta_nu is None:
            self.e_delta_nu = np.array([np.nan]*len(self.nu_max))
        else:
            self.e_delta_nu = e_delta_nu
        if e_t_eff is None:
            self.e_t_eff = np.array([np.nan]*len(self.nu_max))
        else:
            self.e_t_eff = e_t_eff
        self.nu_max_sun = nu_max_sun
        self.delta_nu_sun = delta_nu_sun
        self.t_eff_sun = t_eff_sun
        self.logg_sun = logg_sun
        self.zeta_corr = zeta_corr

        self.homological_principle

        #init derived parameters with nan
        self.mass_seism            = np.empty_like(self.delta_nu)
        self.mass_seism[:]         = np.nan
        self.e_mass_seism          = np.empty_like(self.delta_nu)
        self.e_mass_seism[:]       = np.nan
        self.radius_seism          = np.empty_like(self.delta_nu)
        self.radius_seism[:]       = np.nan
        self.e_radius_seism        = np.empty_like(self.delta_nu)
        self.e_radius_seism[:]     = np.nan
        self.luminosity_seism      = np.empty_like(self.delta_nu)
        self.luminosity_seism[:]   = np.nan
        self.e_luminosity_seism    = np.empty_like(self.delta_nu)
        self.e_luminosity_seism[:] = np.nan
        self.logg_seism            = np.empty_like(self.delta_nu)
        self.logg_seism[:]         = np.nan
        self.e_logg_seism          = np.empty_like(self.delta_nu)
        self.e_logg_seism[:]       = np.nan
        return

    def __repr__(self) -> str:
        return (
            f"ScalingRelations(\n"
            f'    nu_max={repr(self.nu_max)}, \n'
            f'    delta_nu={repr(self.delta_nu)}, \n'
            f'    t_eff={repr(self.t_eff)}, \n'
            f'    e_nu_max={repr(self.e_nu_max)}, e_delta_nu={repr(self.e_delta_nu)}, e_t_eff={repr(self.e_t_eff)}, \n'
            f'    nu_max_sun={repr(self.nu_max_sun)}, delta_nu_sun={repr(self.delta_nu_sun)}, t_eff_sun={repr(self.t_eff_sun)}, logg_sun={repr(self.t_eff_sun)}, \n'
            f'    zeta_corr={repr(self.zeta_corr)}, \n'
            f")"
        )

    @property
    def homological_principle(self) -> None:
        """
            - method to execute the homological principle
                - i.e. relate input values to solar values

            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """


        #correction
        if self.zeta_corr == True:
            n_max = self.nu_max/self.delta_nu
            zeta = 0.57/n_max

            n_max_sun = self.nu_max_sun/self.delta_nu    #actually self.delta_nu_solar?
            zeta_solar = 0.57/n_max_sun 
            
            zeta_bool = (n_max < 15)
            zeta[(zeta_bool == True)] = 0.038

            delta_nu = self.delta_nu*(1+zeta)
            #solar_delta_nu *= (1+zeta_solar)

        else:
            delta_nu = self.delta_nu

        #actual homological relations
        self.nu_max_hom = self.nu_max/self.nu_max_sun
        self.delta_nu_hom = delta_nu/self.delta_nu_sun
        self.t_eff_hom = self.t_eff/self.t_eff_sun

        return

    def get_radius(self) -> np.ndarray:
        """
            - method to calculate asteroseismic radii based on the asteroseismic scaling relations
            
            Parameters
            ----------
 
            Raises
            ------

            Returns
            -------
                - `self.radius_seism`
                    - np.ndarray
                    - seismic radii of the input parameters
                - `self.e_radius_seism`
                    - error estimate corresponding to radius_seism
            Comments
            --------            
                - result in Solar Radii

    """

        self.radius_seism = self.nu_max_hom**1 * self.delta_nu_hom**(-2) * self.t_eff_hom**(1/2)

        #uncertainty estimation
        self.e_radius_seism = self.radius_seism * np.sqrt((1*self.e_nu_max/self.nu_max)**2 + (2*self.e_delta_nu/self.delta_nu)**2 + (1/2*self.e_t_eff/self.t_eff)**2)

        return self.radius_seism, self.e_radius_seism

    def get_mass(self) -> np.ndarray:
        """
            - method to calculate asteroseismic masses based on the asteroseismic scaling relations
            
            Parameters
            ----------
 
            Raises
            ------

            Returns
            -------
                - `self.mass_seism`
                    - np.ndarray
                    - seismic masses of the input parameters
                - `self.e_mass_seism`
                    - error estimate corresponding to mass_seism

            Comments
            --------
                - result in Solar Masses

    """

        #actual scaling relation
        self.mass_seism = self.nu_max_hom**3 * self.delta_nu_hom**(-4) * self.t_eff_hom**(3/2)
        
        #uncertainty estimation
        self.e_mass_seism = self.mass_seism * np.sqrt((3*self.e_nu_max/self.nu_max)**2 + (4*self.e_delta_nu/self.delta_nu)**2 + (3/2*self.e_t_eff/self.t_eff)**2)
    
        return self.mass_seism, self.e_mass_seism

    def get_logg(self) -> np.ndarray:
        """
            - method to get the (logarithmic) surface gravity

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `self.logg_seism`
                    - np.ndarray
                    - seismic estimate for the surface gravity of the input parameters
                - `self.e_logg_seism`
                    - error estimate corresponding to logg_seism

            Comments
            -------- 
                - result w.r.t. solar logg
        """
        
        #log(surface gravity)
        self.logg_seism = self.nu_max_hom**1 * self.t_eff_hom**(1/2)

        #uncertainty estimation
        self.e_logg_seism = np.sqrt((1*self.e_nu_max/self.nu_max)**2 + (1/2*self.e_t_eff/self.t_eff)**2)

        return self.logg_seism, self.e_logg_seism

    def get_luminosity(self) -> np.ndarray:
        """
            - method to get the luminosity from seismic scaling relations
                - equation derived by inserting the radius scaling relation into L = 4*pi*r**2*sigma*T_eff**4
                    - i.e. L/L_sun = (R/R_sun)**2 * (T/T_sun)**4

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `self.luminosity_seism`
                    - np.ndarray
                    - seismic estimate for the luminosity of the input parameters
                - `self.e_luminosity_seism`
                    - error estimate corresponding to luminosity_seism

            Comments
            --------
                - result in Solar Luminosities
                
        """

        #check if mass has been calculated already
        if 'mass_seism' not in dir(self):
            #calculate seismic mass if not calculated yet
            _ = self.get_mass()


        #luminosity
        self.luminosity_seism = self.nu_max_hom**(2)*self.delta_nu_hom**(-4)*self.t_eff_hom**5


        #uncertainty estimation
        self.e_luminosity_seism = self.luminosity_seism * np.sqrt((2*self.e_nu_max/self.nu_max)**2 + (4*self.e_delta_nu/self.delta_nu)**2 + (5*self.e_t_eff/self.t_eff)**2)


        return self.luminosity_seism, self.e_luminosity_seism
        
    def results2pandas(self) -> pd.DataFrame:
        """
            - method to convert the calculated results into a pandas DataFrame

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `df`
                    - pd.DataFrame
                    - dataframe containing all the calculated results

            Comments
            --------
        """

        data=np.array([
            self.nu_max, self.e_nu_max,
            self.delta_nu, self.e_delta_nu,
            self.t_eff, self.e_t_eff,
            np.array([self.zeta_corr]*len(self.nu_max)),
            self.mass_seism, self.e_mass_seism,
            self.radius_seism, self.e_radius_seism,
            self.luminosity_seism, self.e_luminosity_seism,
            self.logg_seism, self.e_logg_seism,
        ]).T

        df = pd.DataFrame(
            columns=[
                'nu_max', 'e_nu_max',
                'delta_nu', 'e_delta_nu',
                't_eff', 'e_t_eff',
                'zeta_corr',
                'mass_seism', 'e_mass_seism',
                'radius_seism', 'e_radius_seism',
                'luminosity_seism', 'e_luminosity_seism',
                'logg_seism', 'e_logg_seism',
            ],
            data=data
        )

        return df
