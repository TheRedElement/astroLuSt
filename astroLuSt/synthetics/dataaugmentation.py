#TODO: - Add obscuring of datapoints (i.e. setting them to some value - min, max, ...)
#TODO: - cropping lc (i.e. only showing part if it, add interpolation_order)
#TODO: - Add flip along x
#TODO: - Add flip along y
#TODO: - Add apply_n (how many transformations to maximally apply to generate any new sample)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Union

from astroLuSt.preprocessing.scaling import AxisScaler

#%% definitions


class AugmentAxis: 
    """
        - class to execute data augmentation on 1D input samples
        
        Attributes
        ----------

            - `shift`
                - tuple, int, optional
                - shift to apply to the input `x` along `axis`
                - will be passed to `np.roll`
                - the default is `None`
                    - will fall be set to 0
                    - no shift
            - `flip`
                - bool, optional
                - whether to apply flipping to the curve or not
                - the default is True
            - `npoints`
                - int, tuple, str, optional
                - number of datapoints to obscure for the chosen axis
                - if an int is passed
                    - will obscure those many points
                - if a tuple is passed
                    - will be interpreted as `low` and `high` from `np.random.randint()` and randomly generate a number
                - the default is `None`
                    - will be set to 0
                    - will not obscure anything
            - `neighbors`
                - bool, optinonal
                - whether the datapoints to obscure shall be neighbors or randomly chosen upon all elements along `axis`
                - if True
                    - will obscure `npoints` neighboring elements along `axis`
                - if False
                    - will randomly choose `npoints` elements along `axis`
                - the default is False
            - `fill_value_obscure`
                - int, str, optional
                - value to use inplace of the obscured entries
                - if an int is passed, this value gets inserted for all elements
                - the following strings are currently allowed
                    - `'random'`
                        - generate `npoints` random datapoints within `fill_value_range`
                - the default is `None`
                    - will be set to 0
            - `fill_value_range`
                - tuple, optional
                - boundaries to define from which range to uniformly sample fill-values
                - will generate `npoints` random fill-values for the requested axis
                - only relevant is `fill_value == 'random'`
                - the default is None
                    - will be set to `(0,1)`
            - `cutout_start`
                - int, tuple, optional
                - index of where to start the cutout region along `axis`
                - if an int
                    - interpreted as the starting index
                -  if a tuple
                    - interpreted as `low` and `high` parameters in `np.random.randint()`
                    - will generate a random integer used as the starting point
                - the default is `None`
                    - defaults to 0
            - `cutout_size`
                - int, tuple, optional
                - length of the cutout region along `axis`
                - if an int
                    - interpreted as the length
                -  if a tuple
                    - interpreted as `low` and `high` parameters in `np.random.randint()`
                    - will generate a random integer used as the length of the cutout
                - the default is `None`
                    - defaults to 0
                        - no transformation applied
            - `interpkind`
                - int, str, optional
                - interpolation method to use
                - parameter of `scipy.interpolate.interp1d()`
                - the default is `None`
                    - defaults to `'linear'`
            - `fill_value_crop`
                - int, tuple, str, optional
                - value used to fill regions where extrapolation is necessary
                - parameter of `scipy.interpolate.interp1d()`
                - if an int
                    - this value will be used for datapoints out of interpolation range
                - if a tuple
                    - the first element will be used for datapoints out of the lower bound of the interpolation range
                    - the second element will be used for datapoints out of the upper bound of the interpolation range
                - supported strings
                    - `'extrapolate'`
                        - extrapolated to infer out-of-bounds values
                - the default is `None`
                    - defaults to `'extrapolate'`   
                - `noise_mag`
                    - float, tuple
                    - magnitude of the noise to apply
                    - if float
                        - will be interpreted as the magnitude
                    - if tuple
                        - interpreted as `low` and `high` parameters in `np.random.uniform()`
                        - will generate a random float to use as the noise_mag
                    - the default is `None`
                        - will default to 0
                        - no noise applied                    
                                

        
            - n_newsamples
                - int, optinal
                - how many new (modified) samples to generate
            - feature_shift
                - np.ndarray, optional
                - defines the range the features will be randomly shifted by
                    - i.e. for feature_shift=[1,6] the features will be shifted by some integer in 1 to 6 indeces
                - contains 2 entries
                    - minimum offset
                    - maximum offset
                - the default is [np.nan, np.nan]
                    - will automatically set the bounds to 0, and number of fetures in the input matrix
            - min_scale
                - np.array, optional
                 - contains 2 entries
                    - lower bound
                    - upper bound
               - defines the range of what the minimum will be scaled to
                    - i.e. for min_scale=[0.9,1] the minimum of the features will be scaled to some float in the range 0.9 to 1
                - the default is [0.9, 1]
            - max_scale
                - np.array, optional
                - contains 2 entries
                    - lower bound
                    - upper bound
               - defines the range of what the maximum will be scaled to
                    - i.e. for min_scale=[1,1.1] the maximum of the features will be scaled to some float in the range 1 to 1.1
                - the default is [1, 1.1]
            - noise_mag
                - np.ndarray, None, optional
                - contains 2 entries
                    - lower bound
                    - upper bound
                - defines the range in which a random number is chose to control the magnitude with which noise gets added
                    - i.e. for noise_mag=[0.001,0.01] a random float will be chosen and the zero-mean unit variance gaussian noise added to the input data will be multiplied by this generated value
                - the default is None
                    - will not apply the noise at all (i.e. multiply the generated noise with 0)
                

        Methods
        -------
            - `shift_features()`
            - `flip_axis()`
            - `obscure_observations()`
            - `crop()`
            - `add_noise()`
            - `rescale()`

            - generate_random_parameters()
            - shift_indices()
            - add_noise()
            - fit_transform()
        
        Dependencies
        ------------
            - numpy
            - matplotlib
            - scipy
            - typing

        Comments
        --------
    """

    def __init__(self,
        nsamples:int=1, sample_weights:list=None,
        shift:Union[tuple,int]=None,
        flip:bool=False,
        npoints:Union[int,tuple]=None, neighbors:bool=False,
        fill_value_obscure:Union[float,str]=None, fill_value_range:tuple=None,
        cutout_start:Union[int,tuple]=None, cutout_size:Union[int,tuple]=None,
        interpkind:Union[str,int]=None, fill_value_crop:Union[int,tuple,str]=None,
        noise_mag:Union[float,tuple]=None,


        min_scale:np.ndarray=None, max_scale:np.ndarray=None,
        n_transformations:int=0,
        axis:tuple=None
        ):
        
        self.nsamples = nsamples
        self.sample_weights = sample_weights

        if shift is None:               self.shift = 0
        else:                           self.shift = shift

        self.flip = flip
        
        if npoints is None: self.npoints = 0
        else:               self.npoints = npoints
        self.neighbors = neighbors
        if fill_value_obscure is None:  self.fill_value_obscure = 0
        else:                           self.fill_value_obscure = fill_value_obscure
        if fill_value_range is None:    self.fill_value_range = (0,1)
        else:                           self.fill_value_range = fill_value_range
        
        if cutout_start is None:        self.cutout_start = 0
        else:                           self.cutout_start = cutout_start
        if cutout_size is None:         self.cutout_size = 0
        else:                           self.cutout_size = cutout_size
        if interpkind is None:          self.interpkind = 'linear'
        else:                           self.interpkind = interpkind
        if fill_value_crop is None:     self.fill_value_crop = 'extrapolate'
        else:                           self.fill_value_crop = fill_value_crop
        
        if noise_mag is None:           self.noise_mag = 0
        else:                           self.noise_mag = noise_mag

        if axis is None:                self.axis = 0
        else:                           self.axis = axis
        # self.min_scale = min_scale
        # self.max_scale = max_scale
        # self.noise_mag = noise_mag


        return


    #helper methods
    def generate_random_parameters(self,
        orig,
        ):
        '''
            - function to generate random-variables needed in fit_transform

            Parameters
            ----------
                - orig
                    - np.array
                    - original, unperturbed input array
            
            Returns
            -------
                - shift
                    - int
                    - random offset in indices
                - scale_min
                    - float
                    - minimum to scale the input to
                - scale_max
                    - float
                    - maximum to scale the input to

        '''
        #generate random shift
        if np.isnan(self.feature_shift[0]):
            min_shift = 1
        else: 
            min_shift = self.feature_shift[0]
        if np.isnan(self.feature_shift[1]):
            max_shift = orig.shape[0]
        else:
            max_shift = self.feature_shift[1]
        shift = np.random.randint(min_shift, max_shift)
        
        #generate scale
        if self.min_scale is not None:
            scale_min = np.random.uniform(self.min_scale[0], self.min_scale[1])
        else:
            scale_min = np.nanmin(orig)
        if self.max_scale is not None:
            scale_max = np.random.uniform(self.max_scale[0], self.max_scale[1])
        else:
            scale_max = np.nanmin(orig)
        
        return shift, scale_min, scale_max



    def shift_features(self,
        x:np.ndarray,
        shift:Union[tuple,int]=None,
        axis:Union[int,tuple]=None,
        ) -> np.ndarray:
        """
            - method to apply a shift along `axis` to one sample `x` in `X`

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the array to be shifted
                - `shift`
                    - tuple, int, optional
                    - shift to apply to the input `x` along `axis`
                    - will be passed to `np.roll`
                    - overrides `self.shift`
                    - the default is `None`
                        - will fall back to `self.shift`
                - `axis`
                    - int, tuple, optional
                    - axis onto which to apply `shift`
                    - will be passed to `np.roll`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`

            Raises
            ------

            Returns
            -------
                - `x_new`
                    - np.ndarray
                    - the input `x` shifted according to the specifications

            Comments
            --------
        """
        
        #default parameters
        if axis is None:
            axis = self.axis
        if shift is None:
            shift = self.shift
        
        #initialize correctly
        if isinstance(shift, int):
            shift = shift
        else:
            shift = np.random.randint(np.nanmin(shift), np.nanmax(shift))

        #apply shift        
        x_new = np.roll(x, shift=shift, axis=axis)

        return x_new

    def flip_axis(self,
        x:np.ndarray,
        axis:Union[int,tuple]=None,
        ) -> np.ndarray:
        """
            - method to flip `x` along a given axis

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the array to be flipped
                - `axis`
                    - int, tuple, optional
                    - axis onto which to apply `shift`
                    - will be passed to `np.roll`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`

            Raises
            ------

            Returns
            -------
                - `x_new`
                    - np.ndarray
                    - the input `x` flipped according to the specifications

            Comments
            --------            
        """

        #default parameters
        if axis is None:
            axis = self.axis

        x_new = np.flip(x, axis=axis)

        return x_new

    def obscure_observations(self,
        x:np.ndarray,
        npoints:Union[int,tuple]=None,
        neighbors:bool=None,
        fill_value_obscure:Union[int,str]=None,
        fill_value_range:tuple=None,
        axis:Union[tuple,int]=None,
        ) -> np.ndarray:
        """
            - method to obscure random datapoints in `x` along a given axis

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the array to be obscured
                - `npoints`
                    - int, tuple, str, optional
                    - number of datapoints to obscure for the chosen axis
                    - if an int is passed
                        - will obscure those many points
                    - if a tuple is passed
                        - will be interpreted as `low` and `high` from `np.random.randint()` and randomly generate a number
                    - the default is `None`
                        - will fall back to `self.npoints`
                - `neighbors`
                    - bool, optinonal
                    - whether the datapoints to obscure shall be neighbors or randomly chosen upon all elements along `axis`
                    - if True
                        - will obscure `npoints` neighboring elements along `axis`
                    - if False
                        - will randomly choose `npoints` elements along `axis`
                    - overrides `self.neighbors`
                    - the default is `None`
                        - will fall back to `self.neighbors`
                - `fill_value_obscure`
                    - int, str, optional
                    - value to use inplace of the obscured entries
                    - if an int is passed, this value gets inserted for all elements
                    - the following strings are currently allowed
                        - `'random'`
                            - generate `npoints` random datapoints within `fill_value_range`
                    - overrides `self.fill_value_obscure`
                    - the default is `None`
                        - will fall back to `self.fill_value_obscure`
                - `fill_value_range`
                    - tuple, optional
                    - boundaries to define from which range to uniformly sample fill-values
                    - will generate `npoints` random fill-values for the requested axis
                    - only relevant is `fill_value == 'random'`
                    - overrides `self.fill_value_range`
                    - the default is `None`
                        - will fall back to `self.fill_value_range`
                - `axis`
                    - int, tuple, optional
                    - axis onto which to apply `shift`
                    - will be passed to `np.roll`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`

            Raises
            ------

            Returns
            -------
                - `x_new`
                    - np.ndarray
                    - the input `x` with elements obscured according to the specifications

            Comments
            --------            
        """        

        #default parameters
        if npoints is None:
            npoints = self.npoints
        if neighbors is None:
            neighbors = self.neighbors
        if fill_value_obscure is None:
            fill_value_obscure = self.fill_value_obscure
        if fill_value_range is None:
            fill_value_range = self.fill_value_range
        if axis is None:
            axis = self.axis

        #initialize correctly
        if isinstance(axis,int):
            axis = (axis,)
        if not isinstance(npoints,int):
            npoints = np.random.randint(low=np.nanmin(npoints), high=np.nanmax(npoints))

        #obscure entries
        x_new = x.copy()
        for ax in axis:
            
            #get correct size for indices and fill_values to generate
            size = np.ones(len(x_new.shape), dtype=int)
            size[ax] = npoints  #5 indices at ax

            #generate random fill_values if requested
            if fill_value_obscure == 'random':
                #generate size for randomly generated fill-values (f_value)
                f_value_size = np.array(x_new.shape)
                f_value_size[ax] = npoints

                #generate random fill-values
                f_value = np.random.uniform(low=np.nanmin(fill_value_range), high=np.nanmax(fill_value_range), size=f_value_size)
            else:
                f_value = fill_value_obscure
            
            #generate random indices to replace
            ##npoints consecutive indices
            if neighbors:
                if x.shape[ax] <= npoints:
                    raise ValueError(
                        f'`npoints` has to be greater than `x.shape` at the requested axis! '
                        f'For your parameters I got `x.shape[{ax}]={x.shape[ax]}` <= `npoints={npoints}`!'
                    )

                startidx = np.random.randint(low=0, high=x.shape[ax]-npoints, size=None)
                ls = np.arange(startidx,startidx+npoints,1, dtype=int).reshape(size)
                idxs = np.zeros(size, dtype=int) + ls

            ##completely random indices
            else:
                idxs = np.random.randint(low=0, high=x.shape[ax], size=size)
            
            #replace values in x_new with f_value (inplace operation)
            np.put_along_axis(x_new, indices=idxs, values=f_value, axis=ax)


        return x_new
    
    def crop(self,
        x:np.ndarray,
        cutout_start:Union[int,tuple]=None, cutout_size:Union[int,tuple]=None,
        interpkind:Union[str,int]=None, fill_value_crop:Union[int,tuple,str]=None,
        axis:Union[int,tuple]=None,
        ) -> np.ndarray:
        """
            - method to crop out a random subset of `x` along a given axis

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the array to be obscured
                - `cutout_start`
                    - int, tuple, optional
                    - index of where to start the cutout region along `axis`
                    - if an int
                        - interpreted as the starting index
                    -  if a tuple
                        - interpreted as `low` and `high` parameters in `np.random.randint()`
                        - will generate a random integer used as the starting point
                    - overrides `self.cutout_start`
                    - the default is `None`
                        - falls back to `self.cutout_start`
                - `cutout_size`
                    - int, tuple, optional
                    - length of the cutout region along `axis`
                    - if an int
                        - interpreted as the length
                    -  if a tuple
                        - interpreted as `low` and `high` parameters in `np.random.randint()`
                        - will generate a random integer used as the length of the cutout
                    - overrides `self.cutout_size`
                    - the default is `None`
                        - falls back to `self.cutout_size`
                - `interpkind`
                    - int, str, optional
                    - interpolation method to use
                    - parameter of `scipy.interpolate.interp1d()`
                    - overrides `self.interpkind`
                    - the default is `None`
                        - falls back to `self.interpkind`
                - `fill_value_crop`
                    - int, tuple, str, optional
                    - value used to fill regions where extrapolation is necessary
                    - parameter of `scipy.interpolate.interp1d()`
                    - if an int
                        - this value will be used for datapoints out of interpolation range
                    - if a tuple
                        - the first element will be used for datapoints out of the lower bound of the interpolation range
                        - the second element will be used for datapoints out of the upper bound of the interpolation range
                    - supported strings
                        - `'extrapolate'`
                            - extrapolated to infer out-of-bounds values
                    - overrides `self.fill_value_crop`
                    - the default is `None`
                        - falls back to `self.fill_value_crop`
                - `axis`
                    - int, tuple, optional
                    - axis onto which to apply `shift`
                    - will be passed to `np.roll`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`

            Raises
            ------

            Returns
            -------
                - `x_new`
                    - np.ndarray
                    - the input `x` with regions cutout from the whole array

            Comments
            --------                    
        """

        #default parameters
        if cutout_start is None:
            cutout_start = self.cutout_start
        if cutout_size is None:
            cutout_size = self.cutout_size
        if fill_value_crop is None:
            fill_value_crop = self.fill_value_crop
        if interpkind is None:
            interpkind = self.interpkind
        if axis is None:
            axis = self.axis
        

        #initialize correctly
        if isinstance(axis,int):
            axis = (axis,) 

        #initialize x_new
        x_new = x.copy()
        for ax in axis:
            #get correct co_size
            if isinstance(cutout_size, int):
                co_size = cutout_size
            else:
                co_size = np.random.randint(np.nanmin(cutout_size), np.nanmax(cutout_size))
            
            #get correct co_start
            if isinstance(cutout_start, int) and cutout_start != -1:
                co_start = cutout_start
            elif cutout_start == -1:
                    co_start = np.random.randint(0, x.shape[ax]-co_size)
            else:
                co_start = np.random.randint(np.nanmin(cutout_start), np.nanmax(cutout_start))

            #apply cut_out (only if the cutout_size is greater than 1, otherwise just return the input-array)
            if co_size > 0:
                xp = np.arange(0,x.shape[ax],1)
                x_interp = interp1d(xp, x_new, kind=interpkind, axis=ax, bounds_error=False, fill_value=fill_value_crop)
                x_new = x_interp(np.linspace(co_start, co_start+co_size ,x_new.shape[ax]))

        return x_new

    def add_noise(self,
        x:np.ndarray,
        noise_mag:Union[float,tuple]=None,
        axis:Union[int,tuple]=None,
        ) -> np.ndarray:
        """
            - method to add gaussian noise to along specified axis

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the array to be obscured
                - `noise_mag`
                    - float, tuple
                    - magnitude of the noise to apply
                    - if float
                        - will be interpreted as the magnitude
                    - if tuple
                        - interpreted as `low` and `high` parameters in `np.random.uniform()`
                        - will generate a random float to use as the noise_mag
                    - overrides `self.noise_mag`
                    - the default is `None`
                        - will fall back to `self.noise_mag`
                - `axis`
                    - int, tuple, optional
                    - axis onto which to apply `shift`
                    - will be passed to `np.roll`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`    
            Raises
            ------

            Returns
            -------
                - `x_new`
                    - np.ndarray
                    - the input `x` with regions cutout from the whole array

            Comments
            --------

        """

        #default parameters
        if axis is None:
            axis = self.axis
        if noise_mag is None:
            noise_mag = self.noise_mag

        #initialize correctly
        if isinstance(noise_mag, float):
            mag = noise_mag
        else:
            mag = np.random.uniform(low=np.nanmin(noise_mag), high=np.nanmax(noise_mag))
        if isinstance(axis, int):
            axis = (axis,)

        #mask for modifying only entries in x.shape not present in axis
        mask = np.ones(len(x.shape), dtype=bool)
        mask[list(axis)] = False
        
        #set all axis not present in axis to 1 (i.e. no noise on those axes)
        noise_size = np.array(x.shape)
        noise_size[mask] = 1

        #apply noise
        noise = np.random.normal(size=noise_size)
        x_new = x + noise*mag

        return x_new

    def rescale(self,
        x:np.ndarray,
        axis:Union[int,tuple]=None,
        ) -> np.ndarray:

        AS = AxisScaler

        return

    def apply_transform(self,
        X:np.ndarray,
        transform_parameters,
        ):
        
        return
    
    def fit(self,
        X:np.ndarray,
        augment:bool=False,
        seed:int=None
        ):

        return
    
    def flow(self,
        X:np.ndarray, y:np.ndarray=None,
        batch_size:int=32,
        shuffle:bool=True,
        sample_weight:list=None,
        seed:int=None,
        ):

        return
    
    def get_random_transform(self,
        
        ) -> dict:
        
        return

    def random_transform(self,
        X:np.ndarray, 
        seed:int=None,
        ):
        
        return

    '''
    def fit_transform(self,
        X:pd.DataFrame, 
        cols2modify:list,
        y:pd.DataFrame=None,
        return_only_new:bool=False,
        testplot:bool=True,
        ):
        """
           - method to apply the specified augmentation to the input matrix X

           Parameters
           ----------
                - X
                    - pd.DataFrame
                    - input dataframe to be augmented
                - cols2modify
                    - list, optional
                    - list of columnnames in X to be modified
                    - useful if not all features shall be considered for augmentation
                    - the default is None
                        - will augment all features
                - y
                    - pd.DataFrame, optional
                    - labels (or any other DataFrame of similar length as y) corresponding to X
                    - the rows corresponding to the augmented samples of X will be added to y
                    - the default is None
                        - will not be considered
                - return_only_new
                    - bool, optional
                    - if True, will only return the newly generated samples
                    - if False, will return a DataFrame containing the input data with appended the newly generated samples
                    - the default is False
                - testplot
                    - bool, optional
                    - whethere to produce testplots of the augmented samples
                    - will produce one plot for every sample!
                    - the default is False

            Raises
            ------

            Returns
            -------
                - df_X_augmented
                    - pd.DataFrame
                    - X with the augmented samples as additional rows
                - df_y_augmented
                    - pd.DataFrame, None
                    - y with labels for the augmented rows in X as additional rows
                    - None if y is not specified
                - if return_only_new == True
                    - df_new_X
                        - pd.DataFrame
                        - newly generated samples
                    - df_new_y
                        - pd.DataFrame, None
                        - labels corresponding to df_new_X
                        - None if y is not specified

            
            Comments
            --------

        """

        #apply to all features if not specified otherwise
        if cols2modify is None: cols2modify = X.columns


        #sample columns to modify
        df_new_X = X.sample(n=self.n_newsamples, replace=True)
        if y is not None: df_new_y = y.loc[df_new_X.index]
        else: df_new_y = None

        for (idx, df_xi) in df_new_X.iterrows():
            
            #original, unperturbed sample
            orig = df_xi[cols2modify]

            #generate parameters for perturbation
            shift, scale_min, scale_max = self.generate_random_parameters(orig)
            

            #shift indices of chosen cols2modify
            shifted = self.shift_indices(orig, shift=shift, testplot=False)
            
            #rescale to chosen interval
            scaled = np.interp(shifted, (np.nanmin(shifted), np.nanmax(shifted)), (scale_min, scale_max))

            #add noise
            noisy = self.add_noise(scaled)

            #assign to dataframe at correct location
            df_new_X.loc[idx, cols2modify] = noisy


            #plotting
            if testplot:
                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                ax1.plot(orig.values, '+', label='Original')
                ax1.plot(shifted.values, '^', label=f'shifted by {shift}')
                ax1.plot(scaled,  '.', label=f'scaled to the interval [{np.nanmin(scaled):.2f}, {np.nanmax(scaled):.2f}]')
                ax1.plot(noisy,  'x', label=f'noisy')
                ax1.plot(df_new_X.loc[idx][cols2modify].values)

                ax1.legend()

                plt.show()

        # print(df_new_X)
        # print(df_new_y)
        
        #return only newly generated samples
        if return_only_new:
            return df_new_X, df_new_y

        #return input data with appended newly generated samples
        else:
            df_X_augmented = pd.concat((X, df_new_X))
            if df_new_y is not None: df_y_augmented = pd.concat((y, df_new_y))
            else: df_y_augmented = None

            return df_X_augmented, df_y_augmented

        '''
