#TODO: - Add obscuring of datapoints (i.e. setting them to some value - min, max, ...)
#TODO: - cropping lc (i.e. only showing part if it, add interpolation_order)
#TODO: - Add flip along x
#TODO: - Add flip along y
#TODO: - Add apply_n (how many transformations to maximally apply to generate any new sample)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Union, List, Tuple, Callable

from astroLuSt.preprocessing.scaling import AxisScaler

#%% definitions


class AugmentAxis: 
    """
        - class to execute data augmentation on 1D input samples
        
        Attributes
        ----------
            - `nsamples`
                - int, optional
                - number of new samples to generate
                - the deafult is 1
            - `ntransformations`
                - int, optional
                - how many transformations to apply to get an augmented sample
                - if negative
                    - will use `len(methods) + 1 + ntransformations` transformations
                    - i.e.
                        - will use all available transformations for `ntransformations == -1`
                        - will use all available transformations but one for `ntransformations == -2`
                        - ect.
                - the default is -1
            - `methods`
                - list, optional
                - transformation methods to use for generation of an augmented sample
                - contains names of the methods to use as strings
                - to get a list of all allowed methods call `self.get_transformations()`
                - during the transformation `eval('self.'+method)(x)` will be called
                    - `method` if hereby an entry of `methods`
                - the default is `None`
                    - will use all internally stored methods
            - `transform_order`
                - str, list, optional
                - order to use for applying transformations
                - the following strings are allowed
                    - `random`
                        - will randomly sample `ntransformations` transformations from `methods`
                    - `unchanged`
                        - will use the unchanged input for `methods` in that order
                - if list
                    - will use the list as array indices to select the respective elements in `methods` as it was passed
                - the default is `None`
                    - will be set to `'unchanged'`
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
            - `feature_range_min`
                - int, tuple, optional
                - the lower bound of the interval the rescaled axis shall have
                - if an int
                    - interpreted as the lower bound directly
                - if a tuple
                    - interpreted as `low` and `high` parameters in `np.random.uniform()`
                - the default is `None`
                    - will default to 0                     
            - `feature_range_max`
                - int, tuple, optional
                - the upper bound of the interval the rescaled axis shall have
                - if an int
                    - interpreted as the upper bound directly
                - if a tuple
                    - interpreted as `low` and `high` parameters in `np.random.uniform()`
                - the default is `None`
                    - will default to 1                             
            - `axis`
                - int, tuple, optional
                - axis onto which to apply the transformations
                - the default is `None`
                    - will be set to 0
            - `seed`
                - int, optional
                - seed of the random number generator
                - the default is `None`
            - verbose
                - int, optional
                - verbosity level
                - the default is 0
                

        Methods
        -------
            - `add_noise()`
            - `apply_transform()`
            - `crop()`
            - `flip_axis()`
            - `get_transformations()`
            - `obscure_observations()`
            - `rescale()`
            - `shift_features()`
            

            
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
        nsamples:int=1,
        ntransformations:int=-1, methods:list=None, transform_order:Union[str,List[int]]=None,
        shift:Union[tuple,int]=None,
        flip:bool=False,
        npoints:Union[int,tuple]=None, neighbors:bool=False,
        fill_value_obscure:Union[float,str]=None, fill_value_range:tuple=None,
        cutout_start:Union[int,tuple]=None, cutout_size:Union[int,tuple]=None,
        interpkind:Union[str,int]=None, fill_value_crop:Union[int,tuple,str]=None,
        noise_mag:Union[float,tuple]=None,
        feature_range_min:Union[int,tuple]=None, feature_range_max:Union[int,tuple]=None,
        axis:tuple=None,
        seed:int=None,
        verbose:int=0,
        ):
        
        self.nsamples = nsamples
        self.ntransformations = ntransformations
        if methods is None:             self.methods = self.get_transformations()
        else:                           self.methods = methods
        if transform_order is None:     self.transform_order = 'unchanged'
        else:                           self.transform_order = transform_order

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

        if feature_range_min is None:   self.feature_range_min = 0
        else:                           self.feature_range_min = feature_range_min
        if feature_range_max is None:   self.feature_range_max = 1
        else:                           self.feature_range_max = feature_range_max

        if axis is None:                self.axis = 0
        else:                           self.axis = axis

        self.seed = seed
        self.verbose = verbose


        #instantiate random number generator
        self.rng = np.random.default_rng(seed=seed)

        return

    def __repr__(self) -> str:
        return (
            f'AugmentAxis(\n'
            f'    nsamples={repr(self.nsamples)},\n'
            f'    ntransformations={repr(self.ntransformations)}, methods={repr(self.methods)}, transform_order={repr(self.transform_order)},\n'
            f'    shift={repr(self.shift)},\n'
            f'    flip={repr(self.flip)},\n'
            f'    npoints={repr(self.npoints)}, neighbors={repr(self.neighbors)},\n'
            f'    fill_value_obscure={repr(self.fill_value_obscure)}, fill_value_range={repr(self.fill_value_range)},\n'
            f'    cutout_start={repr(self.cutout_start)}, cutout_size={repr(self.cutout_size)},\n'
            f'    interpkind={repr(self.interpkind)}, fill_value_crop={repr(self.fill_value_crop)},\n'
            f'    noise_mag={repr(self.noise_mag)},\n'
            f'    feature_range_min={repr(self.feature_range_min)}, feature_range_max={repr(self.feature_range_max)},\n'
            f'    axis={repr(self.axis)},\n'
            f'    seed={repr(self.seed)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

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
        shift = self.rng.integers(min_shift, max_shift)
        
        #generate scale
        if self.min_scale is not None:
            scale_min = self.rng.uniform(self.min_scale[0], self.min_scale[1])
        else:
            scale_min = np.nanmin(orig)
        if self.max_scale is not None:
            scale_max = self.rng.uniform(self.max_scale[0], self.max_scale[1])
        else:
            scale_max = np.nanmin(orig)
        
        return shift, scale_min, scale_max


    def get_transformations(self,
        ) -> List[str]:
        """
            - method to obtain all available transformation methods

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - transformations
                    - list
                    - contains the names of the methods used for applying transformations

            Comments
            --------
        """

        transformations = []
        for d in dir(AugmentAxis):
            if d[:2] != '__':
                if callable(eval(f'self.{d}')):
                    transformations.append(eval(f'self.{d}').__name__)

        #methods not used for transformations
        non_transformations = [
            'apply_transform', 'fit',
            'flow', 'generate_random_parameters',
            'get_random_transform', 'get_transformations',
            'random_transform'
        ]
        for ntf in non_transformations:
            try: 
                transformations.remove(ntf)
            except:
                pass

        return transformations

    def shift_features(self,
        x:np.ndarray,
        shift:Union[tuple,int]=None,
        axis:Union[int,tuple]=None,
        **kwargs,
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
                - `**kwargs`
                    - dict, optional
                    - keyword arguments to pass to the function

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
            shift = self.rng.integers(np.nanmin(shift), np.nanmax(shift))

        #apply shift        
        x_new = np.roll(x, shift=shift, axis=axis)

        return x_new

    def flip_axis(self,
        x:np.ndarray,
        axis:Union[int,tuple]=None,
        **kwargs,
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
                    - axis onto which to apply `flip_axis`
                    - will be passed to `np.flip`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`
                - `**kwargs`
                    - dict, optional
                    - keyword arguments to pass to the function

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
        **kwargs,
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
                    - axis onto which to apply `obscure_observations`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`
                - `**kwargs`
                    - dict, optional
                    - keyword arguments to pass to the function
                        
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
            npoints = self.rng.integers(low=np.nanmin(npoints), high=np.nanmax(npoints))

        #obscure entries
        x_new = x.copy()
        for ax in axis:
            
            #get correct size for indices and fill_values to generate
            size = np.ones(len(x_new.shape), dtype=int)
            size[ax] = npoints  #npoints indices at ax

            #generate random fill_values if requested
            if fill_value_obscure == 'random':
                #generate size for randomly generated fill-values (f_value)
                f_value_size = np.array(x_new.shape)
                f_value_size[ax] = npoints

                #generate random fill-values
                f_value = self.rng.uniform(low=np.nanmin(fill_value_range), high=np.nanmax(fill_value_range), size=f_value_size)
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

                startidx = self.rng.integers(low=0, high=x.shape[ax]-npoints, size=None)
                ls = np.arange(startidx,startidx+npoints,1, dtype=int).reshape(size)
                idxs = np.zeros(size, dtype=int) + ls

            ##completely random indices
            else:
                idxs = self.rng.integers(low=0, high=x.shape[ax], size=size)
            
            #replace values in x_new with f_value (inplace operation)
            np.put_along_axis(x_new, indices=idxs, values=f_value, axis=ax)


        return x_new
    
    def crop(self,
        x:np.ndarray,
        cutout_start:Union[int,tuple]=None, cutout_size:Union[int,tuple]=None,
        interpkind:Union[str,int]=None, fill_value_crop:Union[int,tuple,str]=None,
        axis:Union[int,tuple]=None,
        **kwargs,
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
                    - axis onto which to apply `crop`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`
                - `**kwargs`
                    - dict, optional
                    - keyword arguments to pass to the function
                        
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
                co_size = self.rng.integers(np.nanmin(cutout_size), np.nanmax(cutout_size))
            
            #get correct co_start
            if isinstance(cutout_start, int) and cutout_start != -1:
                co_start = cutout_start
            elif cutout_start == -1:
                    co_start = self.rng.integers(0, x.shape[ax]-co_size)
            else:
                co_start = self.rng.integers(np.nanmin(cutout_start), np.nanmax(cutout_start))

            #apply cut_out (only if the cutout_size is greater than 1, otherwise just return the input-array)
            if co_size > 0:
                xp = np.arange(0,x.shape[ax],1)
                x_interp = interp1d(xp, x_new, kind=interpkind, axis=ax, bounds_error=False, fill_value=fill_value_crop)
                x_new = x_interp(np.linspace(co_start, co_start+co_size ,x_new.shape[ax]))

        return x_new

    def rescale(self,
        x:np.ndarray,
        feature_range_min:Union[int,tuple]=None, feature_range_max:Union[int,tuple]=None,
        axis:Union[int,tuple]=None,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to rescale the input into a specified interval along specified axis

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the array to be obscured
                - `feature_range_min`
                    - int, tuple, optional
                    - the lower bound of the interval the rescaled axis shall have
                    - if an int
                        - interpreted as the lower bound directly
                    - if a tuple
                        - interpreted as `low` and `high` parameters in `np.random.uniform()`
                    - overrides `self.feature_range_min`
                    - the default is `None`
                        - falls back to `self.feature_range_min`
                - `feature_range_max`
                    - int, tuple, optional
                    - the upper bound of the interval the rescaled axis shall have
                    - if an int
                        - interpreted as the upper bound directly
                    - if a tuple
                        - interpreted as `low` and `high` parameters in `np.random.uniform()`
                    - overrides `self.feature_range_max`
                    - the default is `None`
                        - falls back to `self.feature_range_max`
                - `axis`
                    - int, tuple, optional
                    - axis onto which to apply `rescale`
                    - will be passed to `astroLuSt.preprossesing.scaling.AxisScaler()`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`    
                - `**kwargs`
                    - dict, optional
                    - keyword arguments to pass to the function
                        
            Raises
            ------

            Returns
            -------
                - `x_new`
                    - np.ndarray
                    - the input `x` with the requested axis rescaled to the requested feature range

            Comments
            --------
        
        
        """

        #default parameteres
        if feature_range_min is None: feature_range_min = self.feature_range_min
        if feature_range_max is None: feature_range_max = self.feature_range_max
        if axis is None: axis = self.axis

        #initialize correctly
        if isinstance(feature_range_min, int):
            fr_min = feature_range_min
        else:
            fr_min = self.rng.uniform(np.nanmin(feature_range_min), np.nanmax(feature_range_min))
        if isinstance(feature_range_max, int):
            fr_max = feature_range_max
        else:
            fr_max = self.rng.uniform(np.nanmin(feature_range_max), np.nanmax(feature_range_max))

        AS = AxisScaler(
            scaler='range_scaler',
            axis=axis,
            scaler_kwargs={'feature_range':(fr_min, fr_max)}
        )
        x_new = AS.fit_transform(x)

        return x_new

    def add_noise(self,
        x:np.ndarray,
        noise_mag:Union[float,tuple]=None,
        axis:Union[int,tuple]=None,
        **kwargs,
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
                    - axis onto which to apply `add_noise`
                    - overrides `self.axis`
                    - the default is `None`
                        - will fall back to `self.axis`    
                - `**kwargs`
                    - dict, optional
                    - keyword arguments to pass to the function
                        
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
            mag = self.rng.uniform(low=np.nanmin(noise_mag), high=np.nanmax(noise_mag))
        if isinstance(axis, int):
            axis = (axis,)

        #mask for modifying only entries in x.shape not present in axis
        mask = np.ones(len(x.shape), dtype=bool)
        mask[list(axis)] = False
        
        #set all axis not present in axis to 1 (i.e. no noise on those axes)
        noise_size = np.array(x.shape)
        noise_size[mask] = 1

        #apply noise
        noise = self.rng.normal(size=noise_size)
        x_new = x + noise*mag

        return x_new

    def apply_transform(self,
        x:np.ndarray,
        transform_parameters:dict=None,
        ntransformations:int=None,
        methods:List[str]=None,
        transform_order:Union[str,List[int]]='random',
        verbose:int=None,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to apply tranformations to one sample (3d array)

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the array to be transformed
                - `transform_parameters`
                    - dict, optional
                    - parameters passed to all methods applied for the transformation
                        - i.e. entries contained within `methods`
                    - the default is `None`
                        - will be set to `{}`
                        - i.e. uses class attributes
                - `ntransformations`
                    - int, optional
                    - how many transformations to apply to the input `x`
                    - if negative
                        - will use `len(methods) + 1 + ntransformations` transformations
                        - i.e.
                            - will use all available transformations for `ntransformations == -1`
                            - will use all available transformations but one for `ntransformations == -2`
                            - ect.
                    - overrides `self.ntransformations`
                    - the default is `None`
                        - will fall back onto `self.ntransformations`
                - `methods`
                    - list, optional
                    - methods to use on the input `x`
                    - contains names of the methods to use as strings
                    - to get a list of all allowed methods call `self.get_transformations()`
                    - during the transformation `eval('self.'+method)(x)` will be called
                        - `method` if hereby an entry of `methods`
                    - overrides `self.methods`
                    - the default is `None`
                        - will fall back to `self.methods`
                - `transform_order`
                    - str, list, optional
                    - order to use for applying transformations
                    - the following strings are allowed
                        - `random`
                            - will randomly sample `ntransformations` transformations from `methods`
                        - `unchanged`
                            - will use the unchanged input for `methods` in that order
                    - if list
                        - will use the list as array indices to select the respective elements in `methods` as it was passed
                    - overrides `self.transform_order`
                    - the default is `None`
                        - will fall back to `self.transform_order`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `**kwargs`
                    - keyword arguments of the method

            Raises
            ------
                - ValueError
                    - if `transform_order` is not valid

            Returns
            -------
                - `x_new`
                    - np.ndarray
                    - transformed version of the input `x`

            Comments
            --------
                - will apply `methods` in the order that is present in the `to_apply`
                    - `to_apply` is a resampled list `ntransformations >= 0` and `methods` otherwise
                    - in case `methods` none, the alphabetically sorted list obtained with `self.get_transformations()` is used
        """

        #default values
        if ntransformations is None:
            ntransformations = self.ntransformations
        if transform_parameters is None:
            transform_parameters = {}
        if methods is None:
            methods = self.methods
        if transform_order is None:
            transform_order = self.transform_order
        if verbose is None:
            verbose = self.verbose

        methods = np.array(methods)
            
        #initialize correctly
        ##use passed methods
        if isinstance(methods, np.ndarray):
            methods_bool = np.isin(methods, self.get_transformations())
            if np.any(~methods_bool):
                not_allowed = methods[~methods_bool]
                methods = methods[methods_bool]

                if verbose > 0:
                    print(
                        f'WARNING(AugmentAxis.apply_transform):\n'
                        f'    {not_allowed} are invalid methods, will therefore be ignored.\n'
                        f'    Allowed are {self.get_transformations()}.'
                    )

        ##use a random selection of passed methods
        if transform_order == 'random':
            #if negative value for ntransformations is passed use that many less transformations
            if ntransformations < 0:
                ntransformations = len(methods) + 1 + ntransformations 
            to_apply = np.random.choice(methods, size=ntransformations, replace=False)
        elif transform_order == 'unchanged':
            if ntransformations < 0:
                ntransformations = len(methods) + 1 + ntransformations
            to_apply = methods[:ntransformations]
        ##use specified order
        elif isinstance(transform_order, (list,np.ndarray)):
            to_apply = methods[transform_order]
        else:
            raise ValueError(
                f'`transform_order` has to be either a list of int or one of {["unchanged", "random"]} but is {transform_order}!'
            )

        #initilaize augmented sample
        x_new = x.copy()

        #apply augmentations
        for tf in to_apply:
            x_new = eval('self.'+tf)(x_new, **transform_parameters)

        #summary
        if verbose > 1:
            print(
                f'INFO(AugmentAxis.apply_transform):\n'
                f'    Applied the following transformations: {to_apply}'
            )

        return x_new
    
    def fit(self,
        X:np.ndarray,
        augment:bool=False,
        random_seed:int=None
        ) -> None:
        """
            - method to fit the generator to some sample data `X`
            - calculates relevant statistics based on the sample data

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - has to be 4d array
                    - input data containing n samples on its first axis
                - `augment`
                    - bool, optional
                    - whether to fit to randomly augmented data
                    - the default is False
                - `random_seed`
                    - int, optional
                    - random seed
                    - overrides `self.random_seed`
                    - the default is `None`
                        - will fall back to `self.random_seed`

            Raises
            ------

            Returns
            -------

            Comments
            --------
                - not necessary as of now
                    - included to follow the tf.keras structure of `tf.keras.preprocessing.image.ImageDataGenerator`
        """


        return
    
    def flow(self,
        X:np.ndarray, y:np.ndarray=None, X_misc:List[np.ndarray]=None,
        sample_weights:list=None,
        nsamples:int=None,
        verbose:int=None,
        apply_transform_kwargs:dict=None,
        ) -> Tuple[np.ndarray,...]:
        """
            - method to generate `nsamples` new samples by augmenting an input `X`

            Parameters
            ----------
                - `X`
                    - np.ndarray
                    - input dataset serving as template for augmentation
                - `y`
                    - np.ndarray
                    - labels corresponding to `X`
                    - the default is `None`
                    - will be ignored and returned as array of `np.nan`
                - `X_misc`
                    - list, optional
                    - list of np.ndarrays of same first dimension as `X`
                    - miscallaneous datasets that get passed to the output without any modifications
                    - will return as many additional entries in the output as entries in `X_misc`
                    - the default is `None`
                        - will be ignored
                - `sample_weights`
                    - list, optional
                    - has to have same lenght as `X`
                    - probabilities for each sample to be drawn for augmentation
                    - the default is `None`
                        - will assume uniform distribution over the input `X`
                - `nsamples`
                    - int, optional
                    - number of new samples to generate
                    - overrides `self.nsamples`
                    - the deafult is `None`
                        - will fall back to `self.nsamples`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `apply_transform_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.apply_transform()`
                    - the default is `None`
                        -  will be initialized with `{}`

            Raises
            ------

            Returns
            -------
                - `X_new`
                    - np.ndarray
                    - augmented samples generated based on `X`
                - `y_new`
                    - np.ndarray
                    - labels corresponding to `X_new`
                - `*X_misc_new`
                    - np.ndarray
                    - any miscallaneous dataset with entries corresponding to `X_new`

            Comments
            --------
        """
        #default parameters
        if nsamples is None:
            nsamples = self.nsamples
        if verbose is None:
            verbose = self.verbose
        if apply_transform_kwargs is None:
            apply_transform_kwargs = {}


        #no misc data passed
        if X_misc is None:
            X_misc = []

        #if no y was provided create one filled with nan
        if y is None:
            y = np.ones(X.shape[0])
            y[:] = np.nan

        #if not sample_weights have been passed assume uniform distribution
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])/len(X)

        #generate new arrays
        ##initialize
        X_new = np.empty((nsamples, *X.shape[1:]))
        y_new = np.empty((nsamples, *y.shape[1:]))
        X_misc_new = [np.empty((nsamples, *X_m.shape[1:])) for X_m in X_misc]

        ##generate nsamples new samples
        if verbose > 2:
            print(
                f'INFO(AugmentAxis.flow):\n'
                f'    Generating {nsamples} new samples...'
            )
        for n in range(nsamples):
            sample_idx = self.rng.choice(np.arange(0, len(X),1), size=None, replace=False, p=sample_weights)
            X_new[n] = self.apply_transform(X[sample_idx], **apply_transform_kwargs)
            y_new[n] = y[sample_idx]
            for idx in range(len(X_misc_new)):
                X_misc_new[idx][n] = X_misc[idx][sample_idx]

        return X_new, y_new, *X_misc_new
    
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
