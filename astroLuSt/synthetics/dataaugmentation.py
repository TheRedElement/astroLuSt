#TODO: - Add obscuring of datapoints (i.e. setting them to some value - min, max, ...)
#TODO: - cropping lc (i.e. only showing part if it, add interpolation_order)
#TODO: - Add flip along x
#TODO: - Add flip along y
#TODO: - Add apply_n (how many transformations to maximally apply to generate any new sample)

#%%
import numpy as np
import matplotlib.pyplot as plt
from typing import Union



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
            - `flip` TODO
                - bool, optional

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
            - `fill_value`
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
            - generate_random_parameters()
            - shift_indices()
            - add_noise()
            - fit_transform()
        
        Dependencies
        ------------
            - numpy
            - matplotlib
            - typing

        Comments
        --------
    """

    def __init__(self,
        nsamples:int=1, sample_weights:list=None,
        shift:Union[tuple,int]=None,
        npoints:Union[int,tuple]=None,
        neighbors:bool=False,
        fill_value:Union[float,str]=None,
        fill_value_range:tuple=None,

        min_scale:np.ndarray=None, max_scale:np.ndarray=None,
        noise_mag:np.ndarray=None,
        n_transformations:int=0,
        axis:tuple=None
        ):
        
        self.nsamples = nsamples
        self.sample_weights = sample_weights
        if shift is None:
            self.shift = 0
        else:
            self.shift = shift

        if npoints is None:
            self.npoints = 0
        else:
            self.npoints = npoints
        self.neighbors = neighbors
        if fill_value is None:
            self.fill_value = 0
        else:
            self.fill_value = fill_value
        if fill_value_range is None:
            self.fill_value_range = (0,1)
        else:
            self.fill_value_range = fill_value_range


        if axis is None:
            self.axis = 0
        else:
            self.axis = axis
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


    def shift_indices(self,
        x:np.ndarray, shift:int, testplot=False
        ) -> np.ndarray:
        '''
            - method to shift array entris by 'shift' indices

            Parameters
            ----------
                - x
                    - np.ndarray
                    - input array to be shifted
                - shift
                    - int
                    - amount of indices to shift the array by
                - testplot
                    - bool, optional
                    - whether to produce a plot showing the transformation

            Raises
            ------
            
            Returns
            -------
                - new_x
                    - np.ndarray
                    - array with the shifted entries
            
            Comments
            --------

        '''

        shifted_idxs = np.arange(0, x.shape[0], 1) + shift
        shifted_idxs[(shifted_idxs>x.shape[0]-1)] = np.mod(shifted_idxs[(shifted_idxs>x.shape[0]-1)], x.shape[0])
        # print(shifted_idxs)
        
        new_x = x[shifted_idxs]

        #plotting
        if testplot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.plot(x, '.', label='original')
            ax1.plot(new_x, '.', label=f'shifted by {shift}')

            ax1.legend()

            plt.show()

        return new_x

    def add_noise(self,
        x:np.ndarray
        ):
        '''
            - function to add gaussian noise to an array

            Parameters
            ----------
                - x
                    - np.ndarray
                    - input array
            
            Returns
            -------
                - x_noisy
                    - np.ndarray
                    - the noisy version of x
        '''

        if self.noise_mag is None:
            x_noisy = x
        else:
            mag = np.random.uniform(low=np.nanmin(self.noise_mag), high=np.nanmax(self.noise_mag))
            noise = np.random.normal(size=x.shape)
            x_noisy = x + noise*mag

        return x_noisy   



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
        fill_value:Union[int,str]=None,
        fill_value_range:tuple=None,
        axis:Union[tuple,int]=None,
        ):
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
                - `fill_value`
                    - int, str, optional
                    - value to use inplace of the obscured entries
                    - if an int is passed, this value gets inserted for all elements
                    - the following strings are currently allowed
                        - `'random'`
                            - generate `npoints` random datapoints within `fill_value_range`
                    - overrides `self.fill_value`
                    - the default is `None`
                        - will fall back to `self.fill_value`
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
        if fill_value is None:
            fill_value = self.fill_value
        if fill_value_range is None:
            fill_value_range = self.fill_value_range
        if axis is None:
            axis = self.axis

        #initialize correctly
        if isinstance(axis,int):
            axis = (axis,)
        if not isinstance(npoints,int):
            npoints = np.random.randint(low=np.random.randint(np.nanmin(npoints), np.nanmax(npoints)))

        #obscure entries
        x_new = x.copy()
        for ax in axis:
            
            #get correct size for indices and fill_values to generate
            size = np.ones(len(x_new.shape), dtype=int)
            size[ax] = npoints  #5 indices at ax

            #generate random fill_values if requested
            if fill_value == 'random':
                #generate size for randomly generated fill-values (f_value)
                f_value_size = np.array(x_new.shape)
                f_value_size[ax] = npoints

                #generate random fill-values
                f_value = np.random.uniform(low=np.nanmin(fill_value_range), high=np.nanmax(fill_value_range), size=f_value_size)
            else:
                f_value = fill_value
            
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
        idx_range:tuple=None
        ):

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
