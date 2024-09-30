

#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import warnings



#%% definitions


class Augment1D: 
    """
        - class to execute data augmentation on 1D input samples
        
        Attributes
        ----------
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
            - pandas
            - numpy
            - matplotlib

        Comments
        --------
    """

    def __init__(self,
        n_newsamples:int=1,
        feature_shift:np.ndarray=[np.nan, np.nan],
        min_scale:np.ndarray=None, max_scale:np.ndarray=None,
        noise_mag:np.ndarray=None,
        ):
        
        warnings.warn('This class is deprecated. Use `astrLuSt.synthetics.dataaugmentation.AugmentAxis()` instead')
        self.n_newsamples = n_newsamples
        self.feature_shift = feature_shift
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_mag = noise_mag


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



# #%%testing

# import sys
# sys.path.append('../..')
# import custom_modules.load_data as ld

# id_cols = ['oid_vsx', 'tic', 'gaiadr3_id']
# non_flux_feat_cols = ['period', 'variation_amplitude', 'mh_gspphot_gaia', 'teff_gspphot_gaia']


# X, y = ld.load_Xy(
#     '../../data/Xy/X.csv',
#     '../../data/Xy/y.csv',
#     filterrange=[0.5,2.1]
# )

# y['labs'] = np.argmax(y.drop(id_cols+['blazhko'], axis=1, errors='ignore').values, axis=1)
# print(y['labs'].value_counts())


# flux_cols = X.drop(['oid_vsx', 'tic', 'gaiadr3_id', 'period', 'variation_amplitude', 'mh_gspphot_gaia', 'teff_gspphot_gaia'], axis=1).columns

# augment = Augment1D(
#     n_newsamples=2,
#     feature_shift=[np.nan, np.nan],
#     min_scale=[0.9,1], max_scale=[1,1.1],
#     noise_mag=[0.01,0.02],
# )

# df_X_augmented, df_y_augmented = augment.fit_transform(
#     X,
#     cols2modify=flux_cols,
#     y=y,
#     return_only_new=True,
#     testplot=True)

# print(df_X_augmented.shape)
# print(df_y_augmented.shape)
# # print(df_y_augmented)
# print(df_y_augmented['labs'])