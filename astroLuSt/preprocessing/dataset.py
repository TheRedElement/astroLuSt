
#%%imports
import numpy as np
from typing import Union, Tuple, Callable, List, Literal

#%%definitions
def data_split(
    arrays:List[np.ndarray],
    split_fractions:np.ndarray,
    shuffle:bool=True,
    random_state:int=None,
    verbose:int=0,
    ) -> List:
    """
        - function to split a dataset into several partitions
            - i.e., train-, validation-, test-set
        
        Parameters
        ----------
            - `arrays`
                - list
                - contains np.ndarrays
                    - have to have the same length
                    - will get split according to specification
            - split_fractions
                - list
                - fraction of the total dataset each partition shall have
                - all entries have to be smaller than 1
                - the total (`np.sum(split_fractions)`) has to be smaller than 1
                - the remainder (`1 - np.sum(split_fractions)`) will get assigned to the last created partition
            - `train_size`
                - float
                - has to be between 0 and 1
                - fraction of the dataset to use as trainining data
                - the default is 0.6 (60%)
            - `validation_size`
                - float, optional
                - has to be between 0 and 1
                - fraction of the dataset to use as validation data
                - the default is `None`
                    - will be set to 0
                    - will infer `validation_size` based on `train_size`
            - `test_size`
                - float, optional
                - has to be between 0 and 1
                - fraction of the dataset to use as training data
                - the default is `None`
                    - will be set to 0
                    - will infer `test_size` based on `train_size`
            - `shuffle`
                - bool, optional
                - whether to randomly shuffle the data before applying the split
                - the default is True
            - `random_state`
                - int, optional
                - random seed to use for shuffling the data
                - the default is `None`
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0
        
        Raises
        ------
            - `ValueError`
                - if the total of `split_fractions` (`np.sum(split_fractions)`) is greater or equal to 1
                - if any entry of `split_fractions`  is greater or equal to 1
                - if at least one of the entries of `arrays` has a different length than the rest

        Returns
        -------
            - `splits`
                - list
                    - has length of `len(arrays)*(len(split_fractions)+1)`
                - contains splits of input data with elements according to `split_fractions`

        Dependencies
        ------------
            - numpy
        
        Comments
        --------
    """

    #convert to correct types
    split_fractions = np.array(split_fractions)

    #check shapes
    lengths = [len(a) for a in arrays]
    if not np.all(np.isclose(lengths, lengths)):
        raise ValueError('all entries of `arrays` have to have the same length!')
    # #check feasibility
    if np.sum(split_fractions) >= 1:
        raise ValueError('`np.sum(split_fractions)` has to be less than 1!')
    if np.any(split_fractions >= 1):
        raise ValueError('all entries of `split_fractions` have to be less than 1!')


    

    #generate array of (random) indices
    rand_idxs = np.arange(len(arrays[0]))
    if shuffle:
        #locally set random seed
        cur_state = np.random.get_state()
        np.random.seed(random_state)
        
        #shuffle indices
        np.random.shuffle(rand_idxs)

        #reset random seed
        np.random.set_state(cur_state)

    #partition random indices according to specified fractions
    split_idxs = np.zeros(len(split_fractions)+1, dtype=int) #one longer than `split_fractions` to have an initialization index (0)
    for idx, sf in enumerate(split_fractions):
        split_idxs[idx+1] = split_idxs[idx] + int(sf*len(rand_idxs))
    
    #remove first (pseudo-)entry of split indices
    split_idxs = split_idxs[1:]

    #generate splits of `arrays`
    splits = []
    for a in arrays:
        splits += np.split(a[rand_idxs], indices_or_sections=split_idxs, axis=0)

    return splits

