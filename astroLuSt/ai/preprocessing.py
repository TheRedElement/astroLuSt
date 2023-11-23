
#%%imports
import numpy as np
from typing import List

#%%definitions
def train_val_test_split(
    arrays:List[np.ndarray],
    train_size:float=0.6,
    validation_size:float=None,
    test_size:float=None,
    shuffle:bool=True,
    random_state:int=None,
    verbose:int=0,
    ) -> List:
    """
        - function to split a dataset into train, validation, and test partitions
        
        Parameters
        ----------
            - `arrays`
                - list
                - contains np.ndarrays
                    - have to have the same length
                    - will get split according to specification
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
                - if `train_size+validation_size+test_size > 1`
                - if at least one of the entries of `arrays` has a different length than the rest

        Returns
        -------
            - `splits`
                - list
                    - has length of `len(arrays)*nsplits`
                        - `nsplits` is the number of requested splits i.e.,
                            - 1 if `train_size == 1`, `validation_size == 0`, `test_size == 0`
                            - 2 if `train_size < 1`, `validation_size == 1 - train_size`, `test_size == 0`
                            - 2 if `train_size < 1`, `validation_size == 0`, `test_size == 1 - train_size`
                            - 3 if `train_size < 1`, `validation_size < 1`, `test_size < 1`
                - contains train-, validation-, test-splits of inputs (`arrays`)

        Dependencies
        ------------
            - numpy
            - typing
        
        Comments
        --------
    """


    #default values
    if test_size is None:       test_size       = 0
    if validation_size is None: validation_size = 0
    
    #check feasibility
    if train_size+validation_size+test_size > 1:
        raise ValueError('`train_size+validation_size+test_size` has to be less or equal to 1!')
    #check shapes
    lengths = [len(a) for a in arrays]
    if not np.all(np.isclose(lengths, lengths)):
        raise ValueError('all entries of `arrays` have to have the same length!')

    

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
    idx_train = int(train_size*len(rand_idxs))
    idx_val   = idx_train + int(validation_size*len(rand_idxs))
    idx_test  = idx_val + int(test_size*len(rand_idxs))

    #get unique indices for partitioning (remove if `*_size==0`)
    idxs = sorted(set((idx_train,idx_val,idx_test)))

    #generate splits of `arrays``
    splits = []
    for a in arrays:
        splits += np.split(a, indices_or_sections=idxs, axis=0)

    return splits


