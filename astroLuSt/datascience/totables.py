
#%%imports
import numpy as np
import pandas as pd
import re
from tensorflow.keras import Model
import tensorflow as tf
from typing import Dict, List

#%%definitions
def hypergrid2latex(
    hypergrid:Dict[str,list],
    pd_to_latex_kwargs:dict=None
    ) -> str:
    """
        - function to take a dictionary defining a hyperparameter grid a la sklearn convention and produce a LaTeX table from that

        Parameters
        ----------
            - `hypergrid`
                - `Dict[str,list]`
                - dictionary of hyperparameters to test
                - has to have the parameter-names as keys and the values per parameter as values
            - `pd_to_latex_kwargs`
                - `dict`, optional
                - kwargs to pass to `pd.DataFrame.to_latex()`
                - the default is `None`
                    - will be set to `dict(buf=None, index=False, position='!th', label='tab:YOURLABEL', caption=f'Hyperparameter-Grid', escape=False)`

        Raises
        ------

        Returns
        -------
            - `hypergrid_str`
                - `str`
                - string representation of the generated LaTeX table

        Comments
        --------
    """

    #default values
    if pd_to_latex_kwargs is None: pd_to_latex_kwargs = dict(buf=None, index=False, position='!th', label='tab:YOURLABEL', caption=f'Hyperparameter-Grid', escape=False)
    if 'buf' not in pd_to_latex_kwargs.keys():      pd_to_latex_kwargs['buf']       = None
    if 'index' not in pd_to_latex_kwargs.keys():    pd_to_latex_kwargs['index']     = False
    if 'position' not in pd_to_latex_kwargs.keys(): pd_to_latex_kwargs['position']  = '!th'
    if 'label' not in pd_to_latex_kwargs.keys():    pd_to_latex_kwargs['label']     = 'tab:YOURLABEL'
    if 'label' not in pd_to_latex_kwargs.keys():    pd_to_latex_kwargs['label']     = 'Hyperparameter-Grid'
    if 'caption' not in pd_to_latex_kwargs.keys():  pd_to_latex_kwargs['caption']   = False

    #generate DataFrame
    df_hypergrid = pd.DataFrame({k:[v] for k, v in hypergrid.items()}).T.reset_index()
    
    #Set Column names accordingly
    df_hypergrid.columns = ['Parameter', 'Value']
    
    #Replace list-symbols with curly brackets (set of...)
    df_hypergrid['Value'] = df_hypergrid['Value'].astype(str).replace({r'^\[':'\{', r'\]$':'\}'}, regex=True)
    
    #generate latex representation
    hypergrid_str = df_hypergrid.to_latex(**pd_to_latex_kwargs)
    
    return hypergrid_str

def summary2pandas_keras(
    model:Model,
    layer_attrs:List[str]=None,
    verbose:int=0,
    ) -> pd.DataFrame:
    """
        - function to convert the keras `model.summary()` output into a pandas DataFrame

        Parameters
        ----------
            - `model`
                - `tf.keras.Model`
                - model to extract the summary of
            - `layer_attrs`
                - `List[str]`, optional
                - additional layer attributes to extract for each layer
                - the default is `None`
                    - will be set to `["units"]`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
        
        Raises
        ------
                
        Returns
        -------
            - `df_summary`
                - `pd.DataFrame`
                - extracted model summary
        
        Comments
        --------
    """

    #default parameters
    if layer_attrs is None: layer_attrs = ["units"]
    
    #extract some additional information for each layer
    extraction = dict(
        name=[],
        type=[],
        input_shape=[], output_shape=[],
        activation=[],
        trainable_params=[],
        non_trainable_params=[],
        total_params=[],
    )
    #get basic layer attributes
    for idx, layer in enumerate(model.layers):

        # print(dir(layer))
        extraction["name"].append(layer.name)
        extraction["type"].append(layer.__class__.__name__)
        if hasattr(layer.input, "shape"):   extraction["input_shape"].append(layer.input.shape) 
        else:                               extraction["input_shape"].append("")
        if hasattr(layer.output, "shape"):  extraction["output_shape"].append(layer.output.shape) 
        else:                               extraction["output_shape"].append("")
        if hasattr(layer, "activation"):    extraction["activation"].append(layer.activation.__name__)
        else:                               extraction["activation"].append(None)
        extraction["trainable_params"].append(int(np.sum([tf.size(w) for w in layer.trainable_weights])))
        extraction["non_trainable_params"].append(int(np.sum([tf.size(w) for w in layer.non_trainable_weights])))
        extraction["total_params"].append(layer.count_params())

    #get additional attributes
    for attr in layer_attrs:
        extraction[attr] = []
        for layer in model.layers:
            if hasattr(layer, attr):    extraction[attr].append(eval(f"layer.{attr}"))
            else:                       extraction[attr].append(None)

    df_summary = pd.DataFrame(extraction)
    
    return df_summary
