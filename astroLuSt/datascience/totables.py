
#%%imports
import pandas as pd
import re
from tensorflow.keras import Model
from typing import Dict

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
                - dict, optional
                - kwargs to pass to `pd.DataFrame.to_latex()`
                - the default is `None`
                    - will be set to `dict(buf=None, index=False, position='!th', label='tab:YOURLABEL', caption=f'Hyperparameter-Grid', escape=False)`

        Raises
        ------

        Returns
        -------
            - `hypergrid_str`
                - str
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
    model:Model, width:int=250,
    latex_to_file:str=False, write_mode:str="a",
    save:str=False,
    verbose:int=0,
    ) -> pd.DataFrame:
    """
        - function to convert the keras `model.summary()` output into a pandas DataFrame

        Parameters
        ----------
            - `model`
                - `tf.keras.Model`
                - model to extract the summary of
            - `width`
                - `int`, optional
                - width used in the `model.summary()` function
                - the default is 250
            - `latex_to_file`
                - `str`, `bool`, optional
                - filename of the file to export the latex-representation of the table to
                - the default is `False`
                    - will not export the representation
            - `write_mode`
                - `str`, optional
                - mode to use for writing to `latex_to_file`
                - the default is `a`
                    - will append to the file
            - `save`
                - `str`, optional
                - path to location of where to save the model summary
                - the default is `False`
                    - will not be saved
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
                - contains the table created with `model.summary()`
        
        Comments
        --------
    """

    #rewrite info to list
    temp = []
    model.summary(width, print_fn=lambda line: temp.append(line))

    #extract relevant rows
    # name = temp[0][8:-1]
    # header = temp[2]
    data = temp[4:-5]
    footer = temp[-4:-1]

    
    #extract entries and format as pandas dataframe    
    df_summary = pd.DataFrame({
        (model.name, "Layer"):[],
        (model.name, "Type"):[],
        (model.name, "Output Shape"):[],
        (model.name, "Param #"):[],
        (model.name, "Activation"):[],
        (model.name, "Filters"):[],
        (model.name, "Kernel Size"):[],
        (model.name, "Units"):[],
    }, dtype=str)
    
    #extract some additional information for each layer
    activations = []
    filters = []
    kernel_sizes = []
    units = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "activation"):
            activations.append(layer.activation.__name__)
        else:
            activations.append(None)
        if hasattr(layer, "filters"):
            filters.append(layer.filters)
        else:
            filters.append(None)
        if hasattr(layer, "kernel_size"):
            kernel_sizes.append(layer.kernel_size)
        else:
            kernel_sizes.append(None)
        if hasattr(layer, "units"):
            units.append(layer.units)
        else:
            units.append(None)
    
    for idx, (row, activation, filter, kernel_size, u) in enumerate(zip(
        [re.split(r"\ {2,}", d) for d in data[::2]],
        activations,
        filters,
        kernel_sizes,
        units
        )):
        
        #append row to dataframe
        df_summary.loc[df_summary.shape[0]] = [
            re.findall(r"[^\s\(\)]+", row[0][:])[0],
            re.findall(r"[^\s\(\)]+", row[0][1:])[1],
            eval(row[1]),
            int(row[2]),
            activation,
            filter,
            kernel_size,
            u,
        ]
    if verbose > 0: print(df_summary)

    #extract paramter summary and format as pandas dataframe
    df_params = pd.DataFrame({
        "Total Parameters":[re.sub(r"\,", "", footer[0][14:])],
        "Trainable Parameters":[re.sub(r"\,", "", footer[1][18:])],
        "Non-Trainable Parameters":[re.sub(r"\,", "", footer[2][22:])],
    })
    if verbose > 0: print(df_params)

    #write latex-table to external file
    if isinstance(latex_to_file, str):
        with open(latex_to_file, write_mode) as outfile:
            outfile.write("\n\n"+"#"*50+"\n")
            outfile.write(f"Model:{model.name}\n")
            outfile.write(f"Total Parameters:{df_params['Total Parameters'].iloc[0]}\n")
            outfile.write(f"Trainable Parameters:{df_params['Trainable Parameters'].iloc[0]}\n")
            outfile.write(f"Non-Trainable Parameters:{df_params['Non-Trainable Parameters'].iloc[0]}\n")
            outfile.write(df_summary.to_latex(
                index=False, position="H",
                label="tab:YOURLABEL", caption=f"Summary table of {model.name}"))

    if isinstance(save, str): df_summary.to_csv(save, index=False)


    return df_summary
