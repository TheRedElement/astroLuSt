
#%%imports
import pandas as pd

#%%definitions

def hypergrid2latex(
    hypergrid:dict,
    pd_to_latex_kwargs:dict=None
    ) -> str:
    """
        - function to take a dictionary defining a hyperparameter grid a la sklearn convention and produce a LaTeX table from that

        Parameters
        ----------
            - `hypergrid`
                - dict
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

    #generate DataFrame
    df_hypergrid = pd.DataFrame({k:[v] for k, v in hypergrid.items()}).T.reset_index()
    
    #Set Column names accordingly
    df_hypergrid.columns = ['Parameter', 'Value']
    
    #Replace list-symbols with curly brackets (set of...)
    df_hypergrid['Value'] = df_hypergrid['Value'].astype(str).replace({r'^\[':'\{', r'\]$':'\}'}, regex=True)
    
    #generate latex representation
    hypergrid_str = df_hypergrid.to_latex(**pd_to_latex_kwargs)
    
    return hypergrid_str