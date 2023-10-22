
#%%imports
from typing import Literal

#%%definitions
def printf(
    msg:str, context:str=None,
    type:Literal['INFO', 'WARNING']=None,
    verbose:int=0,
    ):
    """
        - function to print a formatted mesage

        Parameters
        ----------
            - `msg`
                - str
                - message to be printed
            - `context`
                - str, optional
                - context to the printed message
                - the default is `None`
                    - will print `''`
            - `type`
                - Literal, optional
                - type of the message
                - allowed strings are
                    - `'INFO'`
                    - `'WARNING'`

        Raises
        ------

        Returns
        -------

        Dependencies
        ------------
            - typing

        Comments
        --------
    """
    if context is None: context = ''
    if type is None: type = 'INFO'
    
    #determine when and what to show
    if type == 'INFO':      vbs_th = 2
    elif type == 'WARNING': vbs_th = 1

    to_print = (
        f'{type}({context}): {msg}'
    )
    if verbose >= vbs_th:
        print(to_print)

    return