
#%%imports
import typing

#%%definitions
def printf(
    msg:str, context:str=None,
    type:str=None
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

        Raises
        ------

        Returns
        -------

        Dependencies
        ------------

        Comments
        --------
    """
    if context is None: context = ''
    if type is None: type = 'INFO'

    to_print = (
        f'{type}({context}): {msg}'
    )

    print(to_print)

    return