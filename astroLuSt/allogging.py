
#%%imports
import logging

#%%definitions
def setup_logger(level:int=logging.INFO):
    """
        - utility function to setup a logger for some script

        Parameters
        ----------
            - `level`
                - `int`, optional
                - logging level to use
                - the default is `logging.INFO`

        Raises
        ------

        Returns
        -------

        Dependencies
        ------------
            - `logging`

        Comments
        --------
            - extra fields
                - `context`
                    - add some context to the loggin (i.e. function name)
                - `level`
                    - defines the indentation level of some message
                    - the default is 0
                        - leftaliged
                - `indent_char`
                    - character(s) to use for start of indentation
                    - gets padded and truncated to a length equivalent to `level`
        
    """

    format = "{indent_char:{level}.{level}}{levelname:<8.8s}: {asctime}, {name} [{processName:<12.12s}]: {message} ({context})"
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(format, defaults=dict(context="", level=0, indent_char=""), style="{")
    handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[handler]
    )

    return

