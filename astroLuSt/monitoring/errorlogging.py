
#%%imports
import pandas as pd
import re
import traceback

#%%definitions
class LogErrors:
    """
        - class to log errors that occured and got caught via try - except statements

        Attributes
        ----------
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Infered Attributes
        ------------------
            - `df_errorlog`
                - pd.DataFrame
                - dataframe to log all the caught errors
                - if multiple files appear in `'file'`
                    - files will be separated by semicolons (`;`)
                - if multiple lines appear in `'promblem_line'`
                    - lines will be separated by semicolons (`;`)
        
        Methods
        -------
            - `print_exc()`
            - `save_log()`
        
        Dependencies
        ------------
            - pandas
            - re
            - traceback
            
        Comments
        --------
    """

    def __init__(self,
        verbose:int=1,
        ) -> None:
        self.verbose = verbose

        self.df_errorlog = pd.DataFrame(
            columns=['exception', 'prefix', 'suffix', 'file', 'line', 'problem_line', 'error_message', 'time']
        )
        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def print_exc(self,
        e:Exception,
        prefix:str=None, suffix:str=None,
        verbose:int=None,
        ) -> None:
        """
            - method to log a formatted version of the caught exception e
            - will print the formatted version if requested

            Parameters
            ----------
                - `e`
                    - Exception
                    - exception that got caught via `try ... except`
                - `prefix`
                    - str, optional
                    - something to print before the caught exception
                    - the default is `None`
                        - will be set to `''`
                - `suffix`
                    - str, optional
                    - something to print after the caught exception
                    - the default is `None`
                        - will be set to `''`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - will override `self.verbose` if passed
                    - the default is `None`
                        - falls back to `self.verbose`
            
            Raises
            ------
            
            Returns
            -------

            Comments
            --------
        """
        
        #initialize
        if prefix is None:  prefix = ''
        if suffix is None:  suffix = ''
        if verbose is None: verbose = self.verbose

        format_exc = traceback.format_exc()

        if verbose > 0:
            print(prefix)
            print(format_exc)
            print(suffix)


        return
    
    def exc2df(self,
        e:Exception,
        prefix:str=None, suffix:str=None,
        ) -> pd.DataFrame:
        """
            - method to store a caught exception e to a pandas DataFrame

            Parameters
            ----------
                - `e`
                    - Exception
                    - exception that got caught via `try ... except`
                - `prefix`
                    - str, optional
                    - something to print before the caught exception
                    - the default is `None`
                        - will be set to `''`
                - `suffix`
                    - str, optional
                    - something to print after the caught exception
                    - the default is `None`
                        - will be set to `''`
            
            Raises
            ------
            
            Returns
            -------
                - `df_temp`
                    - pd.DataFrame
                    - temporary dataframe containing information for last exception

            Comments
            --------
        """

        #initilaize
        if prefix is None:  prefix = ''
        if suffix is None:  suffix = ''        

        format_exc = traceback.format_exc()

        files           = re.findall(r'(?<=File ").+(?=")', format_exc)        
        lines           = re.findall(r'(?<=line )\d+(?=,)', format_exc)
        problem_lines   = re.findall(r'(?<=<module>\n).+',  format_exc)
        # error_msgs      = re.findall(r'\w+Error: [\w ]+',   format_exc)
        error_msgs      = re.findall(r'\w+Error[:\w ]+',    format_exc)
     
        df_temp = pd.DataFrame({
            'exception':[format_exc],
            'prefix':   [prefix],
            'suffix':   [suffix],
            'file':     ';'.join(files),    #transform to 1 line
            'line':     ';'.join(lines),    #transform to 1 line
            'problem_line':problem_lines,
            'error_message':error_msgs,
            'time':[pd.Timestamp.now()],
        })        

        self.df_errorlog = pd.concat([self.df_errorlog, df_temp]).reset_index(drop=True)

        return df_temp

    def save_log(self,
        filename:str,
        **kwargs
        ) -> None:
        """
            - method to save the current errorlog dataframe (`self.df_errorlog`) to a csv file

            Parameters
            ----------
                - `filename`
                    - str, optional
                    - path to the file to save `self.df_errorlog` to
                - `**kwargs`
                    - kwargs passed to `pd.DataFrame.to_csv()`
            
            Raises
            ------
            
            Returns
            -------
            
            Comments
            --------
        """

        self.df_errorlog.to_csv(filename, **kwargs)

        return

