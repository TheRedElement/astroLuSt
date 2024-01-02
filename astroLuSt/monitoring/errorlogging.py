
#%%imports
import pandas as pd
import re
import traceback

from astroLuSt.monitoring import formatting as almof

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
                - rows corresponding to the same error will have the same index
                - has the following columns
                    - 'exception'
                        - the raised `Exception`
                    - 'prefix'
                        - a prefix defined by the user
                        - i.e., for comments
                    - 'suffix'
                        - a suffix defined by the user
                        - i.e., for comments
                    - 'file'
                        - the file that contains the problem-line 
                    - 'line'
                        - number of the problem-line 
                    - 'problem_line'
                        - the problem-line 
                    - 'error_message'
                        - the type of error message 
                    - 'time'
                        - a timestamp of when the error was caught 
                    - 'squeezed'
                        - whether all extracted errors have been squeezed into one line
                        - relevant if some `ValueError` occurs when creating `df_temp`        
                
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
            columns=['exception', 'prefix', 'suffix', 'file', 'line', 'problem_line', 'error_message', 'time', 'squeezed']
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
        store:bool=True,
        verbose:int=None
        ) -> pd.DataFrame:
        """
            - method to store a caught exception e to a `pandas.DataFrame`

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
                - `store`
                    - bool, optional
                    - whether to also store the created dataframe to `self.df_errorlog`
                    - the default is True
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
            
            Raises
            ------
            
            Returns
            -------
                - `df_temp`
                    - pd.DataFrame
                    - temporary dataframe containing information for last exception
                    - has the following columns
                        - 'exception'
                            - the raised `Exception`
                        - 'prefix'
                            - a prefix defined by the user
                            - i.e., for comments
                        - 'suffix'
                            - a suffix defined by the user
                            - i.e., for comments
                        - 'file'
                            - the file that contains the problem-line 
                        - 'line'
                            - number of the problem-line 
                        - 'problem_line'
                            - the problem-line 
                        - 'error_message'
                            - the type of error message 
                        - 'time'
                            - a timestamp of when the error was caught 
                        - 'squeezed'
                            - whether all extracted errors have been squeezed into one line
                            - relevant if some `ValueError` occurs when creating `df_temp`

            Comments
            --------
                - if nested `try...except` statements raised the error
                    - will only care about the last (deepest) one
                    - the rest will be stored as part of `format_exc` in `self.df_errorlog['exception']` anyways
                    
        """

        #default parameters
        if prefix is None:  prefix  = ''
        if suffix is None:  suffix  = ''        
        if verbose is None: verbose = self.verbose

        format_exc = traceback.format_exc()

        #for nested try...except:
        #   - only care about last error
        #   - rest will be stored in `format_exc` anyways
        files           = re.findall(r'(?<=File ").+(?=")', format_exc)        
        lines           = re.findall(r'(?<=line )\d+(?=,)', format_exc)
        # problem_lines   = re.findall(r'(?<=<module>\n).+',  format_exc)
        problem_lines   = re.findall(r'(?<=\n    ).+',  format_exc)
        # error_msgs      = re.findall(r'\w+Error: [\w ]+',   format_exc)
        error_msgs      = re.findall(r'\w+Error[:\w ]+',    format_exc)[-1:]

        #in case no problem lines have been identified
        if len(problem_lines) == 0: problem_lines = ['<NO PROBLEM LINE IDENTIFIED>']*len(files)

        #check if all shapes match nicely
        try:
            df_temp = pd.DataFrame({
                'exception':    [format_exc]*len(files),
                'prefix':       [prefix]*len(files),
                'suffix':       [suffix]*len(files),
                'file':         files,
                'line':         lines,
                'problem_line': problem_lines,
                'error_message':error_msgs*len(files),
                'time':         [pd.Timestamp.now()]*len(files),
                'squeezed':     [False]*len(files),
            })  
        #sqeeze everything in one row if some shapes do not match (no loss of information, but not as nice formatting)
        except ValueError as ve:
            df_temp = pd.DataFrame({
                'exception':    [format_exc],
                'prefix':       [prefix],
                'suffix':       [suffix],
                'file':         [';'.join(files)],          #transform to 1 line
                'line':         [';'.join(lines)],          #transform to 1 line
                'problem_line': [';'.join(problem_lines)],  #transform to 1 line
                'error_message':error_msgs,
                'time':         [pd.Timestamp.now()], 
                'squeezed':     [True],                     #whether the content has been sqeezed into one row

            })
            almof.printf(
                msg=(
                    f'All exceptions sqeezed into one row due to `ValueError`.'
                    f' Likely the shapes of the extracted exceptions did not match.'
                    f' Original error message: {ve}'
                ),
                context=self.exc2df.__name__,
                type='WARNING',
                verbose=verbose
            )

        #set index to be the same accross one extraction
        if self.df_errorlog.last_valid_index() is None: idx = 0
        else: idx = self.df_errorlog.last_valid_index() + 1
        df_temp.index = [idx]*df_temp.shape[0]

        if store:
            self.df_errorlog = pd.concat([self.df_errorlog, df_temp])

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


# %%
