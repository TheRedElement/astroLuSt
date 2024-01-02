#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import random
from scipy.signal import sawtooth
from scipy.stats import norm
import string
from typing import Union, Tuple, List, Callable

from astroLuSt.preprocessing import dataseries_manipulation as alpdm
from astroLuSt.monitoring import formatting as almof

#%%definitions
class GenUniqueStrings:
    """
        - class to generate unique strings from a given set of characters to choose from
        - roughly follows scipy.stats distribution conventions


        Attributes
        ----------
            - `n`
                - int, optional
                - lengths of the strings to generate (not counting `suffix` and `prefix`)
                - the default is 1
            - `char_choices`
                - int, list, optional
                - iterable providing the set of characters to choose from
                - the default is `None`
                    - will generate using uppercase letters and numbers
            - `prefix`
                - str, optional
                - a prefix to put in front of every generated string
                - the default is `None`
            - `suffix`
                - str, optional
                - a suffix to put at the end of every generated string
                - the default is `None`

        Methods
        -------
            - `rvs()`

        Dependencies
        ------------
            - numpy
            - random
            - string
            - typing

        Comments
        --------

    """

    def __init__(self,
        n:int=1,
        char_choices:Union[list,str]=None,
        prefix:str=None,
        suffix:str=None,
        ) -> None:

        self.n = n
        if char_choices is None: self.char_choices = string.ascii_uppercase+string.digits
        else:                    self.char_choices = char_choices
        if prefix is None: self.prefix = ''
        else:              self.prefix = prefix
        if suffix is None: self.suffix = ''
        else:              self.suffix = suffix

        return
    
    def __repr__(self) -> str:

        return (
            f'GenUniqueStrings(\n'
            f'    n={repr(self.n)},\n'
            f'    char_choices={repr(self.char_choices)},\n'
            f'    prefix={repr(self.prefix)},\n'
            f'    suffix={repr(self.suffix)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def rvs(self,
        shape:Union[int, tuple]=None,
        random_state:int=None,
        ):
        """
            - method similar to the `scipy.stats` `rvs()` method
            - rvs ... random variates
            - will generate an array of size `shape` containing randomly generated samples
            
            Parameters
            ----------
                - `shape`
                    - int, tuple optional
                    - number of samples to generate
                    - the default is `None`
                        - will generate a single sample
                - `random_state`
                    - int, optional
                    - seed of the random number generator
                    - provide any integer for reproducible results
                    - the default is `None`
                        - non-reproducible results

            Raises
            ------

            Returns
            -------
                - `output`
                    - np.ndarray, str
                    - array containing the generated strings
                    - if only one sample got generated and `shape` is of type int a single string will be returned
            
            Comments
            --------
        """

        #new instance of RNG
        randstate = random.SystemRandom(random_state)

        if shape is None:
            shape = 1
        # else:
        n_samples = int(np.prod(shape))

        output = np.array([])
        for idx in range(n_samples):
            uid = ''.join(randstate.choices(self.char_choices, k=self.n))
            output = np.append(output, self.prefix + uid + self.suffix)

        output = output.reshape(shape)

        #if a single sample got generated, and an integer was passed just return the string
        if n_samples == 1 and isinstance(shape, int):
            return output[0]
        #otherwise return an np.ndarray
        else:
            return output
    
class GeneratePeriodicSignals:
    """
        - class to generate a unique series of random periodic signals

        Attributes
        ----------
            - `npoints`
                - np.ndarray, int, optional
                - number of points per dataseries/sample
                - defines the number of samples to generate
                    - i.e., `len(npoints)` is equal to the number of generated samples
                - if np.ndarray
                    - has to have a length equal to the number of samples to generate
                    - each entry will be interpreted as the number of datapoints to use for that particular sample
                - if int
                    - will use this many datapoints for all generated samples
                - the default is 100
            - `periods`
                - np.ndarray, int, float, optional
                - periods of the individual generated dataseries
                    - will generate as many unique periodized dataseries as elements in `npoints`
                - if int or float
                    - will use this period for all dataseries
                - if np.ndarray
                    - defines the periods of the composite signals for each sample
                    - has to have shape `(nsamples, ncomposites)`
                        - `nsamples`
                            - number of samples that will be generated in total
                        - `ncomposites`
                            - number of functions that will get superpositioned to defined the final result
                - the default is 1
            - `amplitudes`
                - np.ndarray, int, float, optional
                - amplitudes of the individual generated dataseries
                    - will generate as many unique periodized dataseries as elements in `npoints`
                - if int or float
                    - will use this amplitude for all dataseries
                - if np.ndarray
                    - defines the amplitude of the composite signals for each sample
                    - has to have shape `(nsamples, ncomposites)`
                        - `nsamples`
                            - number of samples that will be generated in total
                        - `ncomposites`
                            - number of functions that will get superpositioned to defined the final result
                - the default is 1
            - `x_min`    
                - np.ndarray, int, float, optional
                - minimum x-value of the individual generated dataseries
                    - will generate as many unique periodized dataseries as elements in `npoints`
                - if int or float
                    - will use this minimum value for all dataseries
                - if np.ndarray
                    - defines the minimum x-value for each sample
                    - has to have shape `(nsamples)`
                        - `nsamples`
                            - number of samples that will be generated in total
                - the default is 0
            - `x_max`
                - np.ndarray, int, float, optional
                - maximum x-value of the individual generated dataseries
                    - will generate as many unique periodized dataseries as elements in `npoints`
                - if int or float
                    - will use this maximum value for all dataseries
                - if np.ndarray
                    - defines the minimum x-value for each sample
                    - has to have shape `(nsamples)`
                        - `nsamples`
                            - number of samples that will be generated in total
                - the default is 10
            - `x_offsets`
                - np.ndarray, int, optional
                - offsets to add to the x-values of the generated periodic dataseries
                - if int
                    - will use that offset for all generated samples
                - if np.ndarray
                    - defines the offsets of the composite signals for each sample
                    - has to have shape `(nsamples, ncomposites)`
                        - `nsamples`
                            - number of samples that will be generated in total
                        - `ncomposites`
                            - number of functions that will get superpositioned to defined the final result
                - the default is 0
            - `choices`
                - np.ndarray, int, optional
                - has to be of dtype object
                - contains either callables (functions) or `np.ndarray`s
                    - callables will be evaluated on phases in `self.rvs()`
                    - `np.ndarray`s will be returned and periodized as they are
                - these choices will be used to randomly generate dataseries
                - the default is `None`
                    - will use an array containing the following methods
                        - `self.sine_()`
                        - `self.cosine_()`
                        - `self.tangent_()`
                        - `self.sawtooth_()`
                        - `self.polynomial_()`
                        - `self.random_()`
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Methods
        -------
            - `sine_()`
            - `cosin_()`
            - `tangent_()`
            - `sawtooth_()`
            - `polynomial_()`
            - `gaussian_()`
                - not used
            - `random_()`
            - `exec_chosen()`
            - `rvs()`
            - `plot_result()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - scipy
            - typing
    """

    def __init__(self,
        npoints:Union[np.ndarray,int]=100,
        periods:Union[np.ndarray,int,float]=None,
        amplitudes:Union[np.ndarray,int,float]=None,
        x_min:Union[np.ndarray,int,float]=0, x_max:Union[np.ndarray,int,float]=10,
        x_offsets:Union[np.ndarray,int]=0,
        choices:np.ndarray=None,
        verbose:int=0,
        ) -> None:
      
        if npoints is None:                         self.npoints    = [100]
        elif isinstance(npoints, int):              self.npoints    = [npoints]
        else:                                       self.npoints    = npoints
        if periods is None:                         self.periods    = np.array([[1]]*len(self.npoints))
        elif isinstance(periods, (int,float)):      self.periods    = np.array([[periods]]*len(self.npoints))
        else:                                       self.periods    = periods
        if amplitudes is None:                      self.amplitudes = [[1]]*len(self.npoints)
        elif isinstance(amplitudes, (int,float)):   self.amplitudes = [[amplitudes]]*len(self.npoints)
        else:                                       self.amplitudes = amplitudes
        if x_min is None:                           self.x_min      = [0]*len(self.npoints)
        elif isinstance(x_min, (int,float)):        self.x_min      = [x_min]*len(self.npoints)
        else:                                       self.x_min      = x_min
        if x_max is None:                           self.x_max      = [10]*len(self.npoints)
        elif isinstance(x_max, (int,float)):        self.x_max      = [x_max]*len(self.npoints)
        else:                                       self.x_max      = x_max
        if x_offsets is None:                       self.x_offsets  = [[0]]*len(self.npoints)
        elif isinstance(x_offsets, (int,float)):    self.x_offsets  = [[x_offsets]]*len(self.npoints)
        else:                                       self.x_offsets  = x_offsets
        self.verbose                                                = verbose


        nsamples = len(self.npoints)

        #check all shapes
        if (len(self.periods) != nsamples) \
            or (len(self.amplitudes) != nsamples) \
            or (len(self.x_min) != nsamples) \
            or (len(self.x_max) != nsamples) \
            or (len(self.x_offsets) != nsamples):
            raise ValueError((
                f'`len(npoints)` has to be the same as'
                f' `len(periods)`, `len(amplitudes)`, `len(x_min)`, `len(x_max)`, `len(x_offsets)`.'
                f' The respective values are:'
                f' {len(self.npoints)=}, {len(self.periods)=}, {len(self.amplitudes)=}, {len(self.x_min)=}, {len(self.x_max)=}, {len(self.x_offsets)=}!'
            ))

        if np.any(self.periods<1e-2):
            almof.printf(
                msg=(
                    f'Found entries in `self.periods` that are < 1e-2.'
                    f' This is likely to cause issues during generation.'
                    f' Consider providing a higher mininmum period or increase `init_res` in `func_kwargs` passed in the call to `self.rvs()` accordingly!'
                ),
                context=self.__class__.__init__.__name__,
                type='WARNING',
                verbose=self.verbose,
            )

        #initialize choices
        self.passed_choices = choices
        if choices is None:
            self.choices = np.array([
                self.sine_,
                self.cosine_,
                self.tangent_,
                self.sawtooth_,
                self.polynomial_,
                # self.gaussian_,
                self.random_,
            ], dtype=object)
        

        pass

    def __repr__(self) -> str:
        choices2print = np.array([c.__name__ for c in self.choices])
        return (
            f'GeneratePeriodicSignals(\n'
            f'    npoints={repr(self.npoints)},\n'
            f'    periods={repr(self.periods)},\n'
            f'    amplitudes={repr(self.amplitudes)},\n'
            f'    x_min={repr(self.x_min)}, x_max={repr(self.x_max)}\n'
            f'    x_offsets={repr(self.x_offsets)},\n'
            # f'    choices={choices2print},\n'
            f'    choices={repr(self.passed_choices)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def sine_(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a sine function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `sin`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """
        sin = np.sin(x*2*np.pi)
        return sin
    
    def cosine_(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a cosine function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `cos`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """
        cos = np.cos(x*2*np.pi)
        return cos
    
    def tangent_(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a tangent function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `tan`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """        
        tan = np.tan(x*2*np.pi)
        return tan 

    def sawtooth_(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a sawtooth function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `st`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """            
        st = sawtooth(x*2*np.pi)
        return st

    def polynomial_(self,
        x:np.ndarray,
        verbose:int=None,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to evaluate a polynomial function

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                - `**kwargs`
                    - kwargs to pass to the function
                        - will use `kwargs['p']`
                            - polynomial coefficients
                        - will use `kwargs['init_res']`
                            - resolution of the template signal
                            - will be scaled by `kwarge['period']` to generate randomness in number of repetitions
                        - will use `kwarge['period']`
                            - period the signal shall have
                            - used to scale `kwargs['init_res']`
                                - to generate randomness in number of repetitions
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `poly`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """
        
        #default parameter
        if verbose is None: verbose =  self.verbose
        
        #resolution of initial (template) 
        ##constant value scaled by period to generate randomness in number of repetitions
        res = int(kwargs['init_res']*kwargs['period'])

        #generate template arrays
        x_ = [np.linspace(0,1,res)]
        poly = [np.polyval(x=x_[0], p=kwargs['p'])]

        #periodize template arrays
        _, poly = alpdm.periodize(
            x_, poly,
            # repetitions=None,
            outshapes=[x.shape[0]],
            testplot=False,
            verbose=verbose-2
        )

        poly = poly[0]

        return poly

    def gaussian_(self,
        x:np.ndarray,
        **kwargs
        ) -> np.ndarray:
        """
            - method to evaluate a gaussian function

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                - `**kwargs`
                    - kwargs to pass to the function
                        - will use `kwargs['loc']`
                            - mean of the gaussian
                        - will use `kwargs['scale']`
                            - standard deviation of the gaussian
                        - will use `kwargs['amp']`
                            - amplitude of the gaussian
                            - i.e. mixing coefficient
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `gauss`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------
                - NOT USED

        """        
        gauss = norm.pdf(x, loc=kwargs['loc'], scale=kwargs['scale']) * kwargs['amp']
        return gauss

    def random_(self,
        x:np.ndarray,
        verbose:int=None,
        **kwargs
        ) -> np.ndarray:
        """
            - method to generate a random array of the same shape as `x`

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `**kwargs`
                    - kwargs to pass to the function
                        - will use `kwargs['init_res']`
                            - resolution of the template signal
                            - will be scaled by `kwarge['period']` to generate randomness in number of repetitions
                        - will use `kwarge['period']`
                            - period the signal shall have
                            - used to scale `kwargs['init_res']`
                                - to generate randomness in number of repetitions                            
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `randarray`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """
        
        #default parameters
        if verbose is None: verbose = self.verbose

        #resolution of initial (template) 
        ##constant value scaled by period to generate randomness in number of repetitions
        res = int(kwargs['init_res']*kwargs['period'])
        
        #generate template arrays
        x_ = [np.linspace(0,1,res)]
        randarray = np.random.randn(1,res)
        
        #periodize template arrays
        _, randarray = alpdm.periodize(
            x_, randarray,
            # repetitions=None,
            outshapes=[x.shape[0]],
            testplot=False,
            verbose=verbose-2
        )

        randarray = randarray[0]

        return randarray

    def exec_chosen(self,
        chosen:Union[Callable,np.ndarray],
        x:np.ndarray, period:float,
        func_kwargs:dict=None
        ) -> np.ndarray:
        """
            - method to execute `chosen`
                - evaluate on `x` if Callable
                - return directly if np.ndarray

            Parameters
            ----------
                - `chosen`
                    - Callable, np.ndarray
                    - callable to be executed or np.ndarray to be returned
                        - if Callable
                            - will be evaluated on `x` given `func_kwargs`
                            - returns (periodized) result
                        - id np.ndarray
                            - returns `x` directly
                - `x`
                    - np.ndarray
                    - input values to evaluate `chosen` on
                - `period`
                    - float
                    - period the periodized signal generated by means of `x` shall have
                - `func_kwargs`
                    - dict, optional
                    - function kwargs passed to `chosen`
                        - i.e. `chosen(x, **func_kwargs)` will be called
                    - the default is `None`
                        - will be initilized with following dict
                            
                            ```python
                            >>> func_kwargs = dict(
                            >>>     p=[np.random.randint(1,5)],
                            >>>     amp=np.random.uniform(0.1,5), loc=np.random.uniform(-1,1), scale=np.random.uniform(0.1,1),
                            >>>     init_res=100,
                            >>> )
                            ```

                        - `p`
                            - will be used by `self.polynomial_()`
                            - polinomial coefficients
                        - `amp`
                            - not used anymore
                            - will be used by `self.gaussian_()`
                            - amplitude of the signal
                        - `loc`
                            - not used anymore
                            - will be used by `self.gaussian_()`
                            - mean of the gaussian
                        - `scale`
                            - will be used by `self.gaussian_()`
                            - not used anymore
                            - standard deviation of the gaussian
                        - `init_res`
                            - will be used by `self.polynomial_()` and `self.random()`
                            - resolution of the template signal that will be periodized
            
            Raises
            ------

            Returns
            -------
                - `chosen`
                    - np.ndarray
                    - `chosen` evaluated on `x`
                        if `chosen` is a Callable
                        - i.e. `chosen(x, **func_kwargs)`
                    - `chosen` directly
                        - if `chosen` is a `np.ndarray`

            Comments
            --------
        """

        if func_kwargs is None:
            func_kwargs = dict(
                p=[np.random.randint(1,5)],
                amp=np.random.uniform(0.1,5), loc=np.random.uniform(-1,1), scale=np.random.uniform(0.1,1),
                init_res=100,
            )
        if 'p' not in func_kwargs:          func_kwargs['p']        = [np.random.randint(1,5)]
        if 'amp' not in func_kwargs:        func_kwargs['amp']      = np.random.uniform(0.1,5)
        if 'loc' not in func_kwargs:        func_kwargs['loc']      = np.random.uniform(-1,1)
        if 'scale' not in func_kwargs:      func_kwargs['scale']    = np.random.uniform(0.1,1)
        if 'init_res' not in func_kwargs:   func_kwargs['init_res'] = 100
        func_kwargs['period']                                       = period

        if callable(chosen):
            chosen = chosen(x, **func_kwargs)
        else:
            if len(chosen) < len(x):
                chosen = np.pad(chosen, (0,len(x)-len(chosen)), mode='constant', constant_values=(np.nan,np.nan))
        return chosen

    def rvs(self,
        choices:np.ndarray=None,
        noise_level_y:float=0.05,
        noise_level_x:float=0.1,
        random_state:int=None,
        verbose:int=None,
        choices_kwargs:dict=None,
        func_kwargs:List[dict]=None,
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
            - method similar to the `scipy.stats` `rvs()` method
            - rvs ... random variates
            - will generate an array of the same length as `self.npoints` containing randomly generated samples
                - each generated dataseries
                    - can have a different length encoded in `self.npoints`
                    - can consist of a superposition of several composite parts (i.e., sines with different periods and amplitudes)
            
            Parameters
            ----------
                - `choices`
                    - np.ndarray, optional
                    - has to be of dtype object
                    - contains either Callables (functions) or `np.ndarrays`
                        - callables will be evaluated on `x`
                        - np.ndarrays will be returned and periodized as they are
                    - these choices will be used to randomly generate dataseries
                    - the default is `None`
                        - will fallback to `self.choices`
                - `noise_level_x`
                    - float, optional
                    - scale of the noise added to the periodized signal in x-direction
                    - the default is 0.05
                - `noise_level_y`
                    - float, optional
                    - scale of the noise added to the periodized signal in y-direction
                    - the default is 0.1
                - `random_state`
                    - int, optional
                    - not implemented yet
                        - use `np.random.seed(...)` outside the function to ensure repruducibility
                    - seed of the random number generator
                    - provide any integer for reproducible results
                    - the default is `None`
                        - non-reproducible results
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`                        
                - `choices_kwargs`
                    - dict, optional
                    - kwargs to pass to `np.random.choice()`
                        - for selection of random generator-function
                    - usually used to pass probabilities for the different entries in `choices`
                        - i.e. `choices_kwargs=dict(p=[...])`
                    - the default is `None`
                        - will be set to `dict()`
                - `func_kwargs`
                    - `List[dict]`, optional
                    - has to have same length as `self.npoints`
                    - contains dicts with kwargs to pass to the selected `choice` from `choices`
                        - i.e. `self.exec_chosen(..., **func_kwargs[i])` will be called
                    - the default is `None`
                        - will initially be set to `dict()`
                        - will autogenerate kwargs needed to use the default `self.choices`

            Raises
            ------

            Returns
            -------
                - `x_gen`
                    - `List[np.ndarray]`
                        - contains np.ndarrays
                            - can have different lengths
                    - each entry contains the x-values of one generated periodic signal (corresponding entry in `y_gen`)
                - `y_gen`
                    - `List[np.ndarray]`
                        - contains np.ndarrays
                            - can have different lengths
                    - each entry contains the y-values of one generated periodic signal
                        - the entry of `x_gen` contains the corresponding x-values
            
            Comments
            --------
                - make sure that whichever callable you pass within choices can accept `**kwargs`
        """
        
        nsamples = len(self.npoints)
        
        #default parameters
        if choices is None:                 choices         = self.choices        
        if verbose is None:                 verbose         = self.verbose
        if choices_kwargs is None:          choices_kwargs  = dict()
        if func_kwargs is None: func_kwargs = [None]*nsamples
        

        #initialize output lists
        x_gen = []
        y_gen = []

        ##individual samples
        for xn, xx, n, p, a, xo, f_kwargs in zip(self.x_min, self.x_max, self.npoints, self.periods, self.amplitudes, self.x_offsets, func_kwargs):

            #init individual sample
            x = np.linspace(xn, xx, n)
            x += np.random.randn(x.shape[0]) * noise_level_x
            y =  np.random.randn(x.shape[0]) * noise_level_y    #init with noise

            #choose generator-function
            chosen = np.random.choice(choices, size=None, **choices_kwargs)

            #composite parts with different periods and amplitudes
            for pi, ai, xoi in zip(p, a, xo):
                
                #convert to phases for superposition
                phases = (x-xoi)/pi
                #execute chosen generator function
                yi = self.exec_chosen(
                    chosen=chosen, x=phases,
                    period=pi,
                    func_kwargs=f_kwargs
                )
                y += ai*yi

            #append to output
            x_gen += [x]
            y_gen += [y]


        return x_gen, y_gen

    def plot_result(self,
        x_gen:list, y_gen:list,
        p_gen:np.ndarray=None,
        fig_kwargs:dict=None, plot_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to create a testplot visualizing the generated dataseries

            Parameters
            ----------
                - `x_gen`
                    - list
                    - contains x-values of newly generated samples
                - `y_gen`
                    - list
                    - contains y-values of newly generated samples
                - `p_gen`
                    - np.ndarray, optional
                    - contains all p_gen of newly generated signals
                    - will only use the first period (`p_gen[:,0]`) for plotting
                    - the default is `None`
                        - will be ignored when plotting
                - `fig_kwargs`
                    - dict, optional
                    - kwargs to pass to `plt.fig()`
                    - the default is `None`
                        - will initialize with empty dict (`fig_kwargs = {}`)
                - `plot_kwargs`
                    - dict, optional
                    - kwargs to pass to `ax.plot()`
                    - the default is `None`
                        - will initialize with `{'marker':'o'}`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - matplotlib Figure
                    - created matplotlib figure
                - `axs`
                    - `plt.Axs`
                    - axes corresponding to `fig`

            Comments
            --------
        """

        if p_gen is None: p_gen = np.array([None]*len(x_gen))
        if fig_kwargs is None:
            fig_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {'marker':'o'}

        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for xi, yi, pi in zip(x_gen, y_gen, p_gen):
            ax1.plot(xi, yi, **plot_kwargs)
            
            if pi[0] is not None: ax2.scatter(alpdm.fold(xi, pi[0])[1], yi, **plot_kwargs)

        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('y')

        axs = fig.axes   

        return fig, axs

class GenerateViaReperiodizing:
    """
        - class to generate new (semi) artifical samples of periodic signals by reperiodizing an existing set of periodic signals

        Attributes
        ----------
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Methods
        -------
            - `rvs()`
            - `plot_result()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing

        Comments
        --------
    
    """
    def __init__(self,
        verbose:int=0,
        ) -> None:
        
        self.verbose = verbose

        return
    
    def __repr__(self) -> str:
        return (
            f'GenerateViaReperiodizing(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def rvs(self,
        x:List[np.ndarray], y:List[np.ndarray],
        new_periods:np.ndarray,
        periods:Union[list,np.ndarray]=None,
        size:int=None,
        verbose:int=None,
        ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
            - 

            Parameters
            ----------
                - `x`
                    - `List[np.ndarray]`
                    - list of x-values of template dataseries
                        - usually arrays of times
                        - if `periods` was not passed
                            - will be interpreted as phases
                        - those will be reperiodized to generate new dataseries with different periods
                    - will choose `size` random samples from `x`
                - `y`
                    - `List[np.ndarray]`
                    - list of y-values of template dataseries
                    - corresponding to `x`
                - `new_periods`
                    - np.ndarray
                    - contains periods to use for generation of reperiodized signals
                    - for every passed period one random sample from `[x,y,periods]` will be reperiodized to that passed period
                - `periods`
                    - list, np.ndarray, optional
                    - periods corresponding to `x` and `y`
                    - used to convert `x` into phase-domain
                    - the default is `None`
                        - will lead to `x` being interpreted as phases
                - `size`
                    - int, optional
                    - how many samples to generate
                    - if `> len(new_periods)`
                        - will choose `size` random new periods with replacement from ` new_periods`
                    - if `< len(new_periods)`
                        - will choose `size` random new periods without replacement from ` new_periods`
                    - if `== len(new_periods)`
                        - will use periods exactly as they are (no change of order etc)
                    - the default is `None`
                        - will behave like `size = len(new_periods)`
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
                - `x_gen`
                    - `List[np.ndarray]`
                        - contains np.ndarrays
                            - can have different lengths
                    - each entry contains the x-values of one generated periodic signal (corresponding entry in `y_gen`)
                - `y_gen`
                    - `List[np.ndarray]`
                        - contains np.ndarrays
                            - can have different lengths
                    - each entry contains the y-values of one generated periodic signal
                        - the entry of `x_gen` contains the corresponding x-values
                - `p_gen`
                    - `np.ndarray`
                    - periods of the generated signals (`y_gen(x_gen)`)

            Comments
            --------
        """
        
        #default parameters
        if size is None or size == len(new_periods):
            size = len(new_periods)
            p_gen = new_periods   #unchanged... use exactly the passed new_periods
        elif size < len(new_periods):
            p_gen = np.random.choice(new_periods, size=size, replace=False)   #choose random subset
        elif size > len(new_periods):
            p_gen = np.random.choice(new_periods, size=size, replace=True)    #choose random subset with replacing
        
        #convert to NON FOLDED phases
        if periods is None: phases = x                                      #interpret `x` as phases if no periods passed
        else:               phases = [xi/p for xi, p in zip(x,periods)]     #calculate phases if periods have been provided

        

        #generate new signals
        x_gen = []
        y_gen = []
        for idx, new_p in enumerate(p_gen):
            #random index to select random sample in `x` and `y`
            randidx = np.random.randint(0, len(x)-1, size=1)

            #reperiodize signal
            x_gen.append(phases[int(randidx)]*new_p) #reperiodize
            y_gen.append(y[int(randidx)])            #y stays the same


        return x_gen, y_gen, p_gen
    
    def plot_result(self,
        x_gen:list, y_gen:list,
        p_gen:np.ndarray=None,
        fig_kwargs:dict=None, plot_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to create a testplot visualizing the generated dataseries

            Parameters
            ----------
                - `x_gen`
                    - list
                    - contains x-values of newly generated samples
                - `y_gen`
                    - list
                    - contains y-values of newly generated samples
                - `p_gen`
                    - np.ndarray, optional
                    - contains all p_gen of newly generated signals
                    - will only use the first period (`p_gen[:,0]`) for plotting
                    - the default is `None`
                        - will be ignored when plotting
                - `fig_kwargs`
                    - dict, optional
                    - kwargs to pass to `plt.fig()`
                    - the default is `None`
                        - will initialize with empty dict (`fig_kwargs = dict()`)
                - `plot_kwargs`
                    - dict, optional
                    - kwargs to pass to `ax.plot()`
                    - the default is `None`
                        - will initialize with `dict(marker='o')`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - matplotlib Figure
                    - created matplotlib figure
                - `axs`
                    - `plt.Axs`
                    - axes corresponding to `fig`

            Comments
            --------
        """

        if p_gen is None: p_gen = np.array([None]*len(x_gen))
        if fig_kwargs is None:
            fig_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {'marker':'o'}

        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for xi, yi, pi in zip(x_gen, y_gen, p_gen):
            ax1.plot(xi, yi, **plot_kwargs)
            
            if pi is not None: ax2.scatter(alpdm.fold(xi, pi)[1], yi, **plot_kwargs)

        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('y')

        axs = fig.axes   

        return fig, axs
    
