# astroLuSt

A collection of functions useful for dataprocessing especially in astronomy.

Author: Lukas Steinwender
Author Email: lukas.steinwender99@gmail.com
Maintainer: Lukas Steinwender
Maintiner Email: lukas.steinwender99@gmail.com
Last Update: 2023-03-01
Version: 0.0.2
URL: [https://github.com/TheRedElement/astroLuSt](https://github.com/TheRedElement/astroLuSt)

## Installation

To install astroLuSt simply call
```shell
python -m pip install git+https://github.com/TheRedElement/astroLuSt
```
from your console.
You might need to use pip3 instead of pip depending on your python version. <br>
Make sure you have `git` installed before you intall __astroLuSt__!

## Files

The different parts of the module are saved in *./astroLuSt*.
The current version consists of the following parts:
- __data_astroLuSt.py__
    - classes and functions useful for data-processing
    - classes for unit conversions
        - not complete
        - will get expanded over time
    - classes and functions useful to create synthetic data-series
- __PHOEBE_astroLuSt.py__
    - classes and functions useful for working with PHOEBE
- __plotting_astroLuSt.py__
    - classes and functions useful for plotting
- __utility_astroLuSt.py__
    - classes and functions for other convenient stuff to improve the workflow

The structure of the module is the following:

```
astroLuSt
|-- astroLuSt
|   |-- files
|   |   `-- Colorcodes.txt
|   |-- __init__.py
|   |-- data_astroLuSt.py
|   |-- PHOEBE_astroLuSt.py
|   |-- plotting_astroLuSt.py
|   `-- utility_astroLuSt.py
|-- astroLuSt_example_usage.ipynb
|-- MANIFEST.in
|-- README.md
`-- setup.py
```

## Dependencies

The current dependencies are
- numpy
- matplotlib
- scipy
- re
- datetime
- copy
- pathlib

## Examples

To get a feel on how the different functions and classes behave, I provided some information in __astroLuSt_example_usage.ipynb__.

## Referencing

If you use any of the utilities provided in the __astroLuSt__-module, I would be very glad if you reference the current homepage:
[https://github.com/TheRedElement/astroLuSt](https://github.com/TheRedElement/astroLuSt)