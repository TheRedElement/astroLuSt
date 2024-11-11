# astroLuSt

A collection of functions useful for dataprocessing especially in astronomy.
This module started out as a central space for all code generated in the frame of my theses.

* Author: <author>
* Author Email: <author_email>
* Maintainer: <maintainer>
* Maintainer Email: <maintainer_email>
* Last Update: <lastupdate>
* Version: <version>
* URL: <url>

## Installation

To install `astroLuSt` simply call the following from your console:
```shell
python -m pip install git+https://github.com/TheRedElement/astroLuSt
```
For installation without dependencies use the following:
```shell
python -m pip install --no-dependencies git+https://github.com/TheRedElement/astroLuSt
```
* You might need to use `pip3` instead of `pip` depending on your python version.
* Make sure you have `git` installed before you intall `astroLuSt`!

Because of package dependencies within [`eleanor`](https://adina.feinste.in/eleanor/) you currently have to manually downgrade `photutils` to version `1.13.0`
```shell
python -m pip install --force-reinstall -v photutils==1.13.0
```

## Examples

Examples for the different functionalities contained within the module can be found in the demo files ([./demos/](./demos/)).
This directory has the same structure as  the main [astroLuSt](./astroLuSt/) directory.

## Parts

* The module is contained within [./astroLuSt/](./astroLuSt/).
* Demonstrations for different functionalities can be found in [demos/](./demos/).
* Previous versions are saved in [./legacy/](./legacy/)

## Referencing

If you use any of the utilities provided in the `astroLuSt`-module, I would be very glad if you reference the current github repo:
[https://github.com/TheRedElement/astroLuSt](https://github.com/TheRedElement/astroLuSt)

## Version Info

| Version   | Date  | Notes |
| -         | -     | -     |
| v1.0.0    | <lastupdate>    | Inclusion of  code from Master's Thesis in computer-science. Reorganization of directory tree. Addition of demo for every function and class.|
| v0.0.0    | 2021-10-21    | Initial release. Code from Bachelor's and Master's Thesis in astronomy. |