
# A setuptools based setup module.
# See:
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# https://github.com/pypa/sampleproject



#%%imports
import astroLuSt

import re

import datetime

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

#%%definitions
def get_dependencies():
    """
        - obtain required packages from requirements.txt
    """

    with open('requirements.txt', 'r') as infile:
        reqs = infile.read()
        infile.close()
    
    # reqs = re.findall(r'.+(?=\n)', reqs)    #with version
    reqs = re.findall(r'.+(?===)', reqs)    #without version
    reqs = [r for r in reqs if 'pywin' not in r]     #exclude windows-specific packages
    return reqs


#%%module specifications
lastupdate = str(datetime.date.today())

#%%get dependencies
dependencies = get_dependencies()


#Get long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


#%%setup
setup(
    name=astroLuSt.__modulename__,
    version=astroLuSt.__version__,
    author=astroLuSt.__author__,
    author_email=astroLuSt.__author_email__,
    maintainer=astroLuSt.__maintainer__,
    maintainer_email=astroLuSt.__maintainer_email__,
    url=astroLuSt.__url__,
    description='Module containing tools useful especially in astronomy.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url=None,
    include_package_data=True,
    # packages=['astroLuSt'],
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Natural Language :: English',
        'Development Status :: 3 - Alpha',
    ],
    platforms=[],
    keywords='Astronomy, Dataanalysis',
    license=None,
    include=['*'],
    exclude=[
        '__pychache__',
        'demos',
        'legacy',
        'templates',
    ],
    # package_dir={"": "astroLuSt"},
    # packages=find_packages("astroLuSt", exclude=["__pycache__", "PHOEBE_astro_LuSt.py"]),
    python_requires=">=3.10",
    install_requires=dependencies,
    # package_data={
    #     "astroLuSt": ['files/Colorcodes.txt'],
    # },    
    # data_files=[
    #     ('files', ['files/Colorcodes.txt']),
    # ],
    project_urls={
        "Source":astroLuSt.__url__
    },
)
