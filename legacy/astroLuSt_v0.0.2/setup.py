
# A setuptools based setup module.
# See:
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# https://github.com/pypa/sampleproject


#parsing
import re

from datetime import datetime

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

#module specifications
modulename = 'astroLuSt'
author = 'Lukas Steinwender'
author_email = 'lukas.steinwender99@gmail.com'
maintainer = 'Lukas Steinwender'
maintainer_email = 'lukas.steinwender99@gmail.com'
url = "https://github.com/TheRedElement/astroLuSt"
lastupdate = str(datetime.date.today())
version = '0.0.3'

dependencies = [
        "numpy",
        "matplotlib",
        "scipy",
        "datetime",
    ],

# Get long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name=programname,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    url=url,
    description="Module containing tools useful especially in astronomy.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url=None,
    include_package_data=True,
    packages=["astroLuSt"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Natural Language :: English",
        "Development Status :: 2 - Pre-Alpha",
    ],
    platforms=[],
    keywords="Astronomy, Dataanalysis",
    license=None,
    include=['*'],
    exclude=['__pychache__', 'legacy', 'data', 'gfx', 'temp', 'templates', 'tests'],
    # package_dir={"": "astroLuSt"},
    # packages=find_packages("astroLuSt", exclude=["__pycache__", "PHOEBE_astro_LuSt.py"]),
    # python_requires=">=3.8, < 4",
    install_requires=dependencies,
    package_data={
        "astroLuSt": ["files/Colorcodes.txt"],
    },    
    # data_files=[
    #     ('files', ['files/Colorcodes.txt']),
    # ],
    project_urls={
        "Source": "https://github.com/TheRedElement/astroLuSt" 
    },
)
