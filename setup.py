
# A setuptools based setup module.
# See:
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# https://github.com/pypa/sampleproject

# A setuptools based setup module.
# See:
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# https://github.com/pypa/sampleproject

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="astroLuSt",
    version="0.0.1",
    description="Module containing tools useful especially in astronomy.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/TheRedElement/astroLuSt",
    author="Steinwender Lukas",
    author_email="lukas.steinwender99@gmail.com",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Natural Language :: English",
        "Development Status :: 2 - Pre-Alpha",
    ],
    keywords="Astronomy, Dataanalysis",
    # package_dir={"": "astroLuSt"},
    # packages=["astroLuSt"],
    packages=find_packages(exclude=["__pycache__", "PHOEBE_astro_LuSt.py"]),
    # python_requires=">=3.8, < 4",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "datetime",
    ],
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
