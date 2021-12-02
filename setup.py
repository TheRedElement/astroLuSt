
from setuptools import setup

setup(
    name="astroLuSt",
    version="0.0.1",
    url="https://github.com/TheRedElement/astroLuSt",
    author="Steinwender Lukas",
    author_email="lukas.steinwender99@gmail.com",
    description="Module containing code especially useful in astronomy.",
    long_description="Module containing code especially useful in astronomy.",
    packages=["astroLuSt"],
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "re",
        "datetime",
        "copy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Topic :: astronomy, datanalysis",
    ]

)