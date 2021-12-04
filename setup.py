
# A setuptools based setup module.
# See:
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
# https://github.com/pypa/sampleproject


from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

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
        "datetime",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Topic :: astronomy, datanalysis",
    ]

)

# from setuptools import setup, find_packages
# import pathlib

# here = pathlib.Path(__file__).parent.resolve()

# # Get long description from the README file
# long_description = (here / "README.md").read_text(encoding="utf-8")

# setup(
#     name="astroLuSt",
#     version="0.0.1",
#     description="Module containing tools useful especially in astronomy.",
#     long_description=long_description,
#     long_description_content_type='text/markdown',
#     url="https://github.com/TheRedElement/astroLuSt",
#     author="Steinwender Lukas",
#     author_email="lukas.steinwender99@gmail.com",
#     classifiers=[
#         "Intended Audience :: Science/Research",
#         "Programming Language :: Python :: 3.8",
#         "Topic :: Scientific/Engineering :: Astronomy",
#         "Natural Language :: English",
#         "Development Status :: 2 - Pre-Alpha",
#     ],
#     keywords="Astronomy, Dataanalysis",
#     packages=["astroLuSt"],
#     install_requires=[
#         "numpy",
#         "matplotlib",
#         "scipy",
#         "datetime",
#     ],
#     data_files=[
#         ('needed_files', ['files/Colorcodes.txt']),
#     ],
#     project_urls={
#         "Source: https://github.com/TheRedElement/astroLuSt" 
#     },
# )