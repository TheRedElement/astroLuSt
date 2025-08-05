
#%%imports

import astroLuSt

import datetime
import glob
import os
import re
import shutil
import toml


#%%definitions
def compile_readme(
    modulename:str,
    author:str, author_email:str,
    maintainer:str, maintainer_email:str,
    lastupdate:str,
    version:str,
    url:str,
    infilename:str='./templates/README.md',
    outfilename:str='README.md'
    ) -> None:
    """
        - function to create README.md from a template file
    """

    with open(infilename, 'r') as template:
        readme = template.read()
        template.close()

    #substitute relevant parts
    readme = re.sub('<modulename>',       modulename,        readme)
    readme = re.sub('<author>',           author,            readme)
    readme = re.sub('<author_email>',     author_email,      readme)
    readme = re.sub('<maintainer>',       maintainer,        readme)
    readme = re.sub('<maintainer_email>', maintainer_email,  readme)
    readme = re.sub('<lastupdate>',       lastupdate,        readme)
    readme = re.sub('<version>',          version,           readme)
    readme = re.sub('<url>',              f'[{url}]({url})', readme)

    #create readme-file
    with open(outfilename, 'w') as f:
        f.write(readme)
        f.close()

    return

def make_toml(req_path:str="requirements.txt"):
    """
        - function to create the pyproject.toml file
    """

    #get dependencies
    with open(req_path, "r") as f:
        deps = [l.rstrip("\n") for l in f.readlines()]

    #load existing pyproject.toml
    data = toml.load("pyproject.toml")

    #update pyproject.toml
    data["build-system"]["requires"] = ["hatchling >= 1.26"]
    data["build-system"]["build-backend"] = "hatchling.build"

    data["project"]["name"] = astroLuSt.__modulename__
    data["project"]["version"] = astroLuSt.__version__
    data["project"]["description"] = "package containing tools useful especially in astronomy."
    data["project"]["readme"] = "README.md"
    data["project"]["requires-python"] = ">=3.10"
    data["project"]["classifiers"] = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
    ]
    data["project"]["license"] = "MIT"
    data["project"]["license-files"] = ["LICEN[CS]E*",]
    data["project"]["dependencies"] = deps
    data["project"]["authors"] = [{"name":astroLuSt.__author__, "email":astroLuSt.__author_email__},]
    data["project"]["maintaners"] = [{"name":astroLuSt.__maintainer__, "email":astroLuSt.__maintainer_email__},]
    data["project"]["urls"] = {"Homepage":astroLuSt.__url__, "Issues":astroLuSt.__url__+"/issues"}
    data["project"]["keywords"] = ["astronomy", "data analysis", "machine learning"]

    data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] = [astroLuSt.__modulename__]
    data["tool"]["hatch"]["build"]["exclude"] = [
        "__pychache__", 
        "demos",
        "legacy",
        "templates",
    ]

    #save updated version
    with open("pyproject.toml", "w") as f:
        toml.dump(data, f)

    return

def current2legacy(
    modulename:str, version:str='0.0.0',
    write:bool=True, overwrite:bool=False
    ) -> None:
    """
        - function to copy the current state of the module to the 'legacy' directory
    """

    # version_int = int(re.sub('\.', '', version))
    # files_root = glob.glob(f'{modulename}')

    #get versions already in ./legacy
    versions = [re.findall(r'\d+\.\d+\.\d+', f)[0] for f in os.listdir('./legacy')]
    versions_int = [int(re.sub(r'\.','',v)) for v in versions]

    #abort if the version already exists and the user does not want to overwrite!
    if version in versions:
        print((
            f"WARNING: The version {version} already exists in `./legacy`.\n"
            f"Delete the directory before moving the current state to `./legacy`!"
        ))
        
        #require user action if potential of overwriting existing version
        overwrite = input("Would you like to overwrite the existing version? (y,n)")
        while overwrite not in ["y","n"]:
            print("Choose `y` to overwite or `n` to abort!")
            overwrite = input("Would you like to overwrite the existing version? (y,n)")
        if overwrite == "n":
            print("Aborting...")
            overwrite = False
            return
        else: 
            print("Overwriting...")
            overwrite = True
            pass
            


    #copy files into legacy directory
    if write:
        dest_dir = f'./legacy/{modulename}_v{version}'

        #delete an existing stored version
        if os.path.exists(dest_dir) and overwrite:
            shutil.rmtree(dest_dir)
        
        #create new directory
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        #copy to `legacy`
        for source in glob.glob('*'):
            if not re.match(r'legacy|__pycache__|.*~\$+|temp\b|testing|TODOs.md', source):
                
                src = f'./{source}'
                dst = f'{dest_dir}/{source}'
                print(f"Copying {src:20s} --> {dst}")
                if os.path.isdir(source):
                    shutil.copytree(src, dst=dst)
                else:
                    shutil.copyfile(src, dst=dst)

    return


def main():

    lastupdate = str(datetime.date.today())

    compile_readme(
        modulename=astroLuSt.__modulename__,
        author=astroLuSt.__author__, author_email=astroLuSt.__author_email__,
        maintainer=astroLuSt.__maintainer__, maintainer_email=astroLuSt.__maintainer_email__,
        lastupdate=lastupdate,
        version=astroLuSt.__version__,
        url=astroLuSt.__url__,
        infilename='./templates/README.md',
        outfilename='README.md'
    )

    make_toml(req_path="requirements.txt")

    current2legacy(
        modulename=astroLuSt.__modulename__, version=astroLuSt.__version__,
        write=False,
    )

    print('FINISHED')


if __name__ == '__main__':
    main()
