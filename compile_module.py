
#%%imports
import re

import glob
import os
import shutil

import datetime
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

def current2legacy(
    modulename:str, version:str='0.0.0'
    ) -> None:
    """
        - function to copy the current state of the module to the 'legacy' directory
    """

    # version_int = int(re.sub('\.', '', version))
    # files_root = glob.glob(f'{modulename}')

    #get versions already in ./legacy
    versions = [re.findall(r'\d+\.\d+\.\d+', f)[0] for f in os.listdir('./legacy')]
    versions_int = [int(re.sub(r'\.','',v)) for v in versions]

    if version in versions:
        print(f'WARNING: The version {version} already exists in "./legacy". Delete the directory before moving the current state to "./legacy"!')
        return


    #copy files into legacy directory
    dest_dir = f'./legacy/{modulename}_v{version}'

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    for source in glob.glob('*'):
        if not re.match(r'legacy|__pycache__|.*~\$+|temp\b|testing|TODOS.md', source):

            if os.path.isdir(source):
                shutil.copytree(f'./{source}', dst=f'{dest_dir}/{source}')
            else:
                shutil.copyfile(f'./{source}', dst=f'{dest_dir}/{source}')

    return


def main():

    #module specifications
    modulename = 'astroLuSt'
    author = 'Lukas Steinwender'
    author_email = 'lukas.steinwender99@gmail.com'
    maintainer = 'Lukas Steinwender'
    maintainer_email = 'lukas.steinwender99@gmail.com'
    url = "https://github.com/TheRedElement/astroLuSt"
    lastupdate = str(datetime.date.today())
    version = '0.0.2'

    compile_readme(
        modulename,
        author, author_email,
        maintainer, maintainer_email,
        lastupdate,
        version,        
        url,
        infilename='./templates/README.md',
        outfilename='README.md'
    )

    current2legacy(modulename=modulename, version=version)

    print('FINISHED')


if __name__ == '__main__':
    main()