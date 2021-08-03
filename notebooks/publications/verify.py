import warnings

warnings.filterwarnings('ignore')

import glob
from os import getcwd, listdir, path

import papermill as pm

if __name__ == '__main__':
    d = getcwd()
    subdirs = [path.join(d, o) for o in listdir(d) if
               path.isdir(path.join(d, o))]

    for dir in subdirs:
        notebooks = glob.glob(f"{dir}/*.ipynb", recursive=False)
        for notebook in notebooks:
            print(f"Executing {notebook}")
            pm.execute_notebook(notebook, '/dev/null', cwd=dir)
