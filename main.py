import numpy as np

from loading_functions import load_matlab_struct

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
files = ['imageKandy']


def main():
    dc = dict()
    for file in files:
        dc[file] = load_matlab_struct(folder, file, view=True)


if __name__ == "__main__":
    main()
