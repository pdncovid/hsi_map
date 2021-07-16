import numpy as np
from map_functions import load_maps, view_maps
from data_functions import min_max_normalise

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
files = ['imageKandy']


def main():
    dc = dict()
    for file in files:
        dc[file] = load_maps(folder, file, view=True)
        dc_init = dc[file]
        dc_norm = min_max_normalise(dc_init, 'all')
        array_init = np.array(list(dc_init.values()))
        array_norm = np.array(list(dc_norm.values()))
        print('haha')


if __name__ == "__main__":
    main()
