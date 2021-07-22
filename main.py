import numpy as np
from fn.map_functions import load_maps, view_maps
from fn.data_functions import min_max_normalise, img_sample

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
file = 'imageKandy'


def main():
    [stacked_arr, field_names] = load_maps(folder, file, view=False, out_array=True)
    data1 = np.copy(stacked_arr)
    data2 = img_sample(data1, 100, stacked=True)
    print('original data', data1.shape)
    print('sampled data', data2.shape)


if __name__ == "__main__":
    main()
