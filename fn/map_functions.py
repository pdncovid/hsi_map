from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math


def view_maps(dict_, fig_name):
    """Function to view a map on matplotlib. Inputs a dictionary.
    TODO: edit to input array and string"""
    print('view maps...')

    field_names = list(dict_.keys())
    cols = 4
    rows = math.ceil(len(field_names) / cols)
    plt.figure(figsize=(3.5 * cols, 3.5 * rows))
    for i in range(len(field_names)):
        field = field_names[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(dict_[field])
        plt.title(field)
    plt.suptitle(fig_name)
    plt.show()

    return


def load_maps(folder='mat_files', file='imageKandy', view=False, out_array=False):
    """Loads the matlab file and returns 2 values where:
    field_names=str is the name of each image (eg: CoastalBlue)
    struct_dc=np.ndarray is the 2D array corresponding to each name OR
    struct_dc=dict where key=field_names and value=2D array (depending on out_array)"""
    print('loading maps...')
    if '.mat' in file:
        file = file.replace('.mat', '')

    dc = loadmat(folder + '/' + file, struct_as_record=True)
    struct = dc[file][0, 0]
    field_names = struct.dtype.names

    struct_dc = dict()
    for field in field_names:
        struct_dc[field] = struct[field]

    if view:
        view_maps(struct_dc, file)

    if out_array:
        _arr = list(struct_dc.values())
        struct_dc = np.array(_arr)

    return [struct_dc, field_names]


def plot_sample(test_sample):
    print('plot samples...')
    col = 4
    row = math.ceil(test_sample.shape[0] / col)
    plt.figure(figsize=(col * 4, row * 4))
    for i in range(test_sample.shape[0]):
        plt.subplot(row, col, i + 1)
        plt.imshow(test_sample[i, :, :])
    plt.suptitle(
        'Windowed hyper-spectral image\nResolution: ' + str(test_sample.shape[-1]) + ' * ' + str(test_sample.shape[1]))
    plt.show()

    return
