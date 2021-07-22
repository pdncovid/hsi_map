from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math


def view_maps(dict_, fig_name):
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
