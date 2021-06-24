import os
from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt


def load_matlab_struct(folder='mat_files', file='imageKandy', view=False):
    if '.mat' in file:
        file = file.replace('.mat', '')

    dc = loadmat(folder + '/' + file, struct_as_record=True)
    struct = dc[file][0, 0]
    field_names = struct.dtype.names

    struct_dc = dict()

    for field in field_names:
        struct_dc[field] = struct[field]

    if view:
        cols = 4
        rows = math.ceil(len(field_names) / cols)
        plt.figure(figsize=(3*cols, 3*rows))
        for i in range(len(field_names)):
            field = field_names[i]
            plt.subplot(rows, cols, i+1)
            plt.imshow(struct_dc[field])
            plt.title(field)
        plt.suptitle(file)
        plt.show()



    return struct_dc
