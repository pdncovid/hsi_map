import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from fn.data_functions import samples_get, samples_zeros
from fn.map_functions import load_maps
from fn.spectral_functions import get_distance, get_affinity, get_laplacian, get_eigen, sigma_sweep

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
file = 'imageKandy'


def main():
    # loading the maps
    [stacked_arr, names] = load_maps(folder, file, view=False, out_array=True)
    """ how many pieces you want to break the map to, and getting those samples (also removing empty parts). To get a
    better idea set view=True """
    # TODO: Find an under-sampling method so that we do not have to window the original image.
    rows = [50, 50]
    data_sampled = samples_get(stacked_arr, rows, view=False, print_=False)
    [data_sampled_full, _] = samples_zeros(data_sampled, view=False, print_=False)
    # getting a test sample for our analysis
    test_sample = data_sampled_full[:, :, :, int(0.7 * data_sampled_full.shape[-1])]

    # are we following Von Luxburg (True) or Ng and Weiss (False)?
    von_luxburg = False

    # set our desired number of clusters
    k = 5
    sig_opt = sigma_sweep(test_sample, k, cluster=True, sig_range=None, von_luxburg=False, abs_eig=True,
                          view_sample=True, view_sig=True)


if __name__ == "__main__":
    main()
