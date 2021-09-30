import os
import time
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from scipy.fftpack import fft, dctn, idctn

from fn.data_functions import samples_get, samples_zeros
from fn.map_functions import load_maps
from fn.spectral_functions import get_distance, get_affinity, get_laplacian, get_eigen, sigma_optimum, cluster_spectral, \
    cluster_kmeans, dct_image

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
file = 'imageKandy'


def main():
    # loading the maps
    [stacked_arr, names] = load_maps(folder, file, view=False, out_array=True)

    # image under-sampling

    # TRYING 2D-DCT (SEEMS OK)
    """
    img1 = stacked_arr[:, 1750:2250, 1750:2250]
    img1 = img1 / np.amax(img1)

    img_new = dct_image(img1, 8, 32, view=True)
    test_sample = np.copy(img_new)
    """

    # TRYING 2D-FFT (FAILED)
    """
    img_fft = np.fft.fft2(img1, norm=None)
    print(img_fft.shape)

    fft_real = np.real(img_fft)
    fft_imag = np.imag(img_fft)

    x_mid = np.round(img_fft.shape[1] / 2, 0).astype(int)
    y_mid = np.round(img_fft.shape[0] / 2, 0).astype(int)

    down_scale = 1.1
    x_len = np.round(0.5 * img_fft.shape[1] / down_scale, 0).astype(int)
    y_len = np.round(0.5 * img_fft.shape[0] / down_scale, 0).astype(int)

    # fft_down = img_fft[y_mid - y_len:y_mid + y_len, x_mid - x_len:x_mid + x_len]
    fft_down = img_fft[0:2*x_len, 0:2*x_len]

    img_final = np.fft.ifft2(fft_down)
            
    # plt.figure()
    # plt.imshow(np.real(fft_down))
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(np.imag(fft_down))
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(img1)
    # plt.show()

    plt.figure()
    plt.imshow(np.abs(img_final))
    plt.show()
    # plt.figure(figsize=(10, 3))
    # plt.subplot(131)
    # plt.imshow(img1)
    # plt.subplot(132)
    # plt.imshow(np.real(img_fft))
    # plt.colorbar()
    # plt.title('real part of 2d-fft')
    # plt.subplot(133)
    # plt.imshow(np.imag(img_fft))
    # plt.colorbar()
    # plt.title('imaginary part of 2d-fft')
    # plt.show()
    # WINDOWING FUNCTIONS
    """

    # WINDOWING THE BIG PICTURE
    """
    rows = [10, 10]
    data_sampled = samples_get(stacked_arr, rows, view=False, print_=True)
    [data_sampled_full, _] = samples_zeros(data_sampled, view=False, print_=True)
    test_sample = data_sampled_full[:, :, :, int(0.5 * data_sampled_full.shape[-1])]
    """

    # CLUSTERING PARTS
    """
    # are we following Von Luxburg (True) or Ng and Weiss (False)?
    method = 'Ng and Weiss'
    # method = 'Von Luxburg'
    # set our desired number of clusters
    k = 4
    # sig_opt = sigma_optimum(test_sample, k, sig_range=[0.3, 0.6], cluster=False, method=method, abs_eig=True,
    #                         view_sample=True, view_sig=True, view_clusters=False)

    # sig_opt = 0.433

    # cluster_spectral(test_sample, k, sig_opt, method=method, view_sample=False, view_clusters=True)
    labels_2d, one_hot_2d = cluster_kmeans(test_sample, k, view_clusters=True, view_sample=True)
    """


if __name__ == "__main__":
    main()
