import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fn.data_functions import samples_get, samples_zeros
from fn.map_functions import load_maps
from fn.spectral_functions import get_distance, get_affinity, get_laplacian, get_eigen

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
file = 'imageKandy'


def main():
    # are we following tutorial or anutthara rupasinghe?
    tutorial = True
    # set the diagonal of the affinity mat to zero or not
    if tutorial:
        diag_zero = False
    else:
        diag_zero = True

    # loading the maps
    [stacked_arr, names] = load_maps(folder, file, view=False, out_array=True)
    # how many pieces you want to break the map to, and getting those samples (also removing empty parts). To get a
    # better idea set view=True
    rows = [100, 100]
    data_sampled = samples_get(stacked_arr, rows, view=False, print_=False)
    [data_sampled1, zero_idx] = samples_zeros(data_sampled, view=False, print_=False)

    # getting a test sample and plotting it
    test_sample = data_sampled1[:, :, :, int(0.7 * data_sampled1.shape[-1])]

    plt.figure(figsize=(4 * 4, 4 * 2))
    for i in range(test_sample.shape[0]):
        plt.subplot(2, 4, i + 1)
        plt.imshow(test_sample[i, :, :])
    plt.show()

    # some initializations for later analysis
    mode, value, affinity_array, label_list, sig_list, eigval_list, eigvec_list = [], [], [], [], [], [], []

    # fig to be used to plot eigenvalue and eigen-gaps
    plt.figure(figsize=(12, 4))

    # nums is the array of sigmas.
    sigma_array = [0.2]

    # for num in range(15):
    for num in range(len(sigma_array)):
        # sig = 0.025 * num + 0.05  # uncomment this and comment below line if you want to sigma sweep.
        sig = sigma_array[num]
        sig_list.append(sig)

        # compute affinity matrix
        affinity = get_affinity(test_sample, sig, diag_zero=diag_zero, compute_dist=True, view_mat=False,
                                view_hist=False)
        affinity_array.append(affinity)

        # listing out the sigma values for future plots
        label_list.append('\u03C3 = ' + str(round(sig, 2)))

        # computing laplacian
        laplacian = get_laplacian(affinity, tutorial=tutorial, normalize=True, degree_int=False, symmetry=True,
                                  view=False)

        # computing eigenvalues and eigen-gaps
        eigen_gap, eigen_value, eigen_vec = get_eigen(laplacian, sig, eigengap_only=False, view=False)

        eigval_list.append(eigen_value)
        eigvec_list.append(eigen_vec)

        mode.append(np.argmax(eigen_gap))
        value.append(np.amax(eigen_gap))

        # plotting eigen-values and eigen-gaps
        plt.subplot(121)
        plt.plot(np.real(eigen_value), label='real part \u03C3 = ' + str(round(sig, 2)))
        plt.plot(np.abs(eigen_value), label='absolute value')
        plt.ylabel('eigen-value'), plt.xlabel('eigen-value index')
        plt.xlim([0, 50])
        plt.subplot(122)
        plt.plot(np.real(eigen_gap), label='\u03C3 = ' + str(round(sig, 2)))
        plt.plot(np.abs(eigen_gap), label='absolute value')
        plt.ylabel('eigen-gap'), plt.xlabel('eigen-value index')
        plt.xlim([0, 50])

        """ LOOPING OVER SIGMA DONE """

    plt.legend()
    plt.show()

    # plotting dominant mode for each sigma.
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.stem(sig_list, mode)
    plt.title('total # of pixels:' + str(round(test_sample.shape[-1] * test_sample.shape[-2], 0)))
    plt.ylabel('mode')
    plt.xlabel('\u03C3')
    plt.subplot(122)
    plt.stem(sig_list, value)
    plt.title('total # of pixels:' + str(round(test_sample.shape[-1] * test_sample.shape[-2], 0)))
    plt.ylabel('largest eigen-gap')
    plt.xlabel('\u03C3')
    if tutorial:
        plt.suptitle('laplacian computed according to von Luxburg')
    else:
        plt.suptitle('laplacian computed according to A Rupasinghe')
    plt.show()

    print('hehe')

# if you want to plot the affinity distribution.

# affinity_array = np.array(affinity_array)
# affinity_temp = np.reshape(affinity_array, (affinity_array.shape[0], -1))
# affinity_df = pd.DataFrame(affinity_temp.T)
# affinity_df.columns = label_list
# plt.figure(figsize=(8, 5)), sns.displot(data=affinity_df, kind='kde', cut=0)
# plt.xlim([0, 1]), plt.ylabel('Inter-node Affinity density for each \u03C3 value'), plt.xlabel('Affinity')
# plt.subplots_adjust(bottom=0.1), plt.show()


if __name__ == "__main__":
    main()
