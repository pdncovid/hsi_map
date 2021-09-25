import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fn.data_functions import samples_get, samples_zeros
from fn.map_functions import load_maps
from fn.spectral_functions import get_distance, get_affinity, get_laplacian, get_eigen_gap

folder = 'mat_files'
# files = ['imageKandy', 'refImage']
file = 'imageKandy'


def main():
    [stacked_arr, names] = load_maps(folder, file, view=False, out_array=True)
    rows = [100, 100]
    data_sampled = samples_get(stacked_arr, rows, view=False, print_=False)
    [data_sampled1, zero_idx] = samples_zeros(data_sampled, view=False, print_=False)

    test_sample = data_sampled1[:, :, :, int(0.7 * data_sampled1.shape[-1])]

    plt.figure(figsize=(4*4, 4*2))
    for i in range(test_sample.shape[0]):
        plt.subplot(2, 4, i+1)
        plt.imshow(test_sample[i, :, :])
    plt.show()

    mode, value, affinity_array, label_list, sig_list = [], [], [], [], []

    # plt.figure(figsize=(12, 4))

    # nums = [0.2, 0.22, 0.24, 0.26, 0.28, 0.3]

    for num in range(15):
        # compute affinity matrix
        sig = 0.025 * num + 0.15
        # sig = 0.2
        # sig = nums[num]
        sig_list.append(sig)

        affinity = get_affinity(test_sample, sig, diag_zero=True, compute_dist=True, view_mat=False, view_hist=False)
        affinity_array.append(affinity)
        label_list.append('\u03C3 = ' + str(round(sig, 2)))

        laplacian = get_laplacian(affinity, normalize=True, degree_int=False, symmetry=True, view=False)

        eigen_gap, eigen_value = get_eigen_gap(laplacian, sig, eigengap_only=False, view=False)

        mode.append(np.argmax(eigen_gap))
        value.append(np.amax(eigen_gap))

        plt.subplot(121)
        plt.plot(eigen_value, label='\u03C3 = ' + str(round(sig, 2)))
        plt.ylabel('eigen-value'), plt.xlabel('eigen-value index')
        plt.xlim([0, 20])
        # plt.ylim([0, 1])

        plt.subplot(122)
        plt.plot(eigen_gap, label='\u03C3 = ' + str(round(sig, 2)))
        plt.ylabel('eigen-gap'), plt.xlabel('eigen-value index')
        plt.xlim([0, 20])

        # plt.ylim([0, 0.6])

    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.stem(sig_list, mode)
    plt.title('total # of pixels:' + str(round(test_sample.shape[-1] * test_sample.shape[-2], 0)))
    plt.ylabel('mode')
    plt.xlabel('\u03C3')
    # plt.ylim([0, 200])
    # plt.yscale('log'), plt.subplots_adjust(left=0.2)
    plt.show()


# affinity_array = np.array(affinity_array)
# affinity_temp = np.reshape(affinity_array, (affinity_array.shape[0], -1))
# affinity_df = pd.DataFrame(affinity_temp.T)
# affinity_df.columns = label_list
# plt.figure(figsize=(8, 5)), sns.displot(data=affinity_df, kind='kde', cut=0)
# plt.xlim([0, 1]), plt.ylabel('Inter-node Affinity density for each \u03C3 value'), plt.xlabel('Affinity')
# plt.subplots_adjust(bottom=0.1), plt.show()

# TODO: compute normalized laplacian. Generate eigenvalues (a function of sigma now)


# TODO: Do the sigma sweep for each cluster number. (eigen-gap vs sigma curve)


if __name__ == "__main__":
    main()
