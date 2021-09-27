import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def get_distance(mat, normalize=True, view=True):
    if normalize:
        for dim in range(mat.shape[0]):
            mat[dim, :, :] = (mat[dim, :, :] - np.amin(mat[dim, :, :])) / (
                    np.amax(mat[dim, :, :]) - np.amin(mat[dim, :, :]))

    # need to flatten the matrix first.
    mat = np.reshape(mat, [8, -1])

    distance_mat = np.zeros([mat.shape[-1], mat.shape[-1]])

    for var1 in range(mat.shape[-1]):
        for var2 in range(mat.shape[-1]):
            distance_mat[var1, var2] = np.sum((mat[:, var1] - mat[:, var2]) ** 2) ** 0.5

    if view:
        plt.figure(figsize=(4, 4))
        plt.imshow(distance_mat)
        plt.title('euclidean distance matrix')
        plt.colorbar()
        plt.show()

    return distance_mat


def get_affinity(mat, sig, diag_zero=True, compute_dist=False, view_mat=True, view_hist=True):
    print('computing affinity matrix for \u03C3 = ' + str(round(sig, 2)))
    if compute_dist:
        dist_mat = get_distance(mat, normalize=True, view=False)
    else:
        dist_mat = np.copy(mat)

    affinity_mat = np.exp(-1 * (dist_mat ** 2) / (2 * (sig ** 2)))

    if diag_zero:
        affinity_mat = affinity_mat - np.identity(affinity_mat.shape[0])

    if view_mat:
        plt.figure(figsize=(8, 4))
        plt.subplot(121), plt.imshow(dist_mat), plt.colorbar()
        plt.title('euclidean distance matrix')
        plt.subplot(122), plt.imshow(affinity_mat), plt.colorbar()
        plt.title('affinity matrix with \u03C3 = ' + str(round(sig, 2)))
        plt.show()

    if view_hist:
        affinity_temp = np.reshape(affinity_mat, (-1))
        plt.figure(figsize=(6, 4))
        sns.displot(affinity_temp, kind='kde')
        plt.show()

    return affinity_mat


def get_laplacian(mat, normalize=True, symmetry=True, view=True):
    print('computing laplacian..')

    # creating degree matrix. sum of values in each row of affinity matrix.
    degree = np.zeros_like(mat)
    for i in range(degree.shape[0]):
        degree[i, i] = np.sum(mat[i, :])

    # print('degree =\n' + str(np.round(degree, 2)))

    if normalize:
        if symmetry:
            degree_sq = (np.linalg.inv(degree)) ** 0.5
            # print('degree inverse sqrt =\n' + str(np.round(degree_sq, 2)))
            laplacian = np.matmul(np.matmul(degree_sq, mat), degree_sq)
        else:
            degree_sq = (np.linalg.inv(degree))
            laplacian = np.identity(mat.shape[0]) - np.matmul(degree_sq, mat)
    else:
        laplacian = degree - mat
    if view:
        plots, names = np.array([laplacian, degree]), ['Laplacian\nnormalized = ' + str(normalize), 'degree matrix']
        plt.figure(figsize=(4 * plots.shape[0], 4))
        for i in range(plots.shape[0]):
            plt.subplot(1, 3, i + 1), plt.title(names[i]), plt.imshow(plots[i, :, :])
        plt.show()
    return laplacian


def get_eigen(mat, sig=None, absolute=True, eigengap_only=False, view=True):
    [eig_val, eig_vec] = np.linalg.eig(mat)
    if absolute:
        eig_val = np.abs(eig_val)
        eigen_gap = np.abs(np.diff(eig_val))

    else:
        eigen_gap = np.diff(eig_val)

    if view:
        plt.figure(figsize=(12, 4))
        plt.subplot(121), plt.stem(eig_val), plt.xlim([-1, 10]), plt.ylim([0, 1])
        plt.title('eigen-values for \u03C3 = ' + str(round(sig, 2)))
        plt.xlabel('eigen-index'), plt.ylabel('eigen-value')
        plt.subplot(122), plt.stem(eigen_gap), plt.xlim([-1, 10]), plt.ylim([0, 1])
        plt.title('eigen-gap for \u03C3 = ' + str(round(sig, 2)))
        plt.xlabel('eigen-index'), plt.ylabel('eigen-gap')
        plt.show()

    if eigengap_only:
        return eigen_gap
    else:
        return eigen_gap, eig_val, eig_vec


def sigma_sweep(test_sample, k, cluster=False, sig_range=None, von_luxburg=False, abs_eig=True, view_sample=False,
                view_sig=False):
    """In this function, we set 'k' which is the amount of clusters we need, and find the
    OPTIMAL SIGMA value for that k, where the sigma corresponding to the LARGEST Kth EIGEN-GAP wins.
    It makes the most sense compared to just going over everything for now because in some cases
    we get 1--2 clusters as optimal for sigma values."""

    if sig_range is None:
        sig_range = [0.1, 0.3]

    # set the diagonal of the affinity mat to zero or not
    if von_luxburg:
        diag_zero = False
        print('computation according to Von Luxburg. Diagonal elements of the Affinity matrix (A) will be 1')
    else:
        diag_zero = True
        print('computation according to Ng and Weiss. Diagonal elements of the Affinity matrix (A) will be zero')

    # some initializations for later analysis
    eig_gaps, sig_labels, cluster_labels = [], [], []

    sig_list = np.linspace(sig_range[0], sig_range[1], 10)

    """ sigma sweep starts here """
    for num in range(len(sig_list)):
        sig = sig_list[num]
        affinity = get_affinity(test_sample, sig, diag_zero=diag_zero, compute_dist=True, view_mat=False,
                                view_hist=False)
        # computing laplacian
        laplacian = get_laplacian(affinity, normalize=True, symmetry=True,
                                  view=False)

        # computing eigenvalues and eigen-gaps
        # TODO: sort the eigenvectors by largest magnitude of eigenvalues
        eigen_gap, eigen_val, eigen_vec = get_eigen(laplacian, sig, absolute=abs_eig, eigengap_only=False, view=False)
        eig_gaps.append(eigen_gap[k])

        if cluster:
            # extracting the 'k' eigenvectors corresponding to the optimal mode k
            mode_x = eigen_vec[:, 0:k]
            mode_y = np.zeros_like(mode_x)
            for row in range(mode_x.shape[0]):
                mode_y[row, :] = mode_x[row, :] / (np.sum(mode_x[row, :] ** 2) ** 0.5)

            k_mean = KMeans(n_clusters=k)
            k_mean.fit(mode_y)

            targets = k_mean.labels_.reshape(-1)
            one_hot_targets = np.eye(k)[targets].transpose()
            one_hot_rank = np.sum(one_hot_targets, axis=-1)
            order = one_hot_rank.argsort()
            ranks = order.argsort()

            one_hot_edited = np.zeros_like(one_hot_targets)
            for i in range(one_hot_targets.shape[0]):
                one_hot_edited[i, :] = one_hot_targets[ranks[i], :]

            one_hot_2d = np.reshape(one_hot_edited,
                                    (one_hot_targets.shape[0], test_sample.shape[-2], test_sample.shape[-1]))

            cols = 4
            rows = math.ceil(k / cols)
            plt.figure(figsize=(cols * 4, rows * 4))
            for i in range(k):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(one_hot_2d[i, :, :])
                plt.title('unsupervised cluster label = ' + str(i + 1))
            plt.suptitle('unsupervised clusters for k=' + str(k) + ' and \u03C3=' + str(round(sig, 2)))
            plt.show()

    """sigma sweep completed >>>"""

    sig_opt = sig_list[np.argmax(eig_gaps)]
    print('optimal \u03C3 value for (k=' + str(k) + ') is: ' + str(round(sig_opt, 3)))
    print('the corresponding eigen-gap value is: ' + str(round(np.amax(eig_gaps), 3)))

    if view_sample:
        col = 4
        row = math.ceil(test_sample.shape[0] / col)
        plt.figure(figsize=(col * 4, row * 4))
        for i in range(test_sample.shape[0]):
            plt.subplot(row, col, i + 1)
            plt.imshow(test_sample[i, :, :])
        plt.show()

    if view_sig:
        plt.figure(figsize=(6, 4))
        plt.stem(sig_list, eig_gaps)
        plt.ylabel('eigen-gap for k = ' + str(k)), plt.xlabel('\u03C3')
        plt.show()

    return sig_opt
