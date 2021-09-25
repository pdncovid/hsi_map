import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math


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


def get_laplacian(mat, tutorial=True, normalize=True, degree_int=True, symmetry=True, view=True):
    print('computing laplacian..')

    # creating degree matrix. sum of values in each row of affinity matrix.
    degree = np.zeros_like(mat)
    for i in range(degree.shape[0]):
        degree[i, i] = np.sum(mat[i, :])

    if normalize:
        if symmetry:
            degree_sq = (np.linalg.inv(degree)) ** 0.5
            """ LOOK HERE TO SEE THE COMPUTATION """
            if tutorial:
                laplacian = np.identity(mat.shape[0]) - np.matmul(np.matmul(degree_sq, mat), degree_sq)
            else:
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


def get_eigen(mat, sig, eigengap_only=False, view=True):
    [eig_val, eig_vec] = np.linalg.eig(mat)
    # eig_val = np.abs(eig_val)
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
