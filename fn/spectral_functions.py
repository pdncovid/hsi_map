import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

from sklearn.cluster import KMeans
from scipy.fftpack import fft, dctn, idctn

from fn.data_functions import convert_one_hot
from fn.map_functions import plot_sample


def get_affinity(mat, sig, diag_zero=True, compute_dist=True, view_mat=False, view_hist=False):
    def get_distance(mat_, normalize=True):
        if normalize:
            for dim in range(mat_.shape[0]):
                mat_[dim, :, :] = (mat_[dim, :, :] - np.amin(mat_[dim, :, :])) / (
                        np.amax(mat_[dim, :, :]) - np.amin(mat_[dim, :, :]))

        # need to flatten the matrix first.
        mat_ = np.reshape(mat_, [8, -1])

        distance_mat = np.zeros([mat_.shape[-1], mat_.shape[-1]])

        for var1 in range(mat_.shape[-1]):
            for var2 in range(mat_.shape[-1]):
                distance_mat[var1, var2] = np.sum((mat_[:, var1] - mat_[:, var2]) ** 2) ** 0.5

        return distance_mat

    print('computing affinity matrix for \u03C3 = ' + str(round(sig, 2)))
    if compute_dist:
        dist_mat = get_distance(mat, normalize=True)
    else:
        dist_mat = np.copy(mat)

    affinity_mat = np.exp(-1 * (dist_mat ** 2) / (2 * (sig ** 2)))

    if diag_zero:
        np.fill_diagonal(affinity_mat, 0)

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


def get_laplacian(mat, normalize=True, symmetry=True, view=False):
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


def get_eigen(mat, sig=None, absolute=True, eigengap_only=False, view=False):
    print('get eigen...')

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


def sigma_optimum(test_sample, k, sig_range=None, cluster=False, method='Weiss', abs_eig=True, view_sample=False,
                  view_sig=False, view_clusters=False):
    """In this function, we set 'k' which is the amount of clusters we need, and find the
    OPTIMAL SIGMA value for that k, where the sigma corresponding to the LARGEST Kth EIGEN-GAP wins.
    It makes the most sense compared to just going over everything for now because in some cases
    we get 1--2 clusters as optimal for sigma values."""
    print('sigma sweep function...')

    def sigma_sweep(test_sample_, sig_range_, method_, abs_eig_):
        # set the diagonal of the affinity mat to zero or not
        if 'weiss' in method_.lower() or 'ng' in method_.lower():
            diag_zero_ = True
            print('computation according to Ng and Weiss. Diagonal elements of the Affinity matrix (A) will be zero')
        elif 'von' in method_.lower() or 'luxburg' in method_.lower():
            diag_zero_ = False
            print('computation according to Von Luxburg. Diagonal elements of the Affinity matrix (A) will be 1')
        else:
            diag_zero_ = True
            print('Cannot detect method.\ncomputation according to Ng and Weiss. Diagonal elements of the Affinity '
                  'matrix (A) will be zero')

        # some initializations for later analysis
        eig_gaps_, eig_vals_, eig_vecs_ = [], [], []

        sig_list_ = np.linspace(sig_range_[0], sig_range_[1], 10)

        """ sigma sweep starts here """
        print('sigma sweep starts here...')

        for num_ in range(len(sig_list_)):
            sig_ = sig_list_[num_]
            affinity = get_affinity(test_sample_, sig_, diag_zero=diag_zero_, compute_dist=True, view_mat=False,
                                    view_hist=False)
            # computing laplacian
            laplacian = get_laplacian(affinity, normalize=True, symmetry=True,
                                      view=False)

            # computing eigenvalues and eigen-gaps
            eigen_gap_, eigen_val_, eigen_vec_ = get_eigen(laplacian, sig_, absolute=abs_eig_, eigengap_only=False,
                                                           view=False)
            eig_gaps_.append(eigen_gap_)
            eig_vecs_.append(eigen_vec_)
            eig_vals_.append(eigen_val_)

        return np.array(eig_gaps_), np.array(eig_vecs_), np.array(eig_vals_)

    if sig_range is None:
        sig_range = [0.1, 0.3]
    sig_list = np.linspace(sig_range[0], sig_range[1], 10)

    eig_gaps, eig_vecs, eig_vals = sigma_sweep(test_sample, sig_range, method, abs_eig)
    sig_opt = sig_list[np.argmax(np.abs(eig_gaps[:, k]))]

    if cluster:
        for num in range(len(sig_list)):
            eigen_vec = eig_vecs[num]
            cluster_spectral(test_sample, k, sig_list[num], eigen_vec=eigen_vec, view_sample=False)

    print('optimal \u03C3 value for (k=' + str(k) + ') is: ' + str(round(sig_opt, 3)))
    print('the corresponding eigen-gap value is: ' + str(round(np.amax(eig_gaps), 3)))

    if view_sample:
        plot_sample(test_sample)

    if view_sig:
        plt.figure(figsize=(6, 4))
        plt.stem(sig_list, eig_gaps[:, k])
        plt.ylabel('eigen-gap for k = ' + str(k)), plt.xlabel('\u03C3')
        plt.show()

    return sig_opt


def cluster_spectral(test_sample, k, sig, eigen_vec=None, return_2d=True, method='Ng and Weiss', view_clusters=False,
                     view_sample=False):
    print('spectral clustering....')

    if 'weiss' in method.lower() or 'ng' in method.lower():
        diag_zero = True
    elif 'von' in method.lower() or 'luxburg' in method.lower():
        diag_zero = False
    else:
        diag_zero = True

    if eigen_vec is None:
        A = get_affinity(test_sample, sig, diag_zero)
        L = get_laplacian(A)
        _, _, eigen_vec = get_eigen(L, sig)
    else:
        pass

    # extracting the 'k' eigenvectors corresponding to the optimal mode k
    mode_x = eigen_vec[:, 0:k]
    mode_y = np.zeros_like(mode_x)
    for row in range(mode_x.shape[0]):
        mode_y[row, :] = mode_x[row, :] / (np.sum(mode_x[row, :] ** 2) ** 0.5)

    title = 'Algorithm: Spectral Clustering according to ' + method + '\nunsupervised clusters for k=' + str(
        k) + 'and \u03C3=' + str(round(sig, 3))

    labels, labels_one_hot = cluster_kmeans(test_sample, k, feature_mat=mode_y, return_2d=True,
                                            view_clusters=view_clusters,
                                            view_sample=view_sample, title=title)

    return labels, labels_one_hot


def cluster_kmeans(test_sample, k, feature_mat=None, return_2d=True, view_clusters=False, view_sample=False,
                   title=None):
    print('kmean clustering...')

    if feature_mat is None:
        feature_mat = np.transpose(test_sample, (1, 2, 0))
        feature_mat = np.reshape(feature_mat, (-1, feature_mat.shape[-1]))

    k_mean = KMeans(n_clusters=k)
    k_mean.fit(feature_mat)

    cluster_labels = k_mean.labels_

    one_hot_ranked = convert_one_hot(cluster_labels, k, rank=True)

    one_hot_2d = np.reshape(one_hot_ranked, (k, test_sample.shape[-2], test_sample.shape[-1]))
    labels_2d = np.reshape(cluster_labels, (test_sample.shape[-2], test_sample.shape[-1]))

    if view_sample:
        plot_sample(test_sample)

    if view_clusters:
        if title is None:
            title = 'k-means clustering where k=' + str(k)
        cols = 4
        rows = math.ceil((k + 1) / cols)
        plt.figure(figsize=(cols * 4, rows * 4))
        for i in range(k + 1):
            plt.subplot(rows, cols, i + 1)
            if i == k:
                plt.imshow(labels_2d)
                plt.title('segmented image (segments in no specific order)')
            else:
                plt.imshow(one_hot_2d[i, :, :])
                plt.title('cluster label = ' + str(i + 1))
        plt.suptitle(title)
        plt.show()

    return labels_2d, one_hot_2d


def dct_image(img1, scale_down, filter_size=8, view=False, title=None):
    print('scale down image...')

    def dct_single(img, scale, filter_=8, view_=False, title_=None):
        if title_ is None:
            title_ = str(scale_down) + 'x compressed image after DCT-II\nFilter size: ' + str(filter_)
        else:
            pass

        sample_size = math.floor(filter_ / scale_down)

        x_len = int(filter_ * math.floor(img.shape[1] / filter_))
        y_len = int(filter_ * math.floor(img.shape[0] / filter_))

        img = img[0:y_len, 0:x_len]
        samples_x = int(img.shape[1] / filter_)
        samples_y = int(img.shape[0] / filter_)

        img_new = np.zeros((samples_y * sample_size, samples_x * sample_size))

        for i in range(samples_y):
            for j in range(samples_x):
                img_seg = img[filter_ * i: filter_ * (i + 1), filter_ * j: filter_ * (j + 1)]
                dct_seg = dctn(img_seg)
                img_sampled = idctn(dct_seg, shape=(sample_size, sample_size))
                img_sampled = img_sampled * np.mean(img_seg) / np.mean(img_sampled)
                img_new[sample_size * i: sample_size * (i + 1), sample_size * j: sample_size * (j + 1)] = img_sampled

        if view_:
            plt.figure(figsize=(8, 4))
            plt.subplot(121), plt.imshow(img)
            plt.title('Original image')
            plt.clim(0, 1)
            plt.subplot(122), plt.imshow(img_new)
            plt.title(title_)
            plt.clim(0, 1)
            plt.show()

        return img_new

    if len(img1.shape) == 3:
        dct_multi = []
        for i in range(img1.shape[0]):
            dct_multi.append(dct_single(img1[i, :, :], scale_down, filter_size, view, title))
        compressed_image = np.array(dct_multi)
    else:
        compressed_image = dct_single(img1, scale_down, filter_size, view, title)

    return compressed_image
