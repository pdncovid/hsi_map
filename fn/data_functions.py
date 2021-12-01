import numpy as np
import matplotlib.pyplot as plt
import random
import math


def min_max_normalise(data_init, axis='all'):
    """0-1 normalises each 2D array in the stacked array
    input and output shapes are the same."""

    def f__(data_, axis_):
        if not axis_ == 'all':
            for k in range(data_.shape[axis_]):  # TODO this part does not work. use axis='all' for now
                data_[k, :] = 1234  # how to choose between this
                data_[:, k] = 1234  # and this
        else:
            data_ = (data_ - np.min(data_)) / (np.max(data_) - np.min(data_))
        return data_

    data = data_init.copy()

    if type(data) == dict:
        fields = list(data.keys())
        for i in range(len(fields)):
            data[fields[i]] = f__(data[fields[i]], axis)
    else:
        data = f__(data, axis)

    return data


def plot_samples(data, title, num=10, cols=5):
    """Plots samples (each tile). Not all tiles are plotted cause its a lot of tiles then."""

    def plot_img(data_, title_, num_, cols_):
        if num_ > data_.shape[-1]:
            print('Error: chosen number of plotting samples greater than actual samples! '
                  '\nChoosing the max number of samples')
            num_ = data_.shape[-1]
        elif num_ < cols_:
            print('Error: chosen number of plotting samples lesser than columns! '
                  '\nChoosing the column number as samples')
            num_ = cols_
        else:
            pass

        rows_ = math.ceil(num_ / cols_)
        idx_ = random.sample(range(data_.shape[-1]), num_)
        plt.figure(figsize=(cols_ * 3, rows_ * 3))
        for k in range(len(idx_)):
            plt.subplot(rows_, cols_, k + 1)
            plt.imshow(data_[:, :, idx_[k]])
        plt.suptitle(title_)
        plt.show()

        return

    stacked_ = True if len(data.shape) == 4 else False
    if stacked_:
        for i in range(data.shape[0]):
            data_plot = data[i, :, :, :]
            plot_img(data_plot, title, num, cols)
    else:
        data_plot = np.copy(data)
        plot_img(data_plot, title, num, cols)
    return


def tile(data_, rows_, cols_):
    """Gets tiles (smaller 2D arrays) from a large 2D array according to no. of rows and columns.
    Similar to cutting a large square cake into equal sized pieces.
    input shape = (image shape)
    output shape = (sample shape) * no. of samples"""
    data__ = np.copy(data_)
    y_size = data__.shape[0] // rows_
    x_size = data__.shape[1] // cols_
    samples = rows_ * cols_
    tiles_ = np.zeros((y_size, x_size, samples))
    sample = 0
    for x in range(cols_):
        for y in range(rows_):
            x_ = x * x_size
            y_ = y * y_size
            temp_ = data__[y_:y_ + y_size, x_:x_ + x_size]
            tiles_[:, :, sample] = temp_
            sample += 1
    return tiles_


def samples_get(data_initial, par, view=False, print_=False):
    """Obtains samples from the original dataset.
    Number of samples is defined by par which is rows and columns (how to divide the big image to sample).
    input shape = stack size * (image shape)
    output shape = stack size * (sample shape) * no. of samples
    stack size may be None if only 1 map (frequency band) is passed as input."""
    print('get samples...')

    if type(par) is int:
        rows, cols = par, par
    elif type(par) is list or type(par) is np.ndarray:
        if len(par) == 1:
            print('Input should be listed as [rows, cols] not length ' + str(len(par)) +
                  '\nUsing the first element of the input and equal ratio')
            rows, cols = par[0], par[0]
        else:
            if len(par) > 2:
                print('Input should be listed as [rows, cols] not length ' + str(len(par)) +
                      '\nUsing the first two elements of the input')
            rows, cols = par[0], par[1]
    else:
        print('input unsupported!')
        return

    stacked = True if len(data_initial.shape) == 3 else False

    if stacked:
        data_final = []
        for i in range(data_initial.shape[0]):
            data_final.append(tile(data_initial[i, :, :], rows, cols))
        data_final = np.array(data_final)
    else:
        data_final = tile(data_initial, rows, cols)

    if print_:
        print('Creating samples from original image..')
        print('original data shape: ', data_initial.shape)
        print('sampled data shape: ', data_final.shape)

    if view:
        title = 'initially obtained samples (contains full zero samples)'
        plot_samples(data_final, title, 30, 5)

    return data_final


def samples_zeros(data, view=False, print_=False):
    """Function to remove full-zero samples from the sampled dataset.(retains only the A-shaped kandy ones)
    input shape = stack size * (sample shape) * no.samples before removing zeros
    output shape = stack size * (sample shape) * no.samples after removing zeros.
    stack size may be None if only 1 map (frequency band) is passed as input."""
    print('remove full zero samples...')

    stacked = True if len(data.shape) == 4 else False
    data_ = np.copy(data)
    zero_ = []
    for i in range(data_.shape[-1]):
        if stacked:
            zero_.append(np.all(data_[:, :, :, i] == 0))
        else:
            zero_.append(np.all(data_[:, :, i] == 0))
    zero_idx = np.reshape(np.array(np.where(zero_)), (-1))
    data__ = np.delete(data_, zero_idx, axis=-1)
    if print_:
        print('Removing zero samples..')
        print('zero samples removed: ' + str(len(zero_idx)) +
              '\nnew shape: ' + str(data__.shape))

    if view:
        title = 'reduced samples (removed zeros)'
        plot_samples(data__, title, 30, 5)

    return data__, zero_idx


def convert_one_hot(labels, k, rank=True):
    print('convert_one_hot...')

    targets = labels.reshape(-1)
    one_hot_ = np.eye(k)[targets].transpose()
    if not rank:
        return one_hot_
    else:
        temp = np.sum(one_hot_, axis=-1)
        order = temp.argsort()
        ranks = order.argsort()
        one_hot_ranked = np.zeros_like(one_hot_)
        for i in range(one_hot_.shape[0]):
            one_hot_ranked[i, :] = one_hot_[ranks[i], :]
        return one_hot_ranked
