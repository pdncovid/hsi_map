import numpy as np


def min_max_normalise(data_init, axis='all'):
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


def tile(data_, rows_, cols_):
    y_size = data_.shape[0] // rows_
    x_size = data_.shape[1] // cols_
    tiles_ = np.zeros((rows_ * cols_, y_size, x_size))
    sample = 0
    for x in range(cols_):
        for y in range(rows_):
            x_ = x * x_size
            y_ = y * y_size
            temp_ = data_[y_:y_ + y_size, x_:x_ + x_size]
            tiles_[sample, :, :] = temp_
            sample += 1
    return tiles_


def img_sample(data_initial, par, stacked=False):
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

    if stacked:
        data_final = []
        for i in range(data_initial.shape[0]):
            data_final.append(tile(data_initial[i, :, :], rows, cols))
        data_final = np.array(data_final)
    else:
        data_final = tile(data_initial, rows, cols)

    return data_final
