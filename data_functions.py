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
