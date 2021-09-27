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


def main():
    # are we following Von Luxburg (True) or Ng and Weiss (False)?
    von_luxburg = False
    # are we just testing how eigenvalues work?

    # set the diagonal of the affinity mat to zero or not
    if von_luxburg:
        diag_zero = False
        print('computation according to von Luxburg. Diagonal elements of the Affinity matrix (A) will be 1')
    else:
        diag_zero = True
        print('computation according to Ng and Weiss. Diagonal elements of the Affinity matrix (A) will be zero')

    a = random.uniform(0, 1)
    b = random.uniform(0, 1)
    c = random.uniform(0, 1)
    d = random.uniform(0, 1)
    e = random.uniform(0, 1)
    f = random.uniform(0, 1)

    affinity = np.array([[1, a, b, 0, 0, 0],
                         [a, 1, c, 0, 0, 0],
                         [b, c, 1, 0, 0, 0],
                         [0, 0, 0, 1, d, e],
                         [0, 0, 0, d, 1, f],
                         [0, 0, 0, e, f, 1]])
    if diag_zero:
        np.fill_diagonal(affinity, 0)

    # computing laplacian
    laplacian = get_laplacian(affinity, normalize=True, symmetry=True,
                              view=False)

    # computing eigenvalues and eigen-gaps
    eigen_gap, eigen_value, eigen_vec = get_eigen(laplacian, absolute=False, eigengap_only=False, view=False)
    print('affinity =\n' + str(np.round(affinity, 2)))
    print('laplacian =\n' + str(np.round(laplacian, 2)))
    print('eigenvalues (real part) = ' + str(np.round(np.real(eigen_value), 3)))
    print('eigenvalues (img part) = ' + str(np.round(np.imag(eigen_value), 3)))


if __name__ == "__main__":
    main()
