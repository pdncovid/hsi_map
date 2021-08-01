from pymatreader import read_mat
from matplotlib import pyplot as plt
import numpy as np
import functions

data = read_mat("refImage.mat")
points = [80.602, 7.332, 80.673, 7.254]

# functions.map_gps_crd(data, points)
functions.map_places(data, points)
# kandy = functions.geopandas_kandy()



