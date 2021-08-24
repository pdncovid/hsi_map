from pymatreader import read_mat
from matplotlib import pyplot as plt
import numpy as np
import functions

data = read_mat("refImage.mat")
"""How are these points obtained? 
Please read them through a .csv file such that one can edit it anytime"""
points = [80.602, 7.332, 80.673, 7.254]

# functions.map_gps_crd(data, points)
"""please try to generalise these functions as much as possible. 
Send 'places, refImage, nir2' as kwargs. 
Then if one wants to map something else they could do it as well.
Otherwise we'd need to have another function for 'CoastalBlue' etc."""
functions.map_places(data, points)
# kandy = functions.geopandas_kandy()



