from pymatreader import read_mat
from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np
import functions

data = read_mat("refImage.mat")
"""Longitude and Latitude extent of the satellite image"""
points = [80.602, 7.332, 80.673, 7.254]

kandy = functions.map_boundries_to_satellite_image(data, points)

# functions.map_gps_crd(data, points)
# kandy = functions.geopandas_kandy()
# functions.map_gps_crd(data, points)



