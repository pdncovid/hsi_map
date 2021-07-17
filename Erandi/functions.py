import pandas as pd
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np


def map_gps_crd(data):
    print(data)
    points = [80.602, 7.332, 80.673, 7.254]
    x_ticks = map(lambda x: round(x, 4), np.linspace(points[0], points[2], num=10))
    y_ticks = map(lambda x: round(x, 4), np.linspace(points[3], points[1], num=11))
    y_ticks = sorted(y_ticks, reverse=True)

    fig, axis1 = plt.subplots(figsize=(10, 10))
    axis1.imshow(data["refImage"]['nir2'])
    axis1.set_xlabel('Longitude')
    axis1.set_ylabel('Latitude')
    axis1.set_xticklabels(x_ticks, rotation=70)
    axis1.set_yticklabels(y_ticks)
    axis1.grid()
    plt.show()


def scaleMinMax(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def merge(data):
    coastalBlue = data["refImage"]['coastalBlue']
    blue = data["refImage"]['blue']
    green = data["refImage"]['green']
    yellow = data["refImage"]['yellow']
    red = data["refImage"]['red']
    redEdge = data["refImage"]['redEdge']
    nir1 = data["refImage"]['nir1']
    nir2 = data["refImage"]['nir2']

    cb = scaleMinMax(coastalBlue)
    b = scaleMinMax(blue)
    g = scaleMinMax(green)
    y = scaleMinMax(yellow)
    r = scaleMinMax(red)
    re = scaleMinMax(redEdge)
    n2 = scaleMinMax(nir2)
    n1 = scaleMinMax(nir1)

    merged = np.stack((r, b, g), axis=2)

    plt.figure()
    plt.imshow(merged)
    plt.show()


def geopandas_kandy():


