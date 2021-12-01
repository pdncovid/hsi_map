import pandas as pd
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np


def map_gps_crd(data, points):
    """please try to generalise these functions as much as possible.
    Send 'places, refImage, nir2' as kwargs.
    Then if one wants to map something else they could do it as well.
    Otherwise we'd need to have another function for 'CoastalBlue' etc."""
    plt.imshow(data["refImage"]['nir2'], extent=[points[0], points[2], points[3], points[1]])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.show()


def map_boundries_to_satellite_image(data, points):
    """please try to generalise these functions as much as possible.
    Send 'places, refImage, nir2' as kwargs.
    Then if one wants to map something else they could do it as well.
    Otherwise we'd need to have another function for 'CoastalBlue' etc."""

    "read data from shape file and create a geopandas dataframe"
    gpd_df = gpd.read_file(r'C:\Users\Erandi\Documents\GitHub\hsi_map\Erandi\kandy_projected\kandy.shp')
    "select gangawata korale area to plot GN boundries on the satellite image"
    gpd_df[gpd_df.DSD_N == 'GANGAWATA KORALE'].plot(facecolor='none')

    "plot salettile image on the same axis"
    plt.imshow(data["refImage"]['nir2'], extent=[points[0], points[2], points[3], points[1]])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(points[0], points[2])
    plt.ylim(points[3], points[1])
    plt.show()

    return gpd_df


def map_risk_areas(data, points):
    """please try to generalise these functions as much as possible.
    Send 'places, refImage, nir2' as kwargs.
    Then if one wants to map something else they could do it as well.
    Otherwise we'd need to have another function for 'CoastalBlue' etc."""

    places = pd.read_csv('highly_affected_areas.csv')
    # ar = np.empty_like(data["refImage"]['nir2'])
    # plt.imshow(ar)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(points[0], points[2])
    plt.ylim(points[3], points[1])
    plt.plot(places['Longitude'], places['Latitude'], 'ro')
    plt.grid()
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
    gpd_df = gpd.read_file(r'C:\Users\Erandi\Documents\GitHub\hsi_map\Erandi\kandy_projected\kandy.shp')
    # df = pd.read_csv(r'C:\Users\Erandi\Documents\GitHub\hsi_map\Erandi\area_of_interest_details.csv')
    # gpd_df = gpd.read_file(r'C:\Users\Erandi\Documents\Shape files\kandy.shp')

    gpd_df.plot()
    # world = gpd_df.to_crs('EPSG:4326')
    # world.plot()
    # world.to_file('kandy.shp')

    plt.show()

    return gpd_df
