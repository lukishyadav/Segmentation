import csv
import math
import numpy as np
from recordclass import recordclass

EARTH_RADIUS = 6371000


def haversine_dist(start, end):
    """
    Finds the great circle distance between two lat,lng coord pairs in meters.
    This function uses haversine distance forumla, more info can be found here,
    http://en.wikipedia.org/wiki/Haversine_formula
    @params: start, end - [lat, lng] of the start and end points. Values are in float.
    @return: Distance in meters between the points.
    """
    start_lat, start_lng = start
    end_lat, end_lng = end

    dlat = math.radians(float(end_lat) - float(start_lat))
    dlon = math.radians(float(end_lng) - float(start_lng))

    dist = (math.sin(dlat / 2) * math.sin(dlat / 2)
            + math.cos(math.radians(start_lat))
            * math.cos(math.radians(end_lat)) * math.sin(dlon / 2)
            * math.sin(dlon / 2))
    dist = (2 * EARTH_RADIUS * math.atan2(math.sqrt(dist),
            math.sqrt(1 - dist)))

    return dist


def createBoundingBox(points):
    """
    creates a 2d bounding box, given a list of (x, y)
    """
    box = {'top': None, 'bottom': None, 'left': None, 'right': None}

    for p in points:
        if box['top'] == None or box['top'] < p.y:
            box['top'] = p.y
        if box['bottom'] == None or box['bottom'] > p.y:
            box['bottom'] = p.y
        if box['left'] == None or box['left'] > p.x:
            box['left'] = p.x
        if box['right'] == None or box['right'] < p.x:
            box['right'] = p.x

    return box


def calculateNonZeroMean(weights):
    """
    calculates non-zero mean of given list of numbers (int/float)
    """
    return np.nanmean(np.where(np.isclose(weights, 0), np.nan, weights))


def calculateNonZeroStDev(weights):
    """
    calculates non-zero stdev of given list of numbers (int/float)
    """
    return np.nanstd(np.where(np.isclose(weights, 0), np.nan, weights))
