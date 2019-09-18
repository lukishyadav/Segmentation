from datetime import datetime
from calculation_utils import haversine_dist, createBoundingBox

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from recordclass import recordclass

class Rental(object):
    """
    for us, a rental is defined as a series of
    time and positions.
    """

    positionColumnNames = ('time', 'lat', 'lng', 'rentalId', 'mileageTotal',)

    def __init__(self, _id):
        self._id = _id
        self.positions = []

    @property
    def duration(self):
        return (self.positions[-1].time - self.positions[0].time).total_seconds()

    @property
    def stdevGeoPositions(self):
        """
        calculate the stdev of all points from the origin
        we don't care about the actual distance from (0,0)
        only how difference each point is from the other
        """
        # get all distance from origin of all values in list
        origin = (0.000000, 0.000000,)

        distFromOriginList = []
        for pos in self.positions:
            distFromOriginList.append(
                    haversine_dist(origin, (pos.lat, pos.lng,)))

        # get stdev
        return np.std(distFromOriginList)

    @property
    def startPosition(self):
        return (self.positions[0].lat, self.positions[0].lng,)

    @property
    def endPosition(self):
        return (self.positions[-1].lat, self.positions[-1].lng,)

    @property
    def startEndDist(self):
        """
        distance between the start and the end positions
        of the rental
        returns 1m if rental is in the exact same lat/lng position
        at beginning and end
        """
        diff = haversine_dist(self.startPosition, self.endPosition)
        if diff == 0:
            return 1
        else:
            return diff

    @property
    def distTraveled(self):
        diff = (self.positions[-1].mileageTotal -
                self.positions[0].mileageTotal)
        return diff

    @property
    def farthestDistance(self):
        """
        determines the farthest points of the rental
        """
        # build convex hull
        geopositions = [[x.lat, x.lng] for x in self.positions]
        ndarray = np.asarray(geopositions, dtype=float)

        try:
            hull = ConvexHull(ndarray)

            # get farthest positions
            farthestDistance = 0
            hullPoints = hull.vertices.tolist()
            for pointsA in hullPoints:
                positionA = geopositions[pointsA]
                for pointsB in hullPoints:
                    positionB = geopositions[pointsB]

                    calcDistance = haversine_dist(positionA, positionB)
                    if farthestDistance < calcDistance:
                        farthestDistance = calcDistance

        except :
            return 0

        return farthestDistance

    @property
    def noDuplicateRecords(self, driftCoefficient=0):
        """
        iterates through positions, removing duplicate lat/lng in a row
        do we account for drift? (via driftCoefficient)
        """
        outPositions = []
        currLat = None
        currLng = None
        for pos in self.positions:
            # same position
            # if pos.lat != currLat or pos.lng != currLng:

            # new position is less than 10 meters away
            # this accounts for drift
            if ((currLat == None or currLng == None) or
                haversine_dist(
                        (currLat, currLng),
                        (pos.lat, pos.lng)) > 10):
                currLat = pos.lat
                currLng = pos.lng
                outPositions.append(pos)
        return outPositions

    @property
    def boundingBox(self):
        coordinates = recordclass('Coordinates', 'x, y')
        coordinatesList = []
        for pos in self.positions:
            newCoordinate = coordinates(pos.lng, pos.lat)
            coordinatesList.append(newCoordinate)

        return createBoundingBox(coordinatesList)

    @property
    def hasValidBox(self):
        return (self.boundingBox['top'] != self.boundingBox['bottom']
                and self.boundingBox['left'] != self.boundingBox['right'])

    @property
    def histogram2d(self):
        """
        precision: sigfigs past the decimal to be used for number of bins
        """
        bBox = self.boundingBox
        # calculate number of bins as a factor of size?
        # start with factor of 100, increase resolution if necessary to 5
        xBins, yBins = 0, 0
        xFactor, yFactor = 2, 2
        while xBins == 0:
            xBins = int(abs((bBox['right'] - bBox['left'])*(10**xFactor)))
            xFactor += 1

        while yBins == 0:
            yBins = int(abs((bBox['top'] - bBox['bottom'])*(10**yFactor)))
            yFactor += 1

        # get corrected records
        xpoints = [pos.lng for pos in self.noDuplicateRecords]
        ypoints = [pos.lat for pos in self.noDuplicateRecords]

        # generate 2d histogram
        return np.histogram2d(xpoints, ypoints, bins=[xBins, yBins],
                              range=[
                                  [bBox['left'], bBox['right']],
                                  [bBox['bottom'], bBox['top']]
                                  ])
