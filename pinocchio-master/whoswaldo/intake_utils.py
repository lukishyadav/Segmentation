import csv
from datetime import datetime
from os import remove
from pathlib import Path
from pandas import read_csv
from recordclass import recordclass

from calculation_utils import calculateNonZeroMean, calculateNonZeroStDev
from validation_utils import filterRentals
from rental import Rental
from settings import ANALYSIS_NONSCORE_ROWS, ANALYSIS_FEATURES


# make new rental objects based on dataset
def ingestRentalRecords(filename):
    """
    dataset: list of named tuples - RentalRecord

    organize raw lat/lng csv data and organize it to be used for
    position data for a rental
    """

    dataset = []

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        headerRow = next(reader)

        RentalRecord = recordclass('RentalRecord', headerRow)

        for data in map(RentalRecord._make, reader):
            dataset.append(data)

    currRental = Rental(None)
    rentalList = {}
    Position = recordclass('PositionRecord', dataset[0]._fields)

    # lines are time, lat, lng, rental_id
    for line in dataset:

        # if there's a change in the rental id, create a new rental object
        if currRental._id != line.rental_id:
            currRental = Rental(line.rental_id)
            rentalList[str(line.rental_id)] = currRental

        position = Position(
                datetime.strptime(line.time, '%Y-%m-%dT%H:%M:%S.%f%z'),
                # float(line.time),
                float(line.lat),
                float(line.lng),
                line.rental_id,
                float(line.mileageTotal)
                )

        currRental.positions.append(position)

    return rentalList


def digestData(rental, classification=''):
    """
    derive calculated data from rental position data
    and output line to csv,
    providing classification (if any) and rental id
    """

    hist, _, _ = rental.histogram2d
    mean = calculateNonZeroMean(hist.tolist())
    stdev = calculateNonZeroStDev(hist.tolist())

    outdata = (rental.duration, rental.stdevGeoPositions,
               rental.distTraveled,
               rental.farthestDistance, rental.startEndDist,
               mean, stdev,
               classification, rental._id,)
    return outdata


def loadDataset(filename):
    """
    parses csv of classified data for use
    by classifier
    """

    dataset = read_csv(filename)
    feature_names = dataset.columns.values.tolist()[:-ANALYSIS_NONSCORE_ROWS]

    feature_scores = dataset.values[:,0:len(feature_names)]
    classifications = dataset.values[:,len(feature_names)]
    ids = dataset.values[:,len(feature_names) + 1]

    return feature_scores, classifications, ids


def classifyData(filename, outputFile, writeMode, classification):
    """
    given a file and classification,
    write data to another file which can be used by a classifier
    """

    # TODO: create interface class if project expands
    dataset = ingestRentalRecords(filename)
    rentals, filtered = filterRentals(dataset.values())

    with open(outputFile, writeMode) as outfile:
        csvwriter = csv.writer(outfile)

        for r in rentals:
            outdata = digestData(r, classification=classification)
            csvwriter.writerow(outdata)

    return rentals, filtered

def generateClassifiedData(files, classifications, outputFile, header=False):
    """
    master method to create a file that has classified data
    even if the classification is a stand-in
    """

    if header:
        with open(outputFile, 'w+') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(ANALYSIS_FEATURES)
    else:
        if Path(outputFile).exists():
            remove(outputFile)

    for i in range(len(files)):
        try:
            rentals, filtered = classifyData(files[i], outputFile, 'a+',
                                             classification=classifications[i])
        except IndexError:
            print('Num files must equal num classifications!')
            return

    return rentals, filtered
