from analysis_utils import testClassifiers
from intake_utils import generateClassifiedData, ingestRentalRecords, loadDataset
from visualization_utils import generateRentalPathViz
from validation_utils import findDuplicateIndices
from settings import NSR_PERCENT_WALDO, SR, NSR, ANALYSIS_FEATURES, SAMPLE_SIZE

from collections import Counter
import csv
from numpy.random import choice
import numpy as np


def predictDataset(trainingFile, candidateFile, classifier):
    customer_id = candidateFile[:-13]

    # load training file
    trainingScores, trainingClasses, trainingIds = loadDataset(trainingFile)

    # candidateClassifiedData = 'classifier_' + candidateFile
    classifiedFilename = customer_id + '_classified_data.csv'
    _, filtered = generateClassifiedData([candidateFile], [''],
                           classifiedFilename, header=True)

    # load candidate file
    candidateScores, _, candidateIds = loadDataset(
            classifiedFilename)

    # remove training data that is in the candidate data
    trainingDupIdIndices, _  = findDuplicateIndices(trainingIds, candidateIds)
    np.delete(trainingScores, trainingDupIdIndices, 0)
    np.delete(trainingClasses, trainingDupIdIndices, 0)
    np.delete(trainingIds, trainingDupIdIndices, 0)

    # train the computer
    Classifier = classifier()
    Classifier.fit(trainingScores, trainingClasses)
    predictions = Classifier.predict(candidateScores)

    # resultsFile = 'results_' + candidateFile
    resultsFile = customer_id + '_results.csv'
    with open(resultsFile, 'w+') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(ANALYSIS_FEATURES)

        for i in range(len(predictions)):
            output = candidateScores[i].tolist()
            output.append(predictions[i])
            output.append(candidateIds[i])
            csvwriter.writerow(output)

    results = dict(Counter(predictions))
    results['invalid'] = len(filtered)
    results['training data'] = len(trainingDupIdIndices)
    return results


def findWaldo(dataset):
    percent_nsr = dataset[NSR]/(dataset[SR] + dataset[NSR])
    if percent_nsr >= NSR_PERCENT_WALDO:
        print('User is a Waldo')
    else:
        print('User is NOT a Waldo')


def generateResultsViz(customerId):
    """
    finds the user's files
    collects random 3 standard and 3 nonstandard rentals from the customer's results
    finds that rental data in the raw data from the customer
    generates visualizations for the collected rentals
    """
    resultsFile = 'customer_{}_results.csv'.format(customerId)
    rawData = 'customer_{}_raw_data.csv'.format(customerId)

    # organize results data to lists of standard and nonstandard rentals
    rentalList = ingestRentalRecords(rawData)
    _, classifications, rentalIds = loadDataset(resultsFile)

    # sort the rentals into standard and nonstandard
    size = len(classifications)
    standardRentals = []
    nonStandardRentals = []
    for i in range(size):
        if classifications[i] == SR:
            standardRentals.append(rentalIds[i])
        else:
            nonStandardRentals.append(rentalIds[i])

    # get random rental samples
    standardSamples = choice(standardRentals, size=SAMPLE_SIZE) if standardRentals else []
    nonStandardSamples = choice(nonStandardRentals, size=SAMPLE_SIZE) if nonStandardRentals else []

    # visualize the rentals
    for rId in standardSamples:
        rental = rentalList[str(rId)]
        title = 'Customer ID: {}, Rental ID: {}\nDuration: {:0.2f} minutes; Standard'.format(customerId, rId, rental.duration/60)
        generateRentalPathViz(rental, title)

    for rId in nonStandardSamples:
        rental = rentalList[str(rId)]
        title = 'Customer ID: {}, Rental ID: {}\nDuration: {:0.2f} minutes; Non-Standard'.format(customerId, rId, rental.duration/60)
        generateRentalPathViz(rental, title)

