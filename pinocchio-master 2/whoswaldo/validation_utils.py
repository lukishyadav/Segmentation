import settings

def filterRentals(rentalList):
    valid = []
    removed = []

    for rental in rentalList:
        if (len(rental.positions) >= settings.MIN_POSITIONS
                and rental.duration >= settings.MIN_DURATION
                and rental.distTraveled >= settings.MIN_DISTANCE
                and rental.hasValidBox):
            valid.append(rental)
        else:
            removed.append(rental)

    return valid, removed


def findDuplicateIndices(list1, list2):
    """
    finds and returns the indices of duplicates
    in both lists
    returns: 2 lists representing the indices of duplicates
        in both lists
    """

    duplicates = list(set(list1).intersection(list2))
    duplicateIndices1 = []

    for i in range(len(list1)):
        if list1[i] in duplicates:
            duplicateIndices1.append(i)

    duplicateIndices2 = []
    for i in range(len(list2)):
        if list2[i] in duplicates:
            duplicateIndices2.append(i)

    return duplicateIndices1, duplicateIndices2
