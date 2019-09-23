import matplotlib as mpl
from numpy import meshgrid
import numpy as np
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from intake_utils import loadDataset

def generateHistogramViz(rental):
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    hist, xedges, yedges = rental.histogram2d

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(
            111,
            title='rental heatmap',
            aspect='equal')
    x, y = meshgrid(xedges, yedges)
    ax.pcolormesh(x, y, hist.T)
    plt.show()


def generateRentalPathViz(rental, title):
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    filename = 'rental_{}.png'.format(rental._id)

    df = DataFrame.from_records(rental.positions, columns=rental.positionColumnNames)

    plt.title(title)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.plot(df.lng, df.lat, marker='x')
    plt.savefig(filename)
    plt.gcf().clear()
    plt.close()
