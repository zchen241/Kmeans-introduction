from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def loaddigitsfromcsv(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

def plotdataset(trainset, labels, testset, predictresult):
    markertypelist = ['1', '+', '|', 'o']
    markercol = ['r', 'k', 'b', 'y']
    for data, la in zip(trainset, labels):
        xs = data[0]
        ys = data[1]
        plt.plot(xs, ys, c=markercol[la], marker=markertypelist[la], markersize=4)
    for data, la in zip(testset, predictresult):
        xs = data[0]
        ys = data[1]
        plt.plot(xs, ys, c=markercol[la], marker=markertypelist[la], markersize=4)
    plt.xlabel('Dist')
    plt.ylabel('Speed')
    plt.savefig('assignment3_kmeans_library.png')
    plt.show()

def runkmens():
    X = loaddigitsfromcsv('dataset.csv') #the dataset.csv has removed the header row
    # 80% data for training and 20% for testing
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
    X_train = X_train[:, 1:3]
    X_test = X_test[:, 1:3]
    # my student id is 300103013 so n_clusters = 4
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X_train)
    labels = kmeans.labels_
    predict = kmeans.predict(X_test)
    plotdataset(X_train, labels, X_test, predict)
    centers = kmeans.cluster_centers_

runkmens()
