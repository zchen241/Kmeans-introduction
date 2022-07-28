import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def centeroidnp(arr): #this is support 2 dimensions data set
    arr = np.asarray(arr)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    centroid = [sum_x/length, sum_y/length]
    return centroid

def checkifpointinlist(p, plist):
    flag = False
    for i in plist:
        if p == i:
            flag = True
            break

    return flag

def get_init_centroids(k, datapoints):
    centroidlist = []
    centroid = centeroidnp(datapoints)

    centroidlist.append(centroid)
    #find k-1 points which have maximum distance with the centroid

    initcen = random.choice(datapoints)
    for i in range(1, k):
        maxd = 0
        dist = 0

        for p in datapoints:
            flag = checkifpointinlist(p, centroidlist)
            if flag == 0:
                for c in centroidlist:
                    dist = eucldist(p, c) + dist
                if dist > maxd:
                    for c in centroidlist:
                        initcen = p
                        maxd = dist
        centroidlist.append(initcen)
    return centroidlist


def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)


def find_cluster_index(centroid, cluster_centers):
    index = 0
    for c in range(0, len(cluster_centers)):
        if c == centroid:
            return index
        index = index+1
def comparetwocentroidslist(oldlist, new_center):
    # verify centroids modified or not
    sum_x = 0
    sum_y = 0
    for ind in range(0, len(new_center)):
        sum_x = sum_x + new_center[ind][0]
        sum_y = sum_y + new_center[ind][1]

    sum_old_x = 0
    sum_old_y = 0
    for ind in range(0, len(new_center)):
        sum_old_x = sum_old_x + oldlist[ind][0]
        sum_old_y = sum_old_y + oldlist[ind][1]

    if (sum_x == sum_old_x) & (sum_y == sum_old_y):
        return True
    else:
        return False


def mykmeans(k, datapoints):
    d = len(datapoints)
    features = len(datapoints[0])
    state_p = [0] * d
    # Randomly Choose Centers for the Clusters
    cluster_centers = []
    # for i in range(0, k):
    #     cluster_centers += [random.choice(datapoints)]
    cluster_centers = get_init_centroids(k, datapoints)


    for p in range(0, len(datapoints)):
        min_dis = float("inf")
        for c in range(0, len(cluster_centers)):
            dist = eucldist(cluster_centers[c], datapoints[p])
            if dist < min_dis:
                state_p[p] = c
                min_dis = dist

    #calculate centroids based on cluseters
    centroid_feature_sum = [0] * k
    cluster_size = [0] * k
    prev_cluster = cluster_centers
    new_center = [0] * k

    while cluster_centers != prev_cluster:
        prev_cluster = cluster_centers
        # for k in range(0, len(cluster_centers)):
        #     new_center = [0] * features
        #     members = 0
        #     for p in range(0, len(datapoints)):
        #         if state_p[p] == k:
        #             for j in range(0, d):
        #                 new_center[j] += datapoints[p][j]
        #             members += 1
        #
        #     for j in range(0, features):
        #         if members != 0:
        #             new_center[j] = new_center[j] / float(members)
        for p in range(0, len(datapoints)):
            for j in range(0, features):
                new_center[state_p[p]][j] += datapoints[p][j]
            cluster_size[state_p[p]] = cluster_size[state_p[p]] + 1

        for j in range(0, features):
            new_center[j] = new_center[j] / float(cluster_size[j])

        cluster_centers = new_center


    markertypelist = ['1', '+', '|', 'o', 'x', 'v', '.']
    markercol = ['r', 'k', 'b', 'y', 'c', 'm', 'g']
    for p in range(len(datapoints)):
        xs = datapoints[p][0]
        ys = datapoints[p][1]
        plt.plot(xs, ys, c=markercol[state_p[p]], marker=markertypelist[state_p[p]], markersize=4)


    plt.xlabel('Dist')
    plt.ylabel('Speed')
    plt.savefig('test_kmeans.png')
    plt.show()

def kmeans(k,datapoints):
    # d - Dimensionality of Datapoints
    d = len(datapoints[0])

    # Limit our iterations
    Max_Iterations = 1000
    i = 0

    # prev_cluster = [k][0]
    force_recalculation = True

    # Randomly Choose Centers for the Clusters
    cluster_centers = []
    # for i in range(0, k):
    #     cluster_centers += [random.choice(datapoints)]

    cluster_centers = get_init_centroids(datapoints, k)

    while (i > Max_Iterations) or (force_recalculation):
        cluster = []
        new_center = [-1] * len(cluster_centers)
        for c in range(0, len(cluster_centers)):
            singlelist = []
            cluster.append(singlelist)
        # assign all points to clusters
        # according to current centroid
        for p in datapoints:
            #calculate point distance to each centroid
            # and determine the shortest distance
            mindis = 10000
            cluflag = cluster_centers[0]
            for c in range(0, len(cluster_centers)):
                dis = eucldist(p, cluster_centers[c])
                if dis < mindis:
                    mindis = dis
                    cluflag = c

            #assign point to cluster
            clusterindex = find_cluster_index(cluflag, cluster_centers)
            cluster[clusterindex].append(p)

        #update centroid
        for k in range(0, len(cluster_centers)):
            new_centroid = centeroidnp(cluster[k])
            new_center[k] = new_centroid

        #verify centroids modified or not
        if new_center == cluster_centers:
            break  # centroid does not change any more that means clusters complete

        cluster_centers = new_center
        i = i+1
    print ("======== Results ========")
    print ("Clusters", new_center)
    print ("Iterations",i)
    print ("Assignments", cluster)
    return new_center, cluster


def loaddigitsfromcsv(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data


def plotdataset(datasetlist):
    markertypelist = ['1', '+', '|', 'o', 'x', 'v', '.']
    markercol = ['r', 'k', 'b', 'y', 'c', 'm', 'g']
    datasetlist = np.asarray(datasetlist)
    index = 0
    for eachset in datasetlist:
        for data in eachset:
            xs = data[0]
            ys = data[1]
            plt.plot(xs, ys, c=markercol[index], marker=markertypelist[index], markersize=4)
        index = index + 1

    plt.xlabel('Dist')
    plt.ylabel('Speed')
    plt.savefig('assignment3_kmeans.png')
    plt.show()


# TESTING THE PROGRAM#
if __name__ == "__main__":

    X = loaddigitsfromcsv('dataset.csv')  # the driverlog.csv has removed the header row
    X = X[:,1:3]
    # my student id is 300103013 so n_clusters = 4
    # centroidlist, clusterslist = kmeans(5, X.tolist())
    # plotdataset(clusterslist)
    mykmeans(3, X.tolist())