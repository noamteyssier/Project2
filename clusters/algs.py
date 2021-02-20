#!/usr/bin/env python3

import numpy as np
import pandas as pd
import numba as nb
from scipy.spatial.distance import squareform
from tqdm import tqdm


@nb.jit(nopython=True, fastmath=True)
def euclidean(x, y):
    """
    calculates euclidean distance of two arrays (expects equal size arrays)

    :param x:
        1d numpy array
    :param y:
        1d numpy array

    :return: Euclidean distance
    :rtype: float
    """
    return np.sqrt(
        np.sum((y-x)**2)
        )


@nb.jit(nopython=True)
def jaccard(x, y):
    """
    calculates jaccard distance of two arrays (expects equal size arrays)

    :param x:
        1d numpy array
    :param y:
        1d numpy array

    :return: Jaccard distance
    :rtype: float
    """
    mask_x = (x > 0)
    mask_y = (y > 0)

    ix = np.sum(mask_x & mask_y)
    un = np.sum(mask_x | mask_y)
    return ix / un


@nb.jit(nopython=True)
def PairwiseIter(n):
    """
    Generator that iterates through pairwise indices over range N

    :param n:
        An integer whose range to generate unique pairwise indices

    :return:
        A generator of unique pairwise indices
    """
    for i in np.arange(n):
        for j in np.arange(i, n):
            if j > i:
                yield (i, j)


@nb.jit(nopython=True)
def PairwiseDistance(m, metric='euclidean'):
    """
    Calculates pairwisde distances of a given 2D array

    :param m:
        2D numpy array
    :param metric:
        String (euclidean / jaccard)

    :return:
        1D Condensed Distance Vector
    """
    distances = np.zeros(
        int((m.shape[0] * (m.shape[0] - 1)) / 2)
    )
    iter = 0

    path = (metric == 'euclidean')

    for (i, j) in PairwiseIter(m.shape[0]):

        if path:
            distances[iter] = euclidean(m[i], m[j])
        else:
            distances[iter] = jaccard(m[i], m[j])
        iter += 1

    return distances


def ClusterSimilarity(label_x, label_y):
    """
    Calculates similarity between two cluster labels through neighborhood
    jaccard index over all observations.

    :param label_x:
        1D numpy array of labels
    :param label_y:
        1D numpy array of labels

    :return:
        mean jaccard similarity (float)
    """

    assert label_x.size == label_y.size, "Requires Equal Sized Arrays"

    similarities = np.zeros(label_x.size)

    # iterates through observations
    for idx in np.arange(label_x.size):

        # identify labels at position idx
        lx = label_x[idx]
        ly = label_y[idx]

        # identify neighbors for each label set
        nx = set(np.flatnonzero(label_x == lx))
        ny = set(np.flatnonzero(label_y == ly))

        # calculates jaccard similarity between both sets
        similarities[idx] = len(nx.intersection(ny)) / len(nx.union(ny))

    # returns mean similarity over all observations
    return similarities.mean()


class Ligand():
    """
    Class to handle IO of ligand information

    :param fn:
        Filename of CSV to read in ligand information
    :param bitspace:
        Expected Dimensions of bitvector
    """

    def __init__(self, fn, bitspace=1024):
        self.fn = fn
        self.bitspace = bitspace

        self.load_csv()
        self.load_mat()
        self.size = len(self)

    def load_csv(self):
        """
        reads in csv and populates frame attribute
        """
        self.frame = pd.read_csv(self.fn)

    def load_mat(self):
        """
        handles creation of sparse matrix
        """

        self.init_mat()
        self.populate_mat()

    def init_mat(self):
        """
        instantiates sparse matrix of shape (N,M)

        N: Number of Molecules
        M: Bitvector Feature Length
        """
        self.mat = np.zeros(
            (len(self), self.bitspace), dtype=np.int8
        )

    def populate_mat(self):
        """
        sets activations for each molecule by iterating through onbits
        """
        for idx, onbits in enumerate(self.frame.OnBits):
            for jdx in self.iter_bits(onbits):
                self.mat[idx, jdx] = True

    def iter_bits(self, onbits):
        """
        Converts CSV activations to generator of integers

        :param onbits:
            CSV-String of activations
        :return:
            Generator of integers representing activations

        """
        for b in onbits.strip().split(","):
            yield int(b)

    def pdist(self, metric='jaccard'):
        """
        calculate pairwise distances within a matrix

        :param metric:
            distance metric to use (either euclidean/jaccard)
        :return:
            2D Distance Matrix
        """
        self.distmat = squareform(
            PairwiseDistance(self.mat, metric=metric)
        )

        return self.distmat

    def __len__(self):
        """
        returns number of molecules in ligand dataset
        """
        return self.frame.shape[0]


class Clustering:
    """
    Parent class for clustering

    :param metric:
        distance metric to use (either euclidean/jaccard)
    :param seed:
        random seed
    """

    def __init__(self, metric='euclidean', seed=None):
        if seed:
            np.random.seed(seed)

        if metric == 'euclidean':
            self.fn_metric = euclidean
        else:
            self.fn_metric = jaccard

    def pdist(self, data):
        """
        calculate pairwise distances within a matrix

        :param metric:
            distance metric to use (either euclidean/jaccard)
        :return:
            2D Distance Matrix
        """

        return squareform(
            PairwiseDistance(data, metric=self.metric)
            )

    def paired_distance(self, x, arr_y):
        """
        calculate distances between x and all values in arr_y

        :param x:
            1D array
        :param arr_y:
            2D array

        :return:
            1D array of distances between x and all values in arr_y
        """

        distances = np.zeros(arr_y.shape[0])
        for i, y in enumerate(arr_y):
            distances[i] = self.fn_metric(x, y)
        return distances

    def argmin(self, x):
        """
        returns argmin of an array with random choice of ties

        :param x:
            1D array

        :return:
            int (index of minimum)
        """

        m = np.min(x)
        return np.random.choice(
            np.flatnonzero(x == m)
        )

    def cluster(self, *args, **kwargs):
        """
        calls cluster method of child class
        """
        return self.__fit__(*args, **kwargs)


class PartitionClustering(Clustering):
    """
    Implementation of KMeans Clustering
    """

    def initialize_centroids(self):
        """
        k++ algorithm for initializing centroids
        """

        # initialize uniform probability
        prob = np.ones(self.n)
        prob /= prob.sum()

        # contains chosen indices
        self._k_indices = []

        # contains centroids
        centroids = np.zeros((self.k, self.m))

        # initialize k centroids
        for k in np.arange(self.k):

            # select observation with some probability
            c_idx = np.random.choice(self.n, p=prob)

            # append choice to container
            self._k_indices.append(c_idx)

            # initialize distance matrix at epoch
            distances = np.zeros((self.n, len(self._k_indices)))

            # for each existing centroid
            for i in np.arange(len(self._k_indices)):

                # calculate distances of all points to that centroid
                distances[:, i] = self.paired_distance(
                    self.data[self._k_indices[i]],
                    self.data[np.arange(self.n)]
                ) ** 2

            # take minimal distance across all centroids
            sq_distances = np.min(distances, axis=1)

            # set already chosen points probability to zero
            sq_distances[self._k_indices] = 0

            # normalize distances to probabilities
            prob = sq_distances / sq_distances.sum()

            # assign centroid to container
            centroids[k] = self.data[c_idx]

        return centroids

    def assign_centroids(self):
        """
        assign all observations to their closest centroid
        """

        # iterate through observations
        for idx in np.arange(self.n):

            # calculate distances to centroids
            k_dist = self.paired_distance(
                self.data[idx], self.centroids
            )

            # save centroid scores
            self._scores[idx] = k_dist

            # assign label to nearest centroid
            self.labels[idx] = self.argmin(k_dist)

    def update_centroids(self):
        """
        create new centroids from the means of their members

        :return:
            global distance (float)
        """

        distances = np.zeros(self.k)
        self.new_centroids = self.centroids.copy()

        for k in np.arange(self.k):

            members = np.flatnonzero(self.labels == k)

            if members.size == 0:
                continue

            updated_centroid = self.data[members].mean(axis=0)

            distances[k] = np.abs((updated_centroid - self.centroids[k]).sum())

            self.new_centroids[k] = updated_centroid

        return distances.sum()

    def score(self):
        """
        calculate silhouette coefficients for each observation.

        a_i = cohesion (mean within-cluster distance)
        b_i = separation (minimum mean between-cluster distance)
        s_i = silhouette coefficient

        s_i = (b_i - a_i) / max(a_i, b_i)

        :return:
            array of silhouette coefficients
        """

        # initialize silhouette score array
        s_i = np.zeros(self.n)

        # iterate through each observation
        for i in np.arange(self.n):

            # subset observation's cached centroid distances
            vals = self._scores[i]

            # subset observation's label
            label = self.labels[i]

            # initialize mask of all clusters observation is outside
            mask = np.arange(self.k) != label

            # cohesion : within-cluster distance
            a_i = vals[label]

            # separation : minimum between-cluster distance
            if mask.sum() == 0:  # case where only 1 k given
                b_i = 0
            else:
                b_i = np.min(vals[mask])

            # calculates silhouette coefficient
            if (a_i == 0) & (b_i == 0):  # case where division by zero
                score = 0
            else:
                score = (b_i-a_i) / np.max([a_i, b_i])

            s_i[i] = score

        return s_i

    def __fit__(self, data, k, max_iter=100):
        """
        K-Means Implementation

        - Initialize centroids with K++ Algorithm.

        Iteratively :
            - Assign Clusters
            - Update Centroids
            - Measure Global Distance

        Quits at distance minima

        :param data:
            2D numpy array to cluster
        :param k:
            Number of clusters (int)
        :param max_iter:
            Number of iterations to run before quitting

        :return:
            1D array of labels
        """

        # cache inputs
        self.data = data
        self.k = k

        # dimensions of input data
        self.n = self.data.shape[0]
        self.m = self.data.shape[1]

        # initialize empty labels array
        self.labels = np.zeros(self.n, dtype=np.int_)

        # initialize empty centroid distance cache
        self._scores = np.zeros((self.n, self.k))

        # initialize centroids
        self.centroids = self.initialize_centroids()

        # catches cases where k < 2
        if self.k < 2:
            self.assign_centroids()
            return self.labels

        else:

            iter = 0
            current_distance = np.inf

            while True:

                # assign each observation to a centroid
                self.assign_centroids()

                # calculate global centroid movement cost
                distance = self.update_centroids()

                # if centroids are moving to minimize distance
                if distance < current_distance:

                    # set new global distance to beat
                    current_distance = distance

                    # update centroids
                    self.centroids = self.new_centroids

                # quit if the max iterations are hit
                elif iter == max_iter:
                    break

                # quit if centroids moved and increased global distance
                else:
                    break

            return self.labels


class HierarchicalClustering(Clustering):

    def linkage_single(self, d):
        return d.min()

    def linkage_complete(self, d):
        return d.max()

    def linkage_average(self, d):
        return d.mean()

    def init_linkage_matrix(self):
        """
        a linkage matrix specified by scipy linkage matrix format

        2D matrix (n-1, 4):
            [cls_i, cls_j, dist, # original observations in new cluster]
        """
        return np.zeros(
            (self.n-1, 4)
            )

    def init_label_lineage(self):
        """
        initialize lineage of labels
        """
        return np.zeros(
            (self.n-1, self.n), dtype=np.int32
        )

    def init_labels(self):
        """
        initializes unique label for each observation
        """

        return np.arange(self.n)

    def minimal_distance(self):
        """
        finds the minimal distance between all clusters and
        returns pair and distance
        """

        # find all unique cluster labels
        unique_clusters = np.unique(self.labels)

        # initialize variables
        min_dist = 0
        min_pair = None

        # iterate through unique pairs of labels
        iter = 0
        for i, j in PairwiseIter(unique_clusters.size):

            label_i = unique_clusters[i]
            label_j = unique_clusters[j]

            # check if distance has already been done
            if (label_i, label_j) not in self.memo:

                # find indices of cluster members
                m1 = np.flatnonzero(self.labels == label_i)
                m2 = np.flatnonzero(self.labels == label_j)

                # build interaction of indices
                indices = np.ix_(m1, m2)

                # subset distance matrix to reflect interaction of indices
                all_distances = self.distmat[indices].ravel()

                # calculate linkage distance
                linkage_distance = self.linkage_method(all_distances)

                # fill memoization
                self.memo[(label_i, label_j)] = linkage_distance

            # draw from existing memoization
            else:
                linkage_distance = self.memo[(label_i, label_j)]

            # accept linkage as new minima or continue
            if (linkage_distance <= min_dist) | (iter == 0):
                min_dist = linkage_distance
                min_pair = (label_i, label_j)

            iter += 1

        return min_pair, min_dist

    def update_linkage_matrix(self, pair, dist, iter):
        """
        updates linkage matrix of incoming merge
        """

        # splits pair into labels
        x, y = pair

        # calculates number of nodes in new merger
        num_orig = (self.labels == x).sum() + (self.labels == y).sum()

        # updates linkage with merge
        self.zmat[iter] = np.array([
            int(x), int(y), dist, int(num_orig)
        ])

    def update_label_lineage(self, iter):
        self.label_lineage[iter] = self.labels

    def update_clusters(self, pair):
        """
        merges cluster X and cluster Y into a single cluster of label z
        """

        # split pair into labels
        x, y = pair

        # name of new label
        z = self.num_clusters

        # increment number of labels
        self.num_clusters += 1

        # select for all indices of x or y
        label_indices = np.flatnonzero(
            (self.labels == x) | (self.labels == y)
        )

        # update merged labels to reflect new cluster label
        self.labels[label_indices] = z

    def get_lineage(self):
        """
        return lineage of labels
        """

        return self.label_lineage

    def score(self, idx):
        """
        score the clustering at a given epoch
        """

        # initialize silhouette score array
        s_i = np.zeros(self.n)

        # iterate through each observation
        for i in np.arange(self.n):

            # subset observation's label
            label = self.label_lineage[idx, i]

            # cohesion : within-cluster distance
            cluster_members = np.flatnonzero(self.label_lineage[idx] == label)

            if cluster_members.size > 1:
                a_i = np.mean([
                    self.memo[
                        tuple(sorted((i, j)))
                        ] for j in cluster_members[cluster_members != i]
                ])
            else:
                a_i = 0

            # separation : minimum between-cluster distance
            cluster_distances = []

            for u in np.unique(self.label_lineage[idx]):
                if u == label:
                    continue
                cluster_members = np.flatnonzero(self.label_lineage[idx] == u)
                cluster_distances.append(
                    np.mean([
                        self.memo[
                            tuple(sorted((i, j)))
                            ] for j in cluster_members
                    ])
                )

            b_i = np.min(cluster_distances)

            # calculates silhouette coefficient
            if (a_i == 0) & (b_i == 0):  # case where division by zero
                score = 0
            else:
                score = (b_i-a_i) / np.max([a_i, b_i])

            s_i[i] = score

        return s_i

    def __fit__(self, data, linkage='single', precomputed=True):

        if precomputed:
            self.distmat = data
        else:
            self.distmat = PairwiseDistance(data, metric=self.metric)

        self.n = self.distmat.shape[0]

        # define linkage method
        self.linkages = {
            "single": self.linkage_single,
            "complete": self.linkage_complete,
            "average": self.linkage_average
        }
        self.linkage_method = self.linkages[linkage]
        self.memo = {}

        # initialize linkage matrix
        self.zmat = self.init_linkage_matrix()

        # initialize global label matrix
        self.label_lineage = self.init_label_lineage()

        # populate initial labels
        self.labels = self.init_labels()

        # track number of unique cluster labels (will increase with mergers)
        self.num_clusters = self.labels.size

        # main loop
        for iter in tqdm(np.arange(self.zmat.shape[0])):

            # find minimal distance
            pair, dist = self.minimal_distance()

            # update linkage matrix
            self.update_linkage_matrix(pair, dist, iter)

            # merge clusters of minimap pairs
            self.update_clusters(pair)

            # update label lineage
            self.update_label_lineage(iter)

        # return linkage matrix
        return self.zmat
