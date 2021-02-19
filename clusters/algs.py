#!/usr/bin/env python3

import numpy as np
import pandas as pd
import numba as nb
from scipy.spatial.distance import squareform
import sys


@nb.jit(nopython=True, fastmath=True)
def euclidean(x, y):
    return np.sqrt(
        np.sum((y-x)**2)
        )


@nb.jit(nopython=True)
def jaccard(x, y):
    mask_x = (x > 0)
    mask_y = (y > 0)

    ix = np.sum(mask_x & mask_y)
    un = np.sum(mask_x | mask_y)
    return ix / un


@nb.jit(nopython=True)
def PairwiseIter(n):
    """
    iterates through pairwise indices over range N
    """
    for i in np.arange(n):
        for j in np.arange(i, n):
            if j > i:
                yield (i, j)


@nb.jit(nopython=True)
def PairwiseDistance(m, metric='euclidean'):
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


class Ligand():

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
        generator to transform csv-string representation
        of activations into integers
        """
        for b in onbits.strip().split(","):
            yield int(b)

    def get_activations(self, index):
        """
        returns activations for a given molecule
        """
        return np.flatnonzero(self.mat[index])

    def get_dense(self, index):
        """
        returns dense array of activations for a given molecule
        """
        return self.mat[index]

    def iter_dense(self):
        for idx in self:
            yield self.get_dense(idx)

    def pdist(self, metric='jaccard'):
        """
        calculate pairwise distances within a matrix
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

    def __iter__(self):
        for idx in np.arange(self.size):
            yield idx


class Clustering:
    def __init__(self, metric='euclidean', seed=None):
        if seed:
            np.random.seed(seed)

        if metric == 'euclidean':
            self.fn_metric = euclidean
        else:
            self.fn_metric = jaccard

    def pdist(self, data):
        return squareform(
            PairwiseDistance(data, metric=self.metric)
            )

    def paired_distance(self, x, arr_y):
        """
        calculate distances between x and all values in arr_y
        """
        distances = np.zeros(arr_y.shape[0])
        for i, y in enumerate(arr_y):
            distances[i] = self.fn_metric(x, y)
        return distances

    def argmin(self, x):
        m = np.min(x)
        return np.random.choice(
            np.flatnonzero(x == m)
        )

    def fit(self, *args, **kwargs):
        return self.__fit__(*args, **kwargs)


class PartitionClustering(Clustering):

    def initialize_centroids(self):

        prob = np.ones(self.n)
        prob /= prob.sum()
        self._k_indices = []
        centroids = np.zeros((self.k, self.m))

        for k in np.arange(self.k):

            # select centroid
            c_idx = np.random.choice(self.n, p=prob)
            self._k_indices.append(c_idx)

            distances = np.zeros((self.n, len(self._k_indices)))

            for i in np.arange(len(self._k_indices)):

                # calculate distances
                distances[:,i] = self.paired_distance(
                    self.data[self._k_indices[i]],
                    self.data[np.arange(self.n)]
                ) ** 2

            sq_distances = np.min(distances, axis=1)

            # set chosen points probability to zero
            sq_distances[self._k_indices] = 0

            # normalize frequencies
            prob = sq_distances / sq_distances.sum()
            centroids[k] = self.data[c_idx]

        return centroids

    def assign_centroids(self):

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

        s_i = np.zeros(self.n)

        for i in np.arange(self.n):

            vals = self._scores[i]
            label = self.labels[i]
            mask = np.arange(self.k) != label

            a_i = vals[label]

            if mask.sum() == 0:
                b_i = 0
            else:
                b_i = np.min(vals[mask])

            if (a_i == 0) & (b_i == 0):
                score = 0
            else:
                score = (b_i-a_i) / np.max([a_i, b_i])

            s_i[i] = score

        return s_i

    def __fit__(self, data, k, max_iter=100):
        self.data = data
        self.k = k

        self.n = self.data.shape[0]
        self.m = self.data.shape[1]

        self.labels = np.zeros(self.n, dtype=np.int_)
        self._scores = np.zeros((self.n, self.k))

        self.centroids = self.initialize_centroids()

        if self.k < 2:
            self.assign_centroids()
            return self.labels

        else:
            iter = 0
            current_distance = np.inf

            while True:

                self.assign_centroids()
                distance = self.update_centroids()

                if distance < current_distance:
                    current_distance = distance
                    self.centroids = self.new_centroids

                elif iter == max_iter:
                    break

                else:
                    break

            return self.labels


def main():

    ligands = Ligand("../data/subset.csv")
    # distmat = ligands.pdist(metric='jaccard')
    # np.save("subset.npy", distmat)
    distmat = np.load("subset.npy")

    km = PartitionClustering(metric='eulidean', seed=42)
    km.fit(distmat, k=3)
    sil = km.score()
    print(sil)


if __name__ == '__main__':
    main()



# class Clustering():
#
#     def __init__(self, ligands, metric='jaccard', seed=42):
#         np.random.seed(seed)
#
#         self.ligands = ligands
#         self.metric = metric
#
#         self.call_metric = {
#             "jaccard": self.jaccard,
#             "euclidean": self.euclidean
#         }
#
#     def jaccard(self, x, y):
#         """
#         calulcates jaccard distance as 1 - intersection over union
#         """
#
#         ix = np.intersect1d(x, y).size
#         un = np.union1d(x, y).size
#
#         return 1 - (ix / un)
#
#     def euclidean(self, x, y):
#         """
#         calculates euclidean distance between two arrays
#         """
#
#         distance = np.sqrt(
#             ((y - x)**2).sum()
#         )
#
#         return distance
#
#     def distance(self, *args):
#         """
#         calls given distance metric
#         """
#
#         return self.call_metric[self.metric](*args)
#
#     def dense_distance(self, indices):
#         """
#         maps given indices to dense array from ligands' sparse matrix
#         """
#
#         idx, jdx = indices
#         return self.distance(
#             self.ligands.get_dense(idx),
#             self.ligands.get_dense(jdx)
#         )
#
#     def pairwise_iter(self, arr):
#         """
#         iterates through pairs of indices in unique pairwise manner
#         """
#
#         for idx in np.arange(arr.size):
#             for jdx in np.arange(arr.size):
#                 if jdx > idx:
#                     yield (idx, jdx)
#
#     def pairwise_distance(self, save=False):
#         """
#         calculates pairwise distances over all pairs of molecules
#         """
#
#         p = Pool()
#         cdist = p.map(
#             self.dense_distance,
#             self.pairwise_iter(self.ligands)
#         )
#         p.close()
#
#         self.distmat = squareform(cdist)
#
#         if save:
#             np.save("subset_dist.npy", self.distmat)
#
#     def combinations(self, iter_i, iter_j):
#         for i in iter_i:
#             for j in iter_j:
#                 yield (i, j)
#
#     def argmin(self, distances, verbose=False):
#         """
#         for each observation return the minimal index
#         if there are multiple at the minimal index a random choice is given
#         """
#
#         # calculate observation minima
#         minimums = np.min(distances, axis=1)
#
#         # initialize empty argmin vector
#         argmins = np.zeros(minimums.size, dtype=np.int32)
#
#         # iterate through minima
#         for idx, m in enumerate(minimums):
#
#             # select random argmin (will just return argmin if singleton)
#             argmins[idx] = np.random.choice(
#                 np.flatnonzero(distances[idx] == m)
#                 )
#
#         return argmins
#
#     def load_dist(self, fn):
#         self.distmat = np.load(fn)
#
#     def fit(self, *args, **kwargs):
#         return self.__fit__(*args, **kwargs)
#
#     def get_distance_matrix(self):
#         """
#         returns distance matrix
#         """
#         return self.distmat
#
#
# class PartitionClustering(Clustering):
#
#     def initialize_centroids(self):
#         """
#         initialize <k> centroids with k++ algorithm
#         """
#
#         centroids = np.zeros((self.k, self.distmat.shape[1]))
#         self._chosen_indices = []
#
#         # initialize uniform density
#         current_probability = np.ones(self.ligands.size)
#         current_probability /= current_probability.sum()
#
#         for i in np.arange(self.k):
#
#             # Choose one center random among the data points.
#             centroid_idx = np.random.choice(
#                 self.ligands.size, p=current_probability
#                 )
#             self._chosen_indices.append(centroid_idx)
#
#             # extract all distances to centroid
#             squared_distances = self.distmat[:, centroid_idx]
#
#             # zero out already chosen centroids
#             squared_distances[self._chosen_indices] = 0
#
#             # set probability to squared distance frequencies
#             current_probability = squared_distances / squared_distances.sum()
#
#             # assign centroid
#             centroids[i] = self.distmat[centroid_idx]
#
#         return centroids
#
#     def assign_centroids(self):
#         """
#         combinatorial distance between centroids and molecules
#         """
#
#         # calculate distances of observations and centroids
#         distances = np.array(
#             list(
#                 map(
#                     lambda x: self.euclidean(x[0], x[1]),
#                     self.combinations(
#                         self.centroids, self.distmat
#                     )
#                 )
#             )
#         )
#
#         # reshape distances to reflect (NxK)
#         self.distances = distances.reshape((self.ligands.size, self.k))
#
#         # return indices of lowest distances
#         return self.argmin(self.distances)
#
#     def update_cluster(self, k):
#         """
#         update a centroid as the mean of its members
#         """
#
#         if np.any(self.labels == k):
#             # calculate mean of members
#             updated_centroid = self.distmat[self.labels == k].mean(axis=0)
#
#             # find any nans (for failed centroids)
#             idx_nan = np.flatnonzero(np.isnan(updated_centroid))
#
#             # reinitialize as random point in space
#             updated_centroid[idx_nan] = np.random.random(idx_nan.size)
#
#             # return updated centroid
#             return updated_centroid
#
#         else:
#             return self.centroids[k]
#
#     def update_clusters(self):
#         """
#         update clusters with membership
#         """
#
#         distances = np.zeros(self.k)
#         updated_centroids = self.centroids.copy()
#
#         for k in np.arange(self.k):
#
#             updated_centroids[k] = self.update_cluster(k)
#
#             distances[k] = self.euclidean(
#                 self.centroids[k],
#                 updated_centroids[k]
#                 )
#
#         global_distance = distances.sum()
#         return global_distance, updated_centroids
#
#     def __fit__(self, k, max_iter=100):
#         self.k = k
#         self.centroids = self.initialize_centroids()
#
#         self.current_distance = 0
#         iter = 0
#
#         while True:
#
#             # assign each observation to centroids
#             self.labels = self.assign_centroids()
#
#             break
#             # calculate global distance and calculate new centroids
#             distance, updated_centroids = self.update_clusters()
#
#             # check for distance minimization
#             if (iter == 0) | (distance < self.current_distance):
#
#                 # set new minima
#                 self.current_distance = distance
#
#                 # set new centroids
#                 self.centroids = updated_centroids.copy()
#
#             # quit if still haven't converged after max iterations
#             elif iter == max_iter:
#                 print("No Convergence, Breaking at {}".format(iter))
#                 break
#
#             # quit if distance calculation increases
#             else:
#                 break
#
#             iter += 1
#
#         return self.labels
#
#
# class HierarchicalClustering(Clustering):
#
#     def linkage_single(self, d):
#         return d.min()
#
#     def linkage_complete(self, d):
#         return d.max()
#
#     def linkage_average(self, d):
#         return d.mean()
#
#     def init_linkage_matrix(self):
#         """
#         a linkage matrix specified by scipy linkage matrix format
#
#         2D matrix (n-1, 4):
#             [cls_i, cls_j, dist, # original observations in new cluster]
#         """
#         return np.zeros(
#             (self.ligands.size-1, 4)
#             )
#
#     def init_labels(self):
#         """
#         initializes unique label for each observation
#         """
#
#         return np.arange(self.ligands.size)
#
#     def minimal_distance(self):
#         """
#         finds the minimal distance between all clusters and
#         returns pair and distance
#         """
#
#         # find all unique cluster labels
#         unique_clusters = np.unique(self.labels)
#
#         # initialize variables
#         min_dist = 0
#         min_pair = None
#
#         # iterate through unique pairs of labels
#         iter = 0
#         for i, j in self.pairwise_iter(unique_clusters):
#
#             label_i = unique_clusters[i]
#             label_j = unique_clusters[j]
#
#             # find indices of cluster members
#             m1 = np.flatnonzero(self.labels == label_i)
#             m2 = np.flatnonzero(self.labels == label_j)
#
#             # build interaction of indices
#             indices = np.ix_(m1, m2)
#
#             # subset distance matrix to reflect interaction of indices
#             all_distances = self.distmat[indices].ravel()
#
#             # calculate linkage distance
#             linkage_distance = self.linkage_method(all_distances)
#
#             # accept linkage as new minima or continue
#             if (linkage_distance <= min_dist) | (iter == 0):
#                 min_dist = linkage_distance
#                 min_pair = (label_i, label_j)
#
#             iter += 1
#
#         return min_pair, min_dist
#
#     def update_linkage_matrix(self, pair, dist, iter):
#         """
#         updates linkage matrix of incoming merge
#         """
#
#         # splits pair into labels
#         x, y = pair
#
#         # calculates number of nodes in new merger
#         num_orig = (self.labels == x).sum() + (self.labels == y).sum()
#
#         # updates linkage with merge
#         self.zmat[iter] = np.array([
#             int(x), int(y), dist, int(num_orig)
#         ])
#
#     def update_clusters(self, pair):
#         """
#         merges cluster X and cluster Y into a single cluster of label z
#         """
#
#         # split pair into labels
#         x, y = pair
#
#         # name of new label
#         z = self.num_clusters
#
#         # increment number of labels
#         self.num_clusters += 1
#
#         # select for all indices of x or y
#         label_indices = np.flatnonzero(
#             (self.labels == x) | (self.labels == y)
#         )
#
#         # update merged labels to reflect new cluster label
#         self.labels[label_indices] = z
#
#     def __fit__(self, linkage='single'):
#
#         # define linkage method
#         self.linkages = {
#             "single": self.linkage_single,
#             "complete": self.linkage_complete,
#             "average": self.linkage_average
#         }
#         self.linkage_method = self.linkages[linkage]
#
#         # initialize linkage matrix
#         self.zmat = self.init_linkage_matrix()
#
#         # populate initial labels
#         self.labels = self.init_labels()
#
#         # track number of unique cluster labels (will increase with mergers)
#         self.num_clusters = self.labels.size
#
#         # main loop
#         for iter in np.arange(self.zmat.shape[0]):
#
#             # find minimal distance
#             pair, dist = self.minimal_distance()
#
#             # update linkage matrix
#             self.update_linkage_matrix(pair, dist, iter)
#
#             # merge clusters of minimap pairs
#             self.update_clusters(pair)
#
#         # return linkage matrix
#         return self.zmat
#
#
# class Silhouette():
#
#     def __init__(self, distmat, labels):
#         self.distmat = distmat
#         self.labels = labels
#         self.unique_labels = np.unique(labels)
#
#     def cohesion(self, idx):
#         """
#         calculate cohesion coefficient
#         (i.e. mean within-cluster distances)
#         """
#
#         # identify neighbors
#         cluster_members = np.flatnonzero(self.labels == self.labels[idx])
#
#         # remove self from list of indices
#         cluster_members = cluster_members[cluster_members != idx]
#
#         # cluster is singleton
#         if cluster_members.size == 0:
#             return 0
#
#         # cluster has multiple members
#         else:
#             # extract within cluster distances
#             distances = self.distmat[idx, cluster_members]
#
#             # return mean
#             return distances.mean()
#
#     def separation(self, idx):
#         """
#         calculate separation coefficient
#         (i.e. minimum mean between-cluster distance)
#         """
#
#         # identify observations cluster label
#         idx_label = self.labels[idx]
#
#         # identify all other cluster labels
#         non_idx_labels = self.unique_labels[self.unique_labels != idx_label]
#
#         # initialize empty mean cluster distance array
#         cluster_distances = np.zeros(non_idx_labels.size)
#
#         # iterate through unique cluster labels
#         for i, u in enumerate(non_idx_labels):
#
#             # identify cluster members of other cluster
#             cluster_members = np.flatnonzero(self.labels == u)
#
#             # extract all distances of observation to all other cluster members
#             all_distances = self.distmat[idx, cluster_members]
#
#             # take the mean of all distances
#             cluster_distances[i] = all_distances.mean()
#
#         # return minimum inter-cluster distance
#         return cluster_distances.min()
#
#     def score(self, cohesion, separation):
#         """
#         calculate silhouette score for a given observation
#         """
#
#         if (separation == 0) & (cohesion == 0):
#             return 0
#
#         else:
#             return (separation - cohesion) / np.max([cohesion, separation])
#
#     def fit(self):
#         """
#         calculates silhouette scores for each observation
#         """
#
#         # initialize silhouette score array
#         s_i = np.zeros(self.labels.size)
#
#         # iterate through observations
#         for idx in np.arange(self.labels.size):
#
#             # calculate cohesion
#             cohesion = self.cohesion(idx)
#
#             # calculate separation
#             separation = self.separation(idx)
#
#             # calculate silhouette score
#             s_i[idx] = self.score(cohesion, separation)
#
#         # return silhouettes
#         return s_i
