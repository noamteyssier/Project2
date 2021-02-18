#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from scipy.spatial.distance import squareform
from multiprocessing import Pool
import sys


# class Ligand():
#
#     """
#     Handles relevant information and tools for ligands
#
#     - IO
#     - SparseArrays
#     - Values
#
#     """
#
#     def __init__(self, csv, bitspace=1024):
#         self.csv = csv
#         self.csv_frame = self._load_csv(csv)
#         self.sparse, self.lookup = self._prepare_sparse()
#
#         self.bitspace = bitspace
#         self.mean = self._mean_activation()
#
#     def _load_csv(self, csv):
#         """
#         read in csv from path
#         """
#
#         frame = pd.read_csv(csv, sep=",")
#         frame['idx'] = np.arange(frame.shape[0])
#         return frame
#
#     def _prepare_sparse(self):
#         """
#         process onbits into sets for pairwise evaluation
#         create lookup table for ligand index with ligand name
#         """
#
#         sparse = {}
#         lookup = {}
#
#         self.csv_frame.apply(
#             lambda x: sparse.update(
#                 {
#                     x.idx: set([int(i) for i in x.OnBits.split(",")])
#                 }),
#             axis=1
#         )
#
#         self.csv_frame.apply(
#             lambda x: lookup.update(
#                 {
#                     x.idx: x.LigandID
#                 }),
#             axis=1
#         )
#
#         return sparse, lookup
#
#     def _mean_activation(self):
#         """
#         finds the mean number of activations across chemical space
#
#         (i.e. the largest value across the sparse vectors)
#         """
#
#         mean_activation = 0
#
#         for s in self.sparse:
#             num_activation = len(self.sparse[s])
#
#             mean_activation += num_activation
#
#         return mean_activation / len(self.sparse)
#
#     def PairwiseIter(self):
#         """
#         iterates unique pairwise combinations of observations
#         """
#
#         for idx, i_bits in self.__iter__():
#
#             for jdx, j_bits in self.__iter__():
#
#                 if jdx <= idx:
#                     continue
#
#                 yield idx, jdx, i_bits, j_bits
#
#     def GetMolecule(self, index):
#         """
#         Returns sparse molecular values of a given index
#
#         Will map index to ligand id and return sparse.
#         """
#         return self.sparse[self.lookup[index]]
#
#     def __iter__(self):
#         idx_order = sorted([i for i in self.sparse])
#         for idx in idx_order:
#             yield (idx, self.sparse[idx])
#
#     def __len__(self):
#         return len(self.sparse)
#
#
# class Clustering():
#
#     """
#     Parent class for HierarchicalClustering and PartitionClustering
#     """
#
#     def __init__(self, ligands, metric='jaccard', **kwargs):
#
#         self.ligands = ligands
#         self.metric = metric
#
#         self.distvec = np.array([])
#         self.index_tup_vec = dict()
#         self.index_vec_tup = dict()
#
#         # initialize empty labels array
#         self.labels = np.array([])
#
#         self._build_distmat()
#
#     def jaccard(self, x, y):
#         """
#         Implements Jaccard Distance
#         Intersection / Union
#
#         Params:
#         ------
#         x :
#             a set of values
#         y :
#             a set of values
#
#         Returns:
#         -------
#         distance :
#             float
#         """
#
#         size_ix = len(x.intersection(y))
#         size_un = len(x.union(y))
#         similarity = (size_ix / size_un)
#         distance = 1 - similarity
#
#         return distance
#
#     def _distance(self, x, y):
#
#         if self.metric == "jaccard":
#             return self.jaccard(x, y)
#         else:
#             print("Given Metric <{}> not implemented".format(self.metric))
#             sys.exit()
#
#     def _build_distmat(self):
#         """
#         initializes pairwise distances for ligand set
#         """
#
#         num_ligands = len(self.ligands)
#         num_comparisons = int((num_ligands * (num_ligands - 1)) / 2)
#
#         self.distvec = np.zeros(num_comparisons)
#         self.index_tup_vec = dict()
#         self.index_vec_tup = dict()
#
#         for index, vals in enumerate(self.ligands.PairwiseIter()):
#             idx, jdx, i_bits, j_bits = vals
#
#             self.index_tup_vec[(idx, jdx)] = index
#             self.index_vec_tup[index] = (idx, jdx)
#
#             self.distvec[index] = self._distance(i_bits, j_bits)
#
#     def _get_distance(self, idx, jdx):
#         """
#         finds distance between ligand index <idx> and ligand index <jdx>
#         """
#         if idx < jdx:
#             return self.distvec[self.index_tup_vec[(idx, jdx)]]
#         else:
#             return self.distvec[self.index_tup_vec[(jdx, idx)]]
#
#     def _pairwise_iter(self, arr):
#         """
#         generic generator for unique pairwise iteration
#         """
#
#         for idx, x in enumerate(arr):
#             for jdx, y in enumerate(arr):
#                 if jdx <= idx:
#                     continue
#
#                 yield (x, y)
#
#     def _get_cluster_members(self, c):
#         """
#         returns indices of cluster <c> members
#         """
#         return np.flatnonzero(self.labels == c)
#
#     def fit(self, **kwargs):
#         return self.__fit__(**kwargs)
#
#
# class HierarchicalClustering(Clustering):
#
#     """
#     Methods for Hierarchical Clustering
#     """
#
#     def _labels(self):
#         """
#         initially populates cluster <--> ligand mappings
#         """
#         return np.arange(len(self.ligands))
#
#     def _linkage_matrix(self):
#         """
#         a linkage matrix specified by scipy linkage matrix format
#
#         2D matrix (n-1, 4):
#             [cls_i, cls_j, dist, # original observations in new cluster]
#         """
#         n = len(self.ligands)
#         return np.zeros((n-1, 4))
#
#     def _update_clusters(self, x, y, verbose=False):
#         """
#         merges cluster X and cluster Y into a single cluster of label z
#         """
#
#         z = self.num_clusters
#         self.num_clusters += 1
#
#         if verbose:
#             print("Merging {} & {} -> {}".format(x, y, z))
#
#         label_indices = np.flatnonzero((self.labels == x) | (self.labels == y))
#         self.labels[label_indices] = z
#
#     def _update_linkage(self, x, y, dist, iter):
#         """
#         update linkage matrix
#         """
#         num_orig = np.flatnonzero(self.labels == x).size + \
#             np.flatnonzero(self.labels == y).size
#
#         self.zmat[iter] = np.array([int(x), int(y), dist, int(num_orig)])
#
#     def _get_cluster_distances(self, x, y):
#         """
#         returns the array of distances between all x and y pairs
#         """
#
#         dist_arr = np.zeros(x.size * y.size)
#         idx = 0
#
#         for i in x:
#             for j in y:
#                 dist_arr[idx] = self._get_distance(i, j)
#                 idx += 1
#
#         return dist_arr
#
#     def _linkage_single(self, d):
#         """
#         implements single linkage
#
#         minimum distance between points in clusters
#         """
#
#         return np.min(d)
#
#     def _linkage_complete(self, d):
#         """
#         implements complete linkage
#
#         maximum distance between points in clusters
#         """
#         return np.max(d)
#
#     def _linkage_average(self, d):
#         """
#         implements average linkage
#
#         average distance between points in clusters
#         """
#         return np.mean(d)
#
#     def _linkage(self, x, y):
#         """
#         calculates pairwise distance between clusters and applies
#         chosen linkage method
#         """
#
#         dists = self._get_cluster_distances(x, y)
#
#         return self.linkage_method(dists)
#
#     def _cluster_members(self):
#         """
#         organizes current clusters and members
#         """
#
#         self.unique_clusters = np.unique(self.labels)
#         self.cluster_members = {
#             u: self._get_cluster_members(u) for u in self.unique_clusters
#         }
#
#         return self.unique_clusters, self.cluster_members
#
#     def _minimum_distance(self, verbose=False):
#         """
#         Finds the minimum distance specificed by the linkage method between
#         all clusters
#         """
#
#         self.unique_clusters, self.cluster_members = self._cluster_members()
#         min_dist = 1
#         min_pair = None
#
#         for c1, c2 in self._pairwise_iter(self.unique_clusters):
#
#             dist = self._linkage(
#                 self.cluster_members[c1],
#                 self.cluster_members[c2]
#                 )
#
#             if dist <= min_dist:
#                 min_dist = dist
#                 min_pair = (c1, c2)
#
#         if verbose:
#             print(min_dist, min_pair)
#
#         x, y = min_pair
#         return x, y, min_dist
#
#     def __fit__(self, linkage="single", verbose=False):
#
#         self.linkages = {
#             "single": self._linkage_single,
#             "complete": self._linkage_complete,
#             "average": self._linkage_average
#         }
#
#         # defines linkage method
#         self.linkage_method = self.linkages[linkage]
#
#         # initializes linkage matrix
#         self.zmat = self._linkage_matrix()
#
#         # populates initial labels
#         self.labels = self._labels()
#
#         # tracks number of unique cluster labels
#         self.num_clusters = self.labels.size
#
#         # main loop
#         for iter in np.arange(self.zmat.shape[0]):
#
#             # find minimal distance
#             x, y, dist = self._minimum_distance(verbose=verbose)
#
#             # update linkage matrix
#             self._update_linkage(x, y, dist, iter)
#
#             # merge clusters of minimal pairs
#             self._update_clusters(x, y, verbose=verbose)
#
#         # returns linkage matrix
#         return self.zmat
#
#
# class PartitionClustering(Clustering):
#
#     """
#     Methods for K-Means Partition Clustering
#     """
#
#     def _member_activations(self, members):
#         """
#         yield all activations found in a cluster
#         """
#         for m in members:
#             for activation in self.ligands.sparse[m]:
#                 yield activation
#
#     def _mean_member_activations(self, members):
#         """
#         calculate the mean number of activations across members
#         """
#         activations = np.array([
#             len(self.ligands.sparse[m]) for m in members
#         ])
#         if activations.size == 0:
#             return 1
#         else:
#             return activations.mean()
#
#     def _get_activation_frequencies(self, members):
#         """
#         get the frequencies of activations across a cluster membership
#         """
#
#         # initialize empty frequencies
#         frequencies = np.zeros(self.ligands.bitspace)
#
#         # gather all activations seen across members
#         all_activations = np.array([
#             a for a in self._member_activations(members)
#         ])
#
#         # counts of each unique activations in cluster
#         activations, counts = np.unique(all_activations, return_counts=True)
#
#         # assign frequency position (missing activations stay zero)
#         try:
#             frequencies[activations] = counts
#         except IndexError:
#             frequencies[np.random.choice(frequencies.size)] = 1
#
#         # convert to probability
#         frequencies = frequencies / frequencies.sum()
#
#         return frequencies
#
#     def _weighted_centroid(self, n, p):
#         """
#         creates a new centroid with a given multinomial parameterization
#         """
#         mn = np.random.multinomial(n, p)
#         return set(np.flatnonzero(mn))
#
#     def _initialize_centroids(self):
#         """
#         initializes <k> centroids as <k> random points in space (k++ method)
#         """
#         centroids = []
#         for i in range(self.k):
#             random_point = self.ligands.sparse[
#                 np.random.choice(len(self.ligands) + 1)
#                 ]
#             centroids.append((i, random_point))
#
#         return centroids
#
#     def _assign_centroids(self):
#         """
#         assigns each ligand to its closest centroid
#         """
#
#         for lig_idx, lig_bits in self.ligands:
#
#             distances = np.array([
#                 self._distance(lig_bits, bits) for idx, bits in self.centroids
#             ])
#
#             assignment = np.random.choice(
#                 np.flatnonzero(distances == distances.max())
#                 )
#
#             self.labels[lig_idx] = assignment
#
#     def _intercluster_distance(self, k):
#         """
#         sum pairwise distances within a cluster <k>
#         """
#
#         clust = self._get_cluster_members(k)
#         distances = np.array([
#             self._get_distance(x, y) for x, y in self._pairwise_iter(clust)
#         ])
#
#         return distances.sum()
#
#     def _evaluate_clusters(self):
#         """
#         sums inter-cluster distances across all clusters
#         """
#
#         cluster_distances = np.array([
#             self._intercluster_distance(k) for k in range(self.k)
#         ])
#
#         return cluster_distances.sum()
#
#     def _update_centroids(self):
#         """
#         updates centroids to reflect members
#         """
#
#         for k in range(self.k):
#
#             # retrieve members of a cluster
#             members = self._get_cluster_members(k)
#
#             # calculate mean activations across members
#             mean_activations = self._mean_member_activations(members)
#
#             # calculate activation frequencies
#             frequencies = self._get_activation_frequencies(members)
#
#             # build centroid with multinomial derived activations
#             centroid = self._weighted_centroid(mean_activations, frequencies)
#
#             self.centroids[k] = (k, centroid)
#
#     def __fit__(self, k, seed=42, max_iter=200):
#         """
#         implements k-means clustering
#
#         will return the minimum score and cluster labels at that score
#         """
#         np.random.seed(seed)
#         self.k = k
#
#         self.centroids = self._initialize_centroids()
#         self.labels = np.zeros(len(self.ligands), dtype=int)
#
#         iter = 0
#
#         self.current_min = 0
#         self.current_labels = self.labels.copy()
#         while iter < max_iter:
#
#             self._assign_centroids()
#             total_distances = self._evaluate_clusters()
#
#             if (total_distances < self.current_min) | (iter == 0):
#                 self.current_min = total_distances
#                 self.current_labels = self.labels.copy()
#
#             self._update_centroids()
#
#             iter += 1
#
#         return self.current_min, self.current_labels
#
#
# class qCluster():
#
#     """
#     Evaluates the quality of a given clustering scheme
#     """
#
#     pass
#
#
# class simCluster():
#
#     """
#     Evaluates the similarity between a set of a clusters
#     """
#
#     pass
#

class Ligand():

    def __init__(self, fn, bitspace=1024):
        self.fn = fn
        self.bitspace = bitspace

        self.load_csv()
        self.build_sparsemat()
        self.size = len(self)

    def load_csv(self):
        """
        reads in csv and populates frame attribute
        """
        self.frame = pd.read_csv(self.fn)

    def build_sparsemat(self):
        """
        handles creation of sparse matrix
        """

        self.init_sparsemat()
        self.populate_sparsemat()

    def init_sparsemat(self):
        """
        instantiates sparse matrix of shape (N,M)

        N: Number of Molecules
        M: Bitvector Feature Length
        """
        self.sparse = dok_matrix(
            (len(self), self.bitspace), dtype=np.int8
        )

    def iter_bits(self, onbits):
        """
        generator to transform csv-string representation
        of activations into integers
        """
        for b in onbits.strip().split(","):
            yield int(b)

    def populate_sparsemat(self):
        """
        sets activations for each molecule by iterating through onbits
        """
        for idx, onbits in enumerate(self.frame.OnBits):
            for jdx in self.iter_bits(onbits):
                self.sparse[idx, jdx] = 1

    def get_activations(self, index):
        """
        returns activations for a given molecule
        """
        return self.sparse[index].nonzero()[1]

    def get_dense(self, index):
        """
        returns dense array of activations for a given molecule
        """
        return self.sparse[index].toarray().ravel()

    def iter_dense(self):
        for idx in self:
            yield self.get_dense(idx)

    def __len__(self):
        """
        returns number of molecules in ligand dataset
        """
        return self.frame.shape[0]

    def __iter__(self):
        for idx in np.arange(self.size):
            yield idx

class Clustering():

    def __init__(self, ligands, metric='jaccard', seed=42):
        np.random.seed(seed)

        self.ligands = ligands
        self.metric = metric

        self.call_metric = {
            "jaccard": self.jaccard,
            "euclidean": self.euclidean
        }

    def jaccard(self, x, y):
        """
        calulcates jaccard distance as 1 - intersection over union
        """

        ix = np.intersect1d(x, y).size
        un = np.union1d(x, y).size

        return 1 - (ix / un)

    def euclidean(self, x, y):
        """
        calculates euclidean distance between two arrays
        """

        distance = np.sqrt(
            ((y - x)**2).sum()
        )

        return distance

    def distance(self, *args):
        """
        calls given distance metric
        """

        return self.call_metric[self.metric](*args)

    def dense_distance(self, indices):
        """
        maps given indices to dense array from ligands' sparse matrix
        """

        idx, jdx = indices
        return self.distance(
            self.ligands.get_dense(idx),
            self.ligands.get_dense(jdx)
        )

    def sparse_distance(self, indices):
        """
        maps given indices to onbits from ligands' sparse matrix
        """

        idx, jdx = indices
        return self.distance(
            self.ligands.get_activations(idx),
            self.ligands.get_activations(jdx)
        )

    def pairwise_iter(self, arr):
        """
        iterates through pairs of indices in unique pairwise manner
        """

        for idx in np.arange(arr.size):
            for jdx in np.arange(arr.size):
                if jdx > idx:
                    yield (idx, jdx)

    def pairwise_distance(self, save=False):
        """
        calculates pairwise distances over all pairs of molecules
        """

        p = Pool()
        condensed_distance_matrix = p.map(
            self.dense_distance,
            self.pairwise_iter(self.ligands)
            )
        p.close()

        self.distmat = squareform(condensed_distance_matrix)

        if save:
            np.save("temp.npy", self.distmat)

    def combinations(self, iter_i, iter_j):
        for i in iter_i:
            for j in iter_j:
                yield (i, j)

    def load_dist(self, fn):
        self.distmat = np.load(fn)

    def fit(self, *args, **kwargs):
        return self.__fit__(*args, **kwargs)

class PartitionClustering(Clustering):

    def initialize_centroids(self):
        """
        initialize <k> centroids as <k> random observations from dataset
        """

        centroids = self.distmat[
            np.random.choice(self.ligands.size, size=self.k)
        ]
        return centroids

    def assign_centroids(self):
        """
        combinatorial distance between centroids and molecules
        """

        print("START HERE")
        sys.exit()
        distances = np.array(
            list(map(
                lambda x : self.euclidean(x[0], x[1]),
                self.combinations(
                    self.centroids, self.distmat
                )
            ))
        )


        print(distances)




    def __fit__(self, k, max_iter=300):
        self.k = k
        self.centroids = self.initialize_centroids()
        self.labels = np.zeros(self.ligands.size, dtype=np.int32)

        iter=0
        while iter < max_iter:

            self.assign_centroids()

            iter += 1
            break


def main():
    ligands = Ligand("../data/test_set.csv")
    kmeans = PartitionClustering(ligands, metric='euclidean')
    # kmeans.pairwise_distance(save=True)
    kmeans.load_dist("temp.npy")

    kmeans.fit(k=3)

if __name__ == '__main__':
    main()
