#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from multiprocessing import Pool
from tqdm import tqdm
import sys


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
        cdist = p.map(
            self.dense_distance,
            self.pairwise_iter(self.ligands)
        )
        p.close()

        self.distmat = squareform(cdist)

        if save:
            np.save("subset_dist.npy", self.distmat)

    def combinations(self, iter_i, iter_j):
        for i in iter_i:
            for j in iter_j:
                yield (i, j)

    def argmin(self, distances):
        """
        for each observation return the minimal index
        if there are multiple at the minimal index a random choice is given
        """

        # calculate observation minima
        minimums = np.min(distances, axis=1)

        # initialize empty argmin vector
        argmins = np.zeros(minimums.size, dtype=np.int32)

        # iterate through minima
        for idx, m in enumerate(minimums):

            # select random argmin (will just return argmin if singleton)
            argmins[idx] = np.random.choice(
                np.flatnonzero(distances[idx] == m)
                )

        return argmins

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

        # calculate distances of observations and centroids
        distances = np.array(
            list(
                map(
                    lambda x: self.euclidean(x[0], x[1]),
                    self.combinations(
                        self.centroids, self.distmat
                    )
                )
            )
        )

        # reshape distances to reflect (NxK)
        self.distances = distances.reshape((self.ligands.size, self.k))

        # return indices of lowest distances
        return self.argmin(self.distances)

    def update_clusters(self):
        """
        update clusters with membership
        """

        distances = np.zeros(self.k)
        updated_centroids = self.centroids.copy()

        for k in np.arange(self.k):

            updated_centroids[k] = self.distmat[self.labels == k].mean(axis=0)

            distances[k] = self.euclidean(
                self.centroids[k],
                updated_centroids[k]
                )

        global_distance = distances.sum()
        return global_distance, updated_centroids

    def __fit__(self, k, max_iter=100):
        self.k = k
        self.centroids = self.initialize_centroids()

        self.current_distance = 0
        iter = 0

        while True:

            # assign each observation to centroids
            self.labels = self.assign_centroids()

            # calculate global distance and calculate new centroids
            distance, updated_centroids = self.update_clusters()

            # check for distance minimization
            if (iter == 0) | (distance < self.current_distance):

                # set new minima
                self.current_distance = distance

                # set new centroids
                self.centroids = updated_centroids.copy()

            # quit if still haven't converged after max iterations
            elif iter == max_iter:
                print("No Convergence, Breaking at {}".format(iter))
                break

            # quit if distance calculation increases
            else:
                break

            iter += 1

        return self.labels


class HierarchicalClustering(Clustering):

    def linkage_single(self, d):
        return d.min()

    def linkage_complete(self, d):
        return d.max()

    def linkage_average(self, d):
        return d.mean()

    def linkage(self, m1, m2):
        """
        calculates appropriate linkage for cluster members
        """
        dist_arr = np.zeros(m1.shape[0] * m2.shape[0])

        idx = 0
        for i, j in self.combinations(m1, m2):
            dist_arr[idx] = self.euclidean(m1, m2)
            idx += 1

        return self.linkage_method(dist_arr)

    def init_linkage_matrix(self):
        """
        a linkage matrix specified by scipy linkage matrix format

        2D matrix (n-1, 4):
            [cls_i, cls_j, dist, # original observations in new cluster]
        """
        return np.zeros(
            (self.ligands.size-1, 4)
            )

    def init_labels(self):
        """
        initializes unique label for each observation
        """

        return np.arange(self.ligands.size)

    def minimal_distance(self):
        unique_clusters = np.unique(self.labels)

        
        for i, j in self.pairwise_iter(unique_clusters):
            distance = self.linkage(
                self.distmat[self.labels == i],
                self.distmat[self.labels == j]
            )
            print(distance)

            break
    def __fit__(self, linkage='single'):

        # define linkage method
        self.linkages = {
            "single": self.linkage_single,
            "complete": self.linkage_complete,
            "average": self.linkage_average
        }
        self.linkage_method = self.linkages[linkage]

        # initialize linkage matrix
        self.zmat = self.init_linkage_matrix()

        # populate initial labels
        self.labels = self.init_labels()

        # track number of unique cluster labels (will increase with mergers)
        self.num_clusters = self.labels.size

        # main loop
        for iter in np.arange(self.ligands.size):

            # find minimal distance
            self.minimal_distance()

            # update linkage matrix

            # merge clusters of minimap pairs

            break

        # return linkage matrix

        pass


def main():
    ligands = Ligand("../data/subset.csv")
    # kmeans = PartitionClustering(ligands, metric='euclidean')
    # kmeans.pairwise_distance(save=True)

    hclust = HierarchicalClustering(ligands, metric='euclidean')
    hclust.load_dist("subset_dist.npy")
    hclust.fit()


if __name__ == '__main__':
    main()
