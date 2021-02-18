#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from scipy.spatial.distance import squareform
from multiprocessing import Pool
import sys


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
        distances = distances.reshape((self.ligands.size, self.k))

        # return indices of lowest distances
        return self.argmin(distances)

    def __fit__(self, k, max_iter=300):
        self.k = k
        self.centroids = self.initialize_centroids()
        self.labels = np.zeros(self.ligands.size, dtype=np.int32)

        iter = 0
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
