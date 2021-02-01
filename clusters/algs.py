#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tqdm import tqdm

class Ligand():

	"""
	Handles relevant information and tools for ligands

	- IO
	- SparseArrays
	- Values

	"""

	def __init__(self, csv):
		self.csv = csv
		self.csv_frame = self._load_csv(csv)
		self.sparse, self.lookup = self._prepare_sparse()

	def _load_csv(self, csv):

		"""
		read in csv from path
		"""

		frame = pd.read_csv(csv, sep=",")
		frame['idx'] = np.arange(frame.shape[0])
		return frame

	def _prepare_sparse(self):

		"""
		process onbits into sets for pairwise evaluation
		create lookup table for ligand index with ligand name
		"""

		sparse = {}
		lookup = {}

		self.csv_frame.apply(
			lambda x : sparse.update(
				{
					x.idx : set([int(i) for i in x.OnBits.split(",")])
				}),
			axis = 1
		)

		self.csv_frame.apply(
			lambda x : lookup.update(
				{
					x.idx : x.LigandID
				}),
			axis = 1
		)

		return sparse, lookup

	def PairwiseIter(self):

		"""
		iterates unique pairwise combinations of observations
		"""

		already_seen = set()

		for idx, i_bits in self.__iter__():

			for jdx, j_bits in self.__iter__():

				if jdx <= idx:
					continue

				yield idx, jdx, i_bits, j_bits

	def __iter__(self):
		idx_order = sorted([i for i in self.sparse])
		for idx in idx_order:
			yield (idx, self.sparse[idx])

	def __len__(self):
		return len(self.sparse)

class Clustering():

	"""
	Parent class for HierarchicalClustering and PartitionClustering
	"""

	def __init__(self, ligands, metric='jaccard'):

		self.ligands = ligands
		self.metric = metric

		self.distvec = np.array([])
		self.index_tup_vec = dict()
		self.index_vec_tup = dict()

		self._build_distmat()

	def distance_jaccard(self, x, y):

		"""
		Implements Jaccard Distance
		Intersection / Union

		Params:
		------
		x :
			a set of values
		y :
			a set of values

		Returns:
		-------
		distance :
			float
		"""

		size_ix = len(x.intersection(y))
		size_un = len(x.union(y))
		similarity = (size_ix / size_un)
		distance = 1 - similarity

		return distance

	def _distance(self, x, y):

		if self.metric == "jaccard":
			return self.distance_jaccard(x, y)
		else:
			print("Given Metric <{}> not implemented".format(self.metric))
			sys.exit()

	def _build_distmat(self):

		"""
		initializes pairwise distances for ligand set
		"""

		num_ligands = len(self.ligands)
		num_comparisons = int((num_ligands * (num_ligands - 1)) / 2)

		self.distvec = np.zeros(num_comparisons)
		self.index_tup_vec = dict()
		self.index_vec_tup = dict()

		for index, (idx, jdx, i_bits, j_bits) in enumerate(self.ligands.PairwiseIter()):

			self.index_tup_vec[(idx,jdx)] = index
			self.index_vec_tup[index] = (idx, jdx)

			self.distvec[index] = self._distance(i_bits, j_bits)

	def fit(self):
		return self.__fit__()

class HierarchicalClustering(Clustering):

	"""
	Methods for Hierarchical Clustering
	"""

	def __fit__(self):
		print(self.distvec)

class PartitionClustering(Clustering):

	"""
	Methods for Hierarchical Clustering
	"""

	pass

class qCluster():

	"""
	Evaluates the quality of a given clustering scheme
	"""

	pass

class simCluster():

	"""
	Evaluates the similarity between a set of a clusters
	"""

	pass

def main():
	l = Ligand("../data/test_set.csv")

	hcl = HierarchicalClustering(l)
	hcl.fit()

	pass

if __name__ == '__main__':
	main()
