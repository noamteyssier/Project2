
import numpy as np
from clusters.algs import Ligand
from clusters.algs import PartitionClustering
from clusters.algs import HierarchicalClustering
from clusters.algs import PairwiseDistance

np.random.seed(42)

def test_pdist():
    """
    checks to see if pairwise distance is returning a condensed distance metric
    """

    for i in np.arange(10):
        n, m = np.random.randint(1, 100, size=2)
        mat = np.random.random((n, m))
        expected_size = (n * (n-1)) / 2
        distvec = PairwiseDistance(mat)

        assert expected_size == distvec.size, "Unexepected condensed vector"


def test_partitioning():
    """
    verifies that kmeans returns expected number of clusters
    """

    for i in np.arange(10):

        random_k = np.random.randint(1, 10)
        n, m = np.random.randint(1, 100, size=2)
        mat = np.random.random((n, m))
        km = PartitionClustering(metric='euclidean', seed=42)
        labels = km.cluster(mat, k=random_k)

        assert_msg = "Unexpected Number of Clusters"
        assert np.unique(labels).size == random_k, assert_msg


def test_hierarchical():
    """
    verifies that algorithm is performing deterministically
    """

    n, m = np.random.randint(1, 100, size=2)
    mat = np.random.random((n, m))
    hc = HierarchicalClustering()
    zmat1 = hc.cluster(mat, precomputed=False)
    zmat2 = hc.cluster(mat, precomputed=False)

    assert (zmat1 - zmat2).sum() == 0
