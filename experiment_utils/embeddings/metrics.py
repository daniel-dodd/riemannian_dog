"""Metrics for evaluating WordNet embeddings. """
from jax import vmap
from jaxtyping import (
    Array,
    Float,
)
import numpy as np
from sklearn.metrics import average_precision_score

from manifold.geometry import AbstractManifold

__all__ = [
    "pairwise_distance",
    "reconstruction_metrics",
]


def pairwise_distance(
    manifold: AbstractManifold,
    embeddings: Float[Array, "N D"],
) -> Float[Array, "N N"]:
    """Compute pairwise distances between embeddings.

    Args:
        manifold: The manifold used for the embeddings.
        embeddings: Embedding table with `N` embeddings and `dim`
            dimensionality.

    Returns:
        Pairwise distances between embeddings.
    """
    return vmap(lambda yi: vmap(lambda xi: manifold.distance(xi, yi))(embeddings))(
        embeddings
    )


def reconstruction_metrics(
    embedding_manifold: AbstractManifold,
    adjacency: dict[int, set[int]],
    embeddings: Float[Array, "N D"],
):
    """
    Reconstruction evaluation metrics. Adapted from Poincare Embeddings code @ https://github.com/facebookresearch/poincare-embeddings/blob/main/hype/graph.py.

    For each object, rank its neighbors by distance
    and compute the average precision of the neighbors
    being ranked before non-neighbors.

    Args:
        embedding_manifold: The manifold used for the embeddings
        adjacency: Adjacency list mapping objects to its neighbors
        embeddings: Embedding table with `N` embeddings and `dim`
            dimensionality
    """
    objects = np.array(list(adjacency.keys()))

    ranksum = nranks = ap_scores = iters = 0
    labels = np.empty(embeddings.shape[0])
    dists_all = np.array(pairwise_distance(embedding_manifold, embeddings))

    for s in objects:
        labels.fill(0)
        neighbors = np.array(list(adjacency[s]), dtype=np.int32)
        dists = dists_all[s]
        dists[s] = 1e12
        sorted_idx = np.argsort(dists)
        (ranks,) = np.where(np.in1d(sorted_idx, neighbors))
        # The above gives us the position of the neighbors in sorted order.  We
        # want to count the number of non-neighbors that occur before each neighbor
        ranks += 1
        N = ranks.shape[0]

        # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
        # As an example, assume the ranks of the neighbors are:
        # 0, 1, 4, 5, 6, 8
        # For each neighbor, we'd like to return the number of non-neighbors
        # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
        # Another way of thinking about it is to return
        # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
        # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
        # Note that we include `N` to account for the source embedding itself
        # always being the nearest neighbor
        ranksum += ranks.sum() - (N * (N - 1) / 2)
        nranks += ranks.shape[0]
        labels[neighbors] = 1
        ap_scores += average_precision_score(labels, -dists)
        iters += 1

    return float(ranksum) / nranks, float(ap_scores) / iters
