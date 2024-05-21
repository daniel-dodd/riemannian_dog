"""Batch and negative sampling for the WordNet embeddings. """
from typing import Callable

from jax import (
    random,
    vmap,
)
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Int,
    Key,
)

from experiment_data.wordnet import RelationsDataset

__all__ = [
    "construct_sampler",
]


def construct_sampler(
    dataset: RelationsDataset, batch_size: int, num_negatives: int, burnin: bool
) -> Callable[[Key], Int[Array, " Q"]]:
    """Sample negative samples uniformly.

    Args:
        batch_size: The number of objects to sample.
        num_negatives: The number of negative samples to take per object.
        num_objects: The number of objects to sample from.

    Returns:
        The sampled objects.
    """

    # Unpack dataset.
    ids = dataset.ids
    pairs = dataset.pairs
    degrees = dataset.degrees
    valid_negatives = dataset.valid_negatives

    num_pairs = len(pairs)
    num_ids = len(ids)

    # This is for a single batch.
    def sample_indicies_fn(key: Key) -> Int[Array, " Q"]:
        """Q is of dimension 2 for a sampled pair (u,v) plus the number of negative samples N(u) for u.

        Args:
            key: The key to use for sampling.

        Returns:
            The sampled indices. This comprise the pair indices (u, v) and then `num_negatives` negative samples N(u) for u.
        """

        # Split key.
        pair_key, negative_key = random.split(key)

        # Sample pair indices.
        pair_idx = pairs[random.randint(pair_key, (), minval=0, maxval=num_pairs - 1)]

        # Unpack pair indices.
        u, v = pair_idx

        # Sample negative indices for u.
        weights = degrees**0.75 if burnin else 1.0
        negative_idx = random.choice(
            negative_key,
            jnp.arange(num_ids),
            shape=(num_negatives,),
            p=weights * valid_negatives[u],
            replace=True,
        )

        # Concatenate indices.
        return jnp.concatenate([pair_idx, negative_idx])

    def sample_indicies_batch_fn(key: Key) -> Int[Array, "B Q"]:
        """Sample indices for a batch of pairs.

        Args:
            key: The key to use for sampling.

        Returns:
            The sampled indices.
        """
        return vmap(sample_indicies_fn)(random.split(key, batch_size))

    return sample_indicies_batch_fn
