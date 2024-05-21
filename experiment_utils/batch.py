"""Batching utilities."""
from typing import (
    Any,
    Tuple,
)

import jax
from jax import vmap
import jax.tree_util as jtu
from jaxtyping import PyTree
from optax import GradientTransformation

from manifold.geometry import AbstractManifold  # noqa: TCH002


def batch_transform(optim: GradientTransformation) -> GradientTransformation:
    """Batch an optimiser when each parameter in the batch is defined on the same manifold.

    Args:
        optim: The optimiser to batch.

    Returns:
        The batched optimiser.
    """

    def init(manifold: AbstractManifold, *args: Any, **kwargs: Any) -> PyTree:
        """Initialise the optimiser state.

        Args:
            manifold: The manifold to optimise on.
            *args: The arguments to pass to the optimiser.
            **kwargs: The keyword arguments to pass to the optimiser.

        Returns:
            The optimiser state.
        """
        batched_state = vmap(lambda *_: optim.init(manifold, *_))(*args, **kwargs)
        return batched_state

    def updates(
        manifold: AbstractManifold, *args: Any, **kwargs: Any
    ) -> Tuple[PyTree, PyTree]:
        """Update the optimiser state.

        Args:
            manifold: The manifold to optimise on.
            *args: The arguments to pass to the optimiser.
            **kwargs: The keyword arguments to pass to the optimiser.

        Returns:
            The updated optimiser state.
        """
        batched_updates, batched_state = vmap(lambda *_: optim.update(manifold, *_))(
            *args, **kwargs
        )
        return batched_updates, batched_state

    return GradientTransformation(init, updates)


def gather_slice(state: PyTree, idx: slice) -> PyTree:
    """Gather a slice of a PyTree.

    Args:
        state: The PyTree to gather from.
        idx: The index of the slice to gather.

    Returns:
        The gathered slice.
    """

    def _slice_fn(leaf: Any) -> Any:
        """Slice a leaf of the State PyTree."""

        # If the leaf is an array and has more than one dimension, gather the slice.
        if isinstance(leaf, jax.Array) and leaf.ndim > 0:
            return leaf[idx]

        # Otherwise, return the leaf.
        return leaf

    return jtu.tree_map(_slice_fn, state)


def update_slice(state: PyTree, idx: slice, slice_state: PyTree) -> PyTree:
    """Update a slice of a PyTree.

    Args:
        state: The PyTree to update.
        idx: The index of the slice to update.
        slice_state: The slice to update with.

    Returns:
        The updated PyTree.
    """

    def _update_fn(leaf: Any, slice_leaf: Any) -> Any:
        """Update a leaf of the State PyTree."""

        # If the leaf is an array and has more than one dimension, update the slice.
        if isinstance(leaf, jax.Array) and leaf.ndim > 0:
            return leaf.at[idx].set(slice_leaf)

        # Otherwise, return the leaf.
        return slice_leaf

    return jtu.tree_map(_update_fn, state, slice_state)
