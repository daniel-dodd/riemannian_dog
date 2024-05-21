"""Combine transformations for Riemannian optimisers."""
from typing import Tuple

from jaxtyping import PyTree  # noqa: TCH002
from optax import GradientTransformation

from manifold.geometry import AbstractManifold  # noqa: TCH001

__all__ = [
    "chain",
]


def chain(
    *args: GradientTransformation,
) -> GradientTransformation:
    """Applies a list of chainable update transformations.

    Given a sequence of chainable transforms, `chain` returns an `init_fn`
    that constructs a `state` by concatenating the states of the individual
    transforms, and returns an `update_fn` which chains the update transformations
    feeding the appropriate state to each.

    Args:
      *args: a sequence of chainable (init_fn, update_fn) tuples.

    Returns:
      A `GradientTransformation` object.
    """
    init_fns, update_fns = zip(*args)

    def init_fn(manifold: AbstractManifold, params: PyTree) -> Tuple:
        return tuple(fn(manifold, params) for fn in init_fns)

    def update_fn(manifold: AbstractManifold, updates, state, params=None):
        if len(update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in "
                "chain! Make sure you have called init first!"
            )

        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(manifold, updates, s, params)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return GradientTransformation(init_fn, update_fn)
