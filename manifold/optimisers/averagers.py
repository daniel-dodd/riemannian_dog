"""Averagers for the optimiser iterates."""

from typing import (
    NamedTuple,
    Tuple,
)

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array  # noqa: TCH002
from jaxtyping import Float  # noqa: TCH002
from jaxtyping import PyTree  # noqa: TCH002
from optax import (
    GradientTransformation,
    safe_int32_increment,
)

from manifold.geometry import AbstractManifold  # noqa: TCH001
from manifold.geometry.zeta import (
    distance_div_sqrt_zeta,
    sqaured_distance_div_zeta,
)
from manifold.optimisers.transformations import (
    RDoGState,
    RDoWGState,
)

__all__ = [
    "default_averager",
    "uniform_averager",
    "rdog_averager",
    "rdowg_averager",
]


class AveragerState(NamedTuple):
    """State for the Averager optimiser."""

    weight_sum: Float[Array, ""]


def default_averager(optim_state: NamedTuple) -> GradientTransformation:
    """Default averager for the optimiser iterates.

    Args:
        optim_state: Optimiser state.

    Returns:
        Gradient transformation.
    """

    if isinstance(optim_state, RDoGState):
        return rdog_averager(optim_state)

    elif isinstance(optim_state, RDoWGState):
        return rdowg_averager(optim_state)

    return uniform_averager(optim_state)


def rdog_averager(optim_state: NamedTuple) -> GradientTransformation:
    """Riemannian distance over gradients (RDoG) averager for the optimiser iterates.

    Args:
        optim_state: Optimiser state.
        curvature: Whether to use the curvature in the RDoG metric.

    Returns:
        Gradient transformation.
    """

    # Whether to use the curvature term in the averaging.
    curvature = optim_state.curvature is True

    if not isinstance(optim_state, RDoGState):
        raise ValueError("Optimiser state must be an RDoGState.")

    def init(
        manifold: AbstractManifold, params: PyTree
    ) -> Tuple[PyTree[Float], AveragerState]:
        """Initialise the weighted average of the optimiser iterates.

        Args:
            manifold: Manifold to use.
            params: Current parameters.

        Returns:
            Initial weighted average and averager state.
        """
        # Get initial max distance setting of initial iterate.
        max_dist = optim_state.max_dist

        # Compute weights of initial iterate.
        weights = (
            max_dist if not curvature else distance_div_sqrt_zeta(manifold, max_dist)
        )

        # Initialise weighted average.
        return params, AveragerState(weight_sum=weights)

    def update(
        manifold: AbstractManifold,
        params: PyTree[Float],
        optim_state: NamedTuple,
        averager_state: AveragerState,
        weighted_average: PyTree[Float],
    ) -> Tuple[PyTree[Float], AveragerState]:
        """Update the weighted average of the optimiser iterates.

        Args:
            manifold: Manifold to use.
            params: Current parameters.
            optim_state: Optimiser state.
            averager_state: Averager state.
            weighted_average: Weighted average of the optimiser iterates.

        Returns:
            Updated weighted average and averager state.
        """
        # Get max distance of the previous iterates.
        max_dist = optim_state.max_dist
        init_params = optim_state.init_params
        weight_sum = averager_state.weight_sum

        # Compute max distance of the current iterate.
        max_dist = jnp.maximum(max_dist, manifold.distance(init_params, params))

        # Compute logarithm of the latest iterate.
        log_weighted_average = manifold.log(weighted_average, params)

        # Compute weights of current iterate.
        weights = (
            max_dist if not curvature else distance_div_sqrt_zeta(manifold, max_dist)
        )

        # Compute weight sum.
        weight_sum += weights

        # Compute tangent space updates.
        updates = jtu.tree_map(
            lambda lwa: weights / weight_sum * lwa,
            log_weighted_average,
        )

        # Update weighted average.
        return manifold.exp(weighted_average, updates), AveragerState(
            weight_sum=weight_sum
        )

    return GradientTransformation(init, update)


def rdowg_averager(optim_state: NamedTuple) -> GradientTransformation:
    """Riemannian distance of weighted gradients (RDoWG) averager for the optimiser iterates.

    Args:
        optim_state: Optimiser state.

    Returns:
        Gradient transformation.
    """

    curvature = optim_state.curvature is True

    if not isinstance(optim_state, RDoWGState):
        raise ValueError("Optimiser state must be an RDoWGState.")

    def init(
        manifold: AbstractManifold, params: PyTree
    ) -> Tuple[PyTree[Float], AveragerState]:
        """Initialise the weighted average of the optimiser iterates.

        Args:
            manifold: Manifold to use.
            params: Current parameters.

        Returns:
            Initial weighted average and averager state.
        """
        # Get initial max distance setting of initial iterate.
        max_dist = optim_state.max_dist

        # Compute weights of initial iterate.
        weights = (
            max_dist**2
            if not curvature
            else sqaured_distance_div_zeta(manifold, max_dist)
        )

        # Initialise weighted average.
        return params, AveragerState(weight_sum=weights)

    def update(
        manifold: AbstractManifold,
        params: PyTree[Float],
        optim_state: NamedTuple,
        averager_state: AveragerState,
        weighted_average: PyTree[Float],
    ) -> Tuple[PyTree[Float], AveragerState]:
        """Update the weighted average of the optimiser iterates.

        Args:
            manifold: Manifold to use.
            params: Current parameters.
            optim_state: Optimiser state.
            averager_state: Averager state.
            weighted_average: Weighted average of the optimiser iterates.

        Returns:
            Updated weighted average and averager state.
        """
        # Get max distance of the previous iterates.
        max_dist = optim_state.max_dist
        init_params = optim_state.init_params
        weight_sum = averager_state.weight_sum

        # Compute max distance of the current iterate.
        max_dist = jnp.maximum(max_dist, manifold.distance(init_params, params))

        # Compute logarithm of the latest iterate.
        log_weighted_average = manifold.log(weighted_average, params)

        # Compute weights of current iterate.
        weights = (
            max_dist**2
            if not curvature
            else sqaured_distance_div_zeta(manifold, max_dist)
        )

        # Compute weight sum.
        weight_sum += weights

        # Compute tangent space updates.
        updates = jtu.tree_map(
            lambda lwa: weights / weight_sum * lwa,
            log_weighted_average,
        )

        # Update weighted average.
        return manifold.exp(weighted_average, updates), AveragerState(
            weight_sum=weight_sum
        )

    return GradientTransformation(init, update)


def uniform_averager(optim_state: NamedTuple) -> GradientTransformation:
    """Uniform averager for the optimiser iterates.

    Args:
        optim_state: Optimiser state.

    Returns:
        Gradient transformation.
    """
    del optim_state

    def init(
        manifold: AbstractManifold, params: PyTree
    ) -> Tuple[PyTree[Float], AveragerState]:
        """Initialise the weighted average of the optimiser iterates.

        Args:
            manifold: Manifold to use.
            params: Current parameters.

        Returns:
            Initial weighted average and averager state.
        """
        del manifold
        return params, AveragerState(weight_sum=jnp.zeros([], jnp.int32))

    def update(
        manifold: AbstractManifold,
        params: PyTree[Float],
        optim_state: NamedTuple,
        averager_state: AveragerState,
        weighted_average: PyTree[Float],
    ) -> Tuple[PyTree[Float], AveragerState]:
        """Update the weighted average of the optimiser iterates.

        Args:
            manifold: Manifold to use.
            params: Current parameters.
            optim_state: Optimiser state.
            averager_state: Averager state.
            weighted_average: Weighted average of the optimiser iterates.

        Returns:
            Updated weighted average and averager state.
        """
        # Get count.
        count = averager_state.weight_sum

        # Increment count.
        count = safe_int32_increment(count)

        # Compute logarithm of the latest iterate.
        log_weighted_average = manifold.log(weighted_average, params)

        # Compute tangent space updates.
        updates = jtu.tree_map(lambda lwa: lwa / count, log_weighted_average)

        # Update weighted average.
        return manifold.exp(weighted_average, updates), AveragerState(weight_sum=count)

    return GradientTransformation(init, update)
