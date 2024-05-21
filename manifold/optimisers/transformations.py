"""Transformations for Riemannian optimisers."""
from typing import (
    Callable,
    NamedTuple,
    Tuple,
)

from jax import tree_util as jtu
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Int,
    PyTree,
)
from optax import (
    EmptyState,
    GradientTransformation,
    bias_correction,
    safe_int32_increment,
)

from manifold.geometry import AbstractManifold
from manifold.geometry.zeta import (
    distance_div_sqrt_zeta,
    sqaured_distance_div_zeta,
)

__all__ = [
    "scale_by_learning_rate",
    "scale_by_radam",
    "scale_by_rdog",
    "scale_by_rdowg",
    "scale_by_nrdog",
]


# STANDARD PARAMETERISED OPTIMISERS:


def scale(step_size: Float[Array, ""]) -> GradientTransformation:
    """Scale updates by some fixed scalar `step_size`.

    Args:
      step_size: The scaling factor.
    """

    def init_fn(manifold: AbstractManifold, params: PyTree) -> EmptyState:
        del params
        del manifold
        return EmptyState()

    def update_fn(
        manifold: AbstractManifold, updates, state, params=None
    ) -> Tuple[PyTree, EmptyState]:
        del params
        del manifold
        updates = jtu.tree_map(lambda u: step_size * u, updates)
        return updates, state

    return GradientTransformation(init_fn, update_fn)


class ScaleByScheduleState(NamedTuple):
    """Maintains count for scale scheduling."""

    count: Int[Array, ""]


def scale_by_schedule(
    step_size_fn: Callable[[Int[Array, ""]], Float[Array, ""]]
) -> GradientTransformation:
    """Scale updates using a custom schedule for the `step_size`.

    Args:
      step_size_fn: A function that takes an update count as input and proposes
        the step_size to multiply the updates by.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(manifold: AbstractManifold, params):
        del params
        del manifold
        return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def update_fn(manifold: AbstractManifold, updates, state, params=None):
        del params
        del manifold
        step_size = step_size_fn(state.count)
        updates = jtu.tree_map(
            lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates
        )
        return updates, ScaleByScheduleState(count=safe_int32_increment(state.count))

    return GradientTransformation(init_fn, update_fn)


def scale_by_learning_rate(
    learning_rate: Float[Array, ""], flip_sign: bool = True
) -> GradientTransformation:
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return scale_by_schedule(lambda count: m * learning_rate(count))
    return scale(m * learning_rate)


class RadamState(NamedTuple):
    """State for the Riemannian Adam algorithm."""

    count: Int[Array, ""]
    tau: PyTree[Float]
    mu: PyTree[Float]
    nu: PyTree[Float[Array, ""]]
    previous: PyTree[Float]


def scale_by_radam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 1e-8,
    ams_grad: bool = False,
) -> GradientTransformation:
    """Rescale updates according to the RADAM algorithm.

    References:
      [Gary Becigneul et al, 2019](https://arxiv.org/abs/1810.00760)

    Args:
        learning_rate: The learning rate.
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the second moment estimates.
        eps: A small constant for numerical stability.
        eps_root: A small constant for numerical stability.
        ams_grad: Whether to use the AMSGrad variant of this algorithm.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(manifold: AbstractManifold, params: PyTree) -> RadamState:
        mu = jtu.tree_map(jnp.zeros_like, params)  # First moment
        tau = jtu.tree_map(jnp.zeros_like, params)  # Parallel transport term

        return RadamState(
            count=jnp.zeros([], jnp.int32),
            tau=tau,
            mu=mu,
            nu=jnp.zeros([]),
            previous=params,
        )

    def update_fn(
        manifold: AbstractManifold, updates: PyTree, state: RadamState, params: PyTree
    ) -> Tuple[PyTree, RadamState]:
        # Unpack the state.
        count = state.count
        tau = state.tau
        mu = state.mu
        nu = state.nu
        previous = state.previous

        # Increment the count.
        count = safe_int32_increment(count)

        # Key difference between Riemannian and Eucliean is that
        # first moment has to transportation to current_params.
        tau = manifold.parallel_transport(previous, params, mu)
        mu = jtu.tree_map(lambda t, u: b1 * t + (1 - b1) * u, tau, updates)

        # Update nu.
        nu += manifold.norm(params, updates) ** 2

        # AMS variant.
        nu = jtu.tree_map(jnp.max, nu, state.nu) if ams_grad else nu

        # Bias correction.
        mu_hat = bias_correction(mu, b1, count)
        nu_hat = bias_correction(nu, b2, count)

        # Updates.
        updates = jtu.tree_map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        )

        return updates, RadamState(count=count, mu=mu, nu=nu, previous=params, tau=tau)

    return GradientTransformation(init_fn, update_fn)


# PARAMETER-FREE OPTIMISERS:


class RDoGState(NamedTuple):
    """State for Riemannian DoG."""

    max_dist: PyTree[Float[Array, ""]]
    sqaured_grad_norm_sum: PyTree[Float[Array, ""]]
    init_params: PyTree[Float]
    curvature: bool


def scale_by_rdog(
    reps: float = 1e-6,
    eps: float = 1e-8,
    curvature: bool = True,
) -> GradientTransformation:
    """RDoG hyperparameter-free optimiser.

    Args:
      reps: Small loading term to avoid zero learning rates and divide-by-zero
        errors.
      eps: Small constant for numerical stability.
      curvature: Whether to use curvature correction or not.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(manifold: AbstractManifold, params: PyTree) -> RDoGState:
        # Initialise the state.
        return RDoGState(
            max_dist=jnp.ones([]) * reps,
            sqaured_grad_norm_sum=jnp.zeros([]),
            init_params=params,
            curvature=curvature,
        )

    def update_fn(
        manifold: AbstractManifold, updates: PyTree, state: RDoGState, params: PyTree
    ) -> Tuple[PyTree, RDoGState]:
        # Unpack the state.
        max_dist = state.max_dist
        init_params = state.init_params
        sqaured_grad_norm_sum = state.sqaured_grad_norm_sum

        # Update the maximum distance from the initial point.
        max_dist = jnp.maximum(max_dist, manifold.distance(init_params, params))

        # Update the squared gradient norm sum.
        sqaured_grad_norm_sum += manifold.norm(params, updates) ** 2

        # Compute weights.
        weights = (
            max_dist if not curvature else distance_div_sqrt_zeta(manifold, max_dist)
        )

        # Compute the updates.
        updates = jtu.tree_map(
            lambda u: -weights / (jnp.sqrt(sqaured_grad_norm_sum) + eps) * u,
            updates,
        )

        return updates, RDoGState(
            max_dist=max_dist,
            sqaured_grad_norm_sum=sqaured_grad_norm_sum,
            init_params=init_params,
            curvature=curvature,
        )

    return GradientTransformation(init_fn, update_fn)


class RDoWGState(NamedTuple):
    """State for Riemannian DoWG."""

    max_dist: PyTree[Float[Array, ""]]
    sqaured_grad_norm_sum: PyTree[Float[Array, ""]]
    init_params: PyTree[Float]
    curvature: bool


def scale_by_rdowg(
    reps: float = 1e-6,
    eps: float = 1e-8,
    curvature: bool = True,
) -> GradientTransformation:
    """RDoWG hyperparameter-free optimiser.

    Args:
      reps: Small loading term to avoid zero learning rates and divide-by-zero
        errors.
      eps: Small constant for numerical stability.
      curvature: Whether to use curvature correction or not.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(manifold: AbstractManifold, params: PyTree) -> RDoWGState:
        # Initialise the state.
        return RDoWGState(
            max_dist=jnp.ones([]) * reps,
            sqaured_grad_norm_sum=jnp.zeros([]),
            init_params=params,
            curvature=curvature,
        )

    def update_fn(
        manifold: AbstractManifold, updates: PyTree, state: RDoWGState, params: PyTree
    ) -> Tuple[PyTree, RDoWGState]:
        # Unpack the state.
        max_dist = state.max_dist
        init_params = state.init_params
        sqaured_grad_norm_sum = state.sqaured_grad_norm_sum

        # Update the maximum distance from the initial point.
        max_dist = jnp.maximum(max_dist, manifold.distance(init_params, params))

        # Compute weights.
        weights = (
            max_dist**2
            if not curvature
            else sqaured_distance_div_zeta(manifold, max_dist)
        )

        # Update weighted squared gradient norm sum.
        sqaured_grad_norm_sum += weights * (manifold.norm(params, updates) ** 2)

        # Compute the updates.
        updates = jtu.tree_map(
            lambda u: -weights / (jnp.sqrt(sqaured_grad_norm_sum) + eps) * u,
            updates,
        )

        return updates, RDoWGState(
            max_dist=max_dist,
            sqaured_grad_norm_sum=sqaured_grad_norm_sum,
            init_params=init_params,
            curvature=curvature,
        )

    return GradientTransformation(init_fn, update_fn)


def scale_by_nrdog(
    reps: float = 1e-6,
    eps: float = 1e-8,
    curvature=True,
) -> GradientTransformation:
    """Normalised RDoG hyperparameter-free optimiser.

    Args:
      reps: Small loading term to avoid zero learning rates and divide-by-zero
        errors.
      eps: Small constant for numerical stability.
      curvature: Whether to use curvature correction or not.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(manifold: AbstractManifold, params: PyTree) -> RDoGState:
        # Initialise the state.
        return RDoGState(
            max_dist=jnp.ones([]) * reps,
            sqaured_grad_norm_sum=jnp.zeros([], dtype=jnp.int32),
            init_params=params,
            curvature=curvature,
        )

    def update_fn(
        manifold: AbstractManifold, updates: PyTree, state: RDoGState, params: PyTree
    ) -> Tuple[PyTree, RDoGState]:
        # Unpack the state.
        max_dist = state.max_dist
        init_params = state.init_params
        count = state.sqaured_grad_norm_sum

        # Update the maximum distance from the initial point.
        max_dist = jnp.maximum(max_dist, manifold.distance(init_params, params))

        # Increment the count.
        count = safe_int32_increment(count)

        # Compute weights.
        weights = (
            max_dist if not curvature else distance_div_sqrt_zeta(manifold, max_dist)
        )

        # Compute the updates.
        updates = jtu.tree_map(
            lambda u: -weights
            / (jnp.sqrt(count) * manifold.norm(params, updates) + eps)
            * u,
            updates,
        )

        return updates, RDoGState(
            max_dist=max_dist,
            sqaured_grad_norm_sum=count,
            init_params=init_params,
            curvature=curvature,
        )

    return GradientTransformation(init_fn, update_fn)
