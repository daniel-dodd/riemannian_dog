"""Zeta function that depends on the curvature of the manifold."""
from jax import lax
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from manifold.geometry.base import AbstractManifold

__all__ = [
    "zeta",
]

EPSILON = 1e-15


def zeta(manifold: AbstractManifold, distance: Float[Array, ""]) -> Float[Array, ""]:
    """Zeta function that depends on the curvature of the manifold.

    Args:
        manifold: The manifold.
        distance: The distance.

    Returns:
        The zeta function.
    """
    kappa = manifold.curvature_bound
    constant = jnp.sqrt(jnp.abs(kappa))
    constant_times_distance = jnp.clip(constant * distance, EPSILON)

    return lax.cond(
        kappa >= 0.0,
        lambda: jnp.array(1.0),
        lambda: jnp.squeeze(
            constant_times_distance / jnp.tanh(constant_times_distance)
        ),
    )


def distance_div_sqrt_zeta(
    manifold: AbstractManifold, distance: Float[Array, ""]
) -> Float[Array, ""]:
    """Distance divided by the square root of the zeta function.

    Args:
        manifold: The manifold.
        distance: The distance.

    Returns:
        The distance divided by the square root of the zeta function.
    """
    kappa = manifold.curvature_bound
    constant = jnp.sqrt(jnp.abs(kappa))

    return lax.cond(
        kappa >= 0.0 - EPSILON,
        lambda: jnp.squeeze(distance),
        lambda: jnp.squeeze(
            jnp.sqrt(distance * jnp.tanh(constant * distance) / constant)
        ),
    )


def sqaured_distance_div_zeta(
    manifold: AbstractManifold, distance: Float[Array, ""]
) -> Float[Array, ""]:
    """Squared distance divided by the zeta function.

    Args:
        manifold: The manifold.
        distance: The distance.

    Returns:
        The squared distance divided by the zeta function.
    """
    kappa = manifold.curvature_bound
    constant = jnp.sqrt(jnp.abs(kappa))
    return lax.cond(
        kappa >= 0.0 - EPSILON,
        lambda: jnp.squeeze(distance**2),
        lambda: jnp.squeeze(distance * jnp.tanh(constant * distance) / constant),
    )
