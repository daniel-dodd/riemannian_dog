"""Test the Zeta function that depends on the curvature of the manifold."""
from jax import jit
import jax.numpy as jnp
import pytest

from manifold.geometry.base import AbstractManifold
from manifold.geometry.grassmann import Grassmann
from manifold.geometry.poincare import Poincare
from manifold.geometry.sphere import Sphere
from manifold.geometry.zeta import zeta

EPSILON = 1e-32


@pytest.mark.parametrize("distance", [0.0, 1.0, 2.0])
@pytest.mark.parametrize(
    "manifold",
    [
        Grassmann(10, 2),
        Sphere(3),
        Poincare(10),
    ],
)
def test_zeta(manifold: AbstractManifold, distance: float) -> None:
    kappa = manifold.curvature_bound

    jitted_zeta = jit(zeta, static_argnums=(0,))

    if kappa >= 0.0:
        # Zeta function is 1 for positive curvature.
        assert zeta(manifold, distance) == 1.0
        assert jitted_zeta(manifold, distance) == 1.0

    else:
        # Zeta function is sqrt{|kappa|} * distance / tanh(sqrt{|kappa|} * distance) for negative curvature.
        constant = jnp.sqrt(jnp.abs(kappa))
        true = (
            1.0
            if distance == 0.0
            else constant * distance / jnp.tanh(constant * distance)
        )
        result = zeta(manifold, distance)
        assert jnp.allclose(true, result)

        jitted_result = jitted_zeta(manifold, distance)
        assert jnp.allclose(true, jitted_result)
