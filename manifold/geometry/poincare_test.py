"""Test Poincaré manifold."""
from jax import (
    config,
    random,
)
import jax.numpy as jnp
import numpy as np
from numpy import testing as np_testing
import pytest

config.update("jax_enable_x64", True)

from manifold.geometry.base import AbstractManifold
from manifold.geometry.poincare import (
    Poincare,
    conformal_factor,
)


@pytest.mark.parametrize("dimension", [1, 3, 5])
def test_conformal_factor(dimension: int) -> None:
    """Test the conformal factor dot product."""
    key = random.PRNGKey(123)
    manifold = Poincare(dimension)
    point = manifold.random_point(key) / 2
    np_testing.assert_allclose(
        1 - 2 / conformal_factor(point), np.linalg.norm(point) ** 2
    )


class TestPoincareManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seed = seed = 123
        self.dimension = dimension = 10

        self.key = random.PRNGKey(seed)
        self.manifold = Poincare(dimension)

    def test_init(self):
        assert self.manifold.dimension == self.dimension

    def test_curvature(self):
        assert self.manifold.curvature_bound == -1.0

    def test_type(self):
        assert isinstance(self.manifold, AbstractManifold)

    def test_distance(self):
        # Generate three random points on the manifold.
        key, subkey = random.split(self.key)
        key_x, key_y, key_z = random.split(subkey, 3)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)
        z = self.manifold.random_point(key_z)

        # Test separability.
        np_testing.assert_almost_equal(self.manifold.distance(x, x), 0.0)

        # Test symmetry.
        np_testing.assert_allclose(
            self.manifold.distance(x, y), self.manifold.distance(y, x)
        )

        # Test triangle inequality.
        assert self.manifold.distance(x, y) <= self.manifold.distance(
            x, z
        ) + self.manifold.distance(z, y)

        # check that distance is consistent with log.
        np_testing.assert_allclose(
            self.manifold.distance(x, y),
            self.manifold.norm(x, self.manifold.log(x, y)),
        )

        # Check that distance is consistent with the "correct" distance.
        correct_dist = jnp.arccosh(
            1
            + 2
            * jnp.linalg.norm(x - y) ** 2
            / (1 - jnp.linalg.norm(x) ** 2)
            / (1 - jnp.linalg.norm(y) ** 2)
        )
        np_testing.assert_allclose(correct_dist, self.manifold.distance(x, y))

    def test_inner_product(self):
        # Generate three random points on the manifold.
        key_x, key_y, key_z = random.split(self.key, 3)
        x = self.manifold.random_point(key_x) / 2
        y = self.manifold.random_point(key_y)
        z = self.manifold.random_point(key_z)

        # Use two latter points to generate two random tangent vector at x.
        u = self.manifold.log(x, y)
        v = self.manifold.log(x, z)

        # Test the result of applying inner_product with the "correct" inner product.
        np_testing.assert_allclose(
            (2 / (1 - jnp.linalg.norm(x) ** 2)) ** 2 * jnp.inner(u, v),
            self.manifold.inner_product(x, u, v),
        )

        # Test that angles are preserved.
        cos_eangle = jnp.sum(u * v) / jnp.linalg.norm(u) / jnp.linalg.norm(v)
        cos_rangle = (
            self.manifold.inner_product(x, u, v)
            / self.manifold.norm(x, u)
            / self.manifold.norm(x, v)
        )
        np_testing.assert_allclose(cos_rangle, cos_eangle)

        # Test symmetry.
        np_testing.assert_allclose(
            self.manifold.inner_product(x, u, v),
            self.manifold.inner_product(x, v, u),
        )

    def test_euclidean_to_riemannian_gradient(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        eg = self.manifold.log(x, y)

        # Test the result of applying euclidean_to_riemannian_gradient with the "correct" gradient.
        rg = self.manifold.euclidean_to_riemannian_gradient(x, eg)

        # Compute conformal factor.
        lambda_x = conformal_factor(x)

        # Riemannian gradients are rescaled Euclidean gradients
        truth = eg / lambda_x**2
        assert rg.shape == eg.shape
        assert jnp.linalg.norm(rg - truth) / jnp.linalg.norm(truth) < 1e-8

    def test_norm(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # Test the result of applying norm with the "correct" norm.
        np_testing.assert_allclose(
            2 / (1 - jnp.linalg.norm(x) ** 2) * jnp.linalg.norm(u),
            self.manifold.norm(x, u),
        )

    def test_random_point(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Ensure that the points are generated on the manifold.
        np_testing.assert_array_less(np.linalg.norm(x), 1)
        np_testing.assert_array_less(np.linalg.norm(y), 1)

        # Ensure that the two points are different.
        assert self.manifold.distance(x, y) > 1e-6

    def test_transport(self):
        # Generate three random points on the manifold.
        key_x, key_y, key_z = random.split(self.key, 3)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)
        z = self.manifold.random_point(key_z)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, z)

        # Test the parallel transport of u from x to y.
        result = self.manifold.parallel_transport(x, y, u)

        # Check parallel transport preserves norm.
        np_testing.assert_allclose(
            self.manifold.norm(y, result), self.manifold.norm(x, u)
        )

    def test_exp(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # Test exponential map gives point on manifold.
        np_testing.assert_array_less(self.manifold.exp(x, u), 1)

        # Test that for small vectors exp(x, u) = x + u.
        u = u * 1e-6
        np_testing.assert_allclose(self.manifold.exp(x, u), x + u)

    def test_exp_log_inverse(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # We now undo the log by applying exp, and assert that we get y back.
        np_testing.assert_allclose(self.manifold.exp(x, u), y)

        # And vice versa. We take the exp first and then apply log, and assert we get u back.
        z = self.manifold.exp(x, u)
        np.testing.assert_allclose(self.manifold.log(x, z), u)
