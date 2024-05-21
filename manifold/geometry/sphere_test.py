"""Test Sphere manifold. Adapted from Pymanopt."""
from jax import random
import numpy as np
from numpy import testing as np_testing
import pytest

from manifold.geometry.base import AbstractManifold
from manifold.geometry.sphere import Sphere


class TestSphereManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seed = seed = 123
        self.dimension = dimension = 100

        self.key = random.PRNGKey(seed)
        self.manifold = Sphere(dimension)

    def test_dim(self):
        assert self.manifold.dimension == 100

    def test_curvature(self):
        assert self.manifold.curvature_bound == 1.0

    def test_type(self):
        assert isinstance(self.manifold, AbstractManifold)

    def test_distance(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # We now compare the result of applying distance with the "correct" distance
        correct_dist = np.arccos(np.dot(x, y))
        np_testing.assert_almost_equal(correct_dist, self.manifold.distance(x, y))

    def test_inner_product(self):
        # Generate three random points on the manifold.
        key_x, key_y, key_z = random.split(self.key, 3)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)
        z = self.manifold.random_point(key_z)

        # Use the two latter points to generate a two random tangent vectors at x.
        u = self.manifold.log(x, y)
        v = self.manifold.log(x, z)

        # We now compare the result of applying inner_product with the "correct" inner product
        np_testing.assert_almost_equal(
            np.sum(u * v), self.manifold.inner_product(x, u, v)
        )

    def test_euclidean_to_riemannian_gradient(self):
        # Generate a random point x on the manifold.
        key, subkey = random.split(self.key)
        x = self.manifold.random_point(subkey)

        #  Construct a vector h in the ambient space.
        key, subkey = random.split(key)
        h = random.normal(subkey, (self.dimension,))

        #  Compare the projections.
        np_testing.assert_array_almost_equal(
            h - x * np.dot(x, h),
            self.manifold.euclidean_to_riemannian_gradient(x, h),
        )

    def test_retraction(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # Test that the result is on the manifold./
        xretru = self.manifold.retraction(x, u)
        np_testing.assert_almost_equal(np.linalg.norm(xretru), 1)

        # Test for small tangent vectors it has little effect.
        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)
        np_testing.assert_allclose(xretru, x + u)

    def test_norm(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # We now compare the result of applying norm with the "correct" norm.
        np_testing.assert_almost_equal(self.manifold.norm(x, u), np.linalg.norm(u))

    def test_random_point(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Ensure that the first point is generated on the manifold.
        np_testing.assert_almost_equal(np.linalg.norm(x), 1)

        # And do the same for the second point.
        np_testing.assert_almost_equal(np.linalg.norm(y), 1)

        # Ensure that the two points are different.
        assert np.linalg.norm(x - y) > 1e-3

    def test_transport(self):
        # Generate three random points on the manifold.
        key_x, key_y, key_z = random.split(self.key, 3)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)
        z = self.manifold.random_point(key_z)

        # Use the latter points to generate a random tangent vector at x.
        u = self.manifold.log(x, z)

        # We now compare the result of applying transport with the "correct" transport which is the projection (euclidean_to_riemannian_gradient).
        np_testing.assert_allclose(
            self.manifold.parallel_transport(x, y, u),
            self.manifold.euclidean_to_riemannian_gradient(y, u),
        )

    def test_exp_log_inverse(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        yexplog = self.manifold.exp(x, self.manifold.log(x, y))
        np_testing.assert_array_almost_equal(y, yexplog)

    def test_log_exp_inverse(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        ulogexp = self.manifold.log(x, self.manifold.exp(x, u))
        np_testing.assert_array_almost_equal(u, ulogexp)
