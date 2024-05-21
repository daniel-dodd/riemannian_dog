"""Test the Euclidean manifold."""
from jax import (
    config,
    random,
)

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpy import testing as np_testing
import pytest

from manifold.geometry.base import AbstractManifold
from manifold.geometry.euclidean import Euclidean


class TestEuclideanManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seed = seed = 123
        self.dimension = dimension = 100

        self.key = random.PRNGKey(seed)
        self.manifold = Euclidean(dimension)

    def test_type(self):
        assert isinstance(self.manifold, AbstractManifold)
        assert isinstance(self.manifold, Euclidean)
        assert self.manifold.dimension == self.dimension

    def test_curvature(self):
        # The Grassmann manifold has positive curvature.
        assert self.manifold.curvature_bound == 0.0

    def test_distance(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # We now compare the result of applying distance with norm.
        np_testing.assert_almost_equal(
            self.manifold.distance(x, y),
            jnp.linalg.norm(x - y),
        )

    def test_norm(self):
        # Generate a random point on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Test against euclidean norm.
        np_testing.assert_almost_equal(
            self.manifold.norm(x, y),
            jnp.linalg.norm(y),
        )

        # We now compare the result of applying norm with distance.
        np_testing.assert_almost_equal(
            self.manifold.norm(x, self.manifold.log(x, y)),
            self.manifold.distance(x, y),
        )

    def test_random_point(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Ensure that the first point is generated on the manifold.
        assert x.shape == (self.dimension,)

        # And do the same for the second point.
        assert y.shape == (self.dimension,)

        # Ensure that the two points are different.
        assert np.linalg.norm(x - y) > 1e-6

    def test_exp_log_inverse(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # We now compare the result of applying exp and log.
        z = self.manifold.exp(x, u)

        # And ensure this difference is zero.
        np_testing.assert_almost_equal(0, self.manifold.distance(y, z), decimal=5)

    def test_log_exp_inverse(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # We now compare the result of applying exp and log.
        y = self.manifold.exp(x, u)
        v = self.manifold.log(x, y)

        # And ensure this difference is zero.
        np_testing.assert_almost_equal(0, self.manifold.norm(x, u - v))
