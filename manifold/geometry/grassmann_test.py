"""Test the Grassman manifold. Inspired from Pymanopt library."""
from jax import (
    config,
    random,
)

config.update("jax_enable_x64", True)

import numpy as np
from numpy import testing as np_testing
import pytest

from manifold.geometry.base import AbstractManifold
from manifold.geometry.grassmann import Grassmann
from manifold.geometry.utils import multitransp


class TestGrassmannManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.seed = seed = 123
        self.row_dimension = row_dimension = 5
        self.column_dimension = column_dimension = 2

        self.key = random.PRNGKey(seed)
        self.manifold = Grassmann(row_dimension, column_dimension)

    def test_type(self):
        assert isinstance(self.manifold, AbstractManifold)

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
            self.manifold.norm(x, self.manifold.log(x, y)),
        )

    def test_retraction(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Use the latter to generate a random tangent vector at x.
        u = self.manifold.log(x, y)

        # We now compare the result of applying retraction.
        xretru = self.manifold.retraction(x, u)

        np_testing.assert_allclose(
            multitransp(xretru) @ xretru, np.eye(self.column_dimension), atol=1e-10
        )

        # We now consider this for small u.
        u = u * 1e-6
        xretru = self.manifold.retraction(x, u)

        # Recall the reraction here is locally taylor expansion!
        np_testing.assert_allclose(xretru, x + u)

    def test_random_point(self):
        # Generate two random points on the manifold.
        key_x, key_y = random.split(self.key)
        x = self.manifold.random_point(key_x)
        y = self.manifold.random_point(key_y)

        # Ensure that the first point is generated on the manifold.
        np_testing.assert_allclose(
            multitransp(x) @ x, np.eye(self.column_dimension), atol=1e-10
        )

        # And do the same for the second point.
        np_testing.assert_allclose(
            multitransp(y) @ y, np.eye(self.column_dimension), atol=1e-10
        )

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
