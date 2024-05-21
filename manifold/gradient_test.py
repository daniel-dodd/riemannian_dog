"""Test Riemannian gradient functions."""

from jax import (
    grad,
    random,
    vmap,
)
from jax.tree_util import (
    tree_map,
    tree_reduce,
)
from jaxtyping import PyTree
import pytest

from experiment_utils.objectives import frechlet
from manifold.geometry import (
    Euclidean,
    Sphere,
)
from manifold.geometry.base import AbstractManifold
from manifold.gradient import (
    rgrad,
    value_and_rgrad,
)


@pytest.mark.parametrize("manifold", [Sphere(3), Euclidean(4)])
def test_rgrad(manifold: AbstractManifold) -> None:
    # Construct the frechlet objective.
    objective = frechlet(manifold)

    # Generate point on the manifold.
    key, subkey = random.split(random.PRNGKey(123))
    point = manifold.random_point(subkey)

    # Generate a batch of points on the manifold, that will be used as the points we compute the frechlet mean of.
    points = vmap(lambda key: manifold.random_point(key))(random.split(key, 5))

    # Compute the Riemannian gradient of the frechlet objective.
    g = rgrad(manifold, objective)(point, points)

    # Check that the Riemannian gradient is a PyTree.
    assert isinstance(g, PyTree)

    # Check that the Riemannian gradient is a PyTree of the same structure as the point.
    assert tree_reduce(all, tree_map(lambda x, y: x.shape == y.shape, g, point))

    # Check against the projected Euclidean gradient.
    eg = grad(objective)(point, points)
    peg = manifold.euclidean_to_riemannian_gradient(point, eg)
    assert tree_reduce(all, tree_map(lambda x, y: (x == y).all(), g, peg))


@pytest.mark.parametrize("manifold", [Sphere(3), Euclidean(4)])
def test_value_and_rgrad(manifold: AbstractManifold) -> None:
    # Construct the frechlet objective.
    objective = frechlet(manifold)

    # Generate point on the manifold.
    key, subkey = random.split(random.PRNGKey(123))
    point = manifold.random_point(subkey)

    # Generate a batch of points on the manifold, that will be used as the points we compute the frechlet mean of.
    points = vmap(lambda key: manifold.random_point(key))(random.split(key, 5))

    # Compute the Riemannian gradient of the frechlet objective.
    g_true = rgrad(manifold, objective)(point, points)

    # Compute the value and Riemannian gradient of the frechlet objective.
    v_true = objective(point, points)

    # Now test the value and Riemannian gradient function.
    v, g = value_and_rgrad(manifold, objective)(point, points)

    # Check that the Riemannian gradient is a PyTree.
    assert isinstance(g, PyTree)

    # Check that the Riemannian gradient is a PyTree of the same structure as the point.
    assert tree_reduce(all, tree_map(lambda x, y: x.shape == y.shape, g, point))

    # Check against the Riemannian gradient.
    assert tree_reduce(all, tree_map(lambda x, y: (x == y).all(), g, g_true))

    # Check against the value.
    assert v == v_true
