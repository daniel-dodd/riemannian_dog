"""Test alias for our Riemannian optimisers!

Tests are minimal and just crudely check that the aliases give the updates we expect according to the underlying transformations.
"""
from equinox import tree_equal
from jax import random
from optax import GradientTransformation
import pytest

from manifold.geometry import (
    Euclidean,
    Sphere,
)
from manifold.optimisers.alias import (
    radam,
    rdog,
    rdowg,
    rsgd,
)
from manifold.optimisers.combine import chain
from manifold.optimisers.transformations import (
    scale_by_learning_rate,
    scale_by_radam,
    scale_by_rdog,
    scale_by_rdowg,
)


@pytest.mark.parametrize("manifold", [Euclidean(1), Sphere(3)])
@pytest.mark.parametrize("learning_rate", [1e-3, 1e-2, 1e-1])
def test_rsgd(manifold, learning_rate):
    """Test RSGD alias."""
    key = random.PRNGKey(123)

    optimiser = rsgd(learning_rate)
    equiv = scale_by_learning_rate(learning_rate)

    # Check types.
    assert isinstance(optimiser, GradientTransformation)

    # Generate random point on manifold.
    key, subkey = random.split(key)
    point = manifold.random_point(subkey)

    # Test init is the same.
    optimiser_state = optimiser.init(manifold, point)
    equiv_state = equiv.init(manifold, point)

    assert tree_equal(optimiser_state, equiv_state)

    # Test update is the same.
    key, subkey = random.split(key)
    tangent_vector = manifold.log(point, manifold.random_point(subkey))
    optimiser_update, optimiser_state = optimiser.update(
        manifold, tangent_vector, optimiser_state, point
    )
    equiv_update, equiv_state = equiv.update(
        manifold, tangent_vector, equiv_state, point
    )

    assert tree_equal(optimiser_state, equiv_state)
    assert tree_equal(optimiser_update, equiv_update)


@pytest.mark.parametrize(
    "manifold",
    [Euclidean(1), Sphere(3)],
)
@pytest.mark.parametrize("learning_rate", [1e-3, 1e-2, 1e-1])
def test_radam(manifold, learning_rate):
    """Test RADAM alias."""
    key = random.PRNGKey(123)

    optimiser = radam(learning_rate)
    equiv = chain(scale_by_radam(), scale_by_learning_rate(learning_rate))

    # Check types.
    assert isinstance(optimiser, GradientTransformation)

    # Generate random point on manifold.
    key, subkey = random.split(key)
    point = manifold.random_point(subkey)

    # Test init is the same.
    optimiser_state = optimiser.init(manifold, point)
    equiv_state = equiv.init(manifold, point)

    assert tree_equal(optimiser_state, equiv_state)

    # Test update is the same.
    key, subkey = random.split(key)
    tangent_vector = manifold.log(point, manifold.random_point(subkey))
    optimiser_update, optimiser_state = optimiser.update(
        manifold, tangent_vector, optimiser_state, point
    )
    equiv_update, equiv_state = equiv.update(
        manifold, tangent_vector, equiv_state, point
    )

    assert tree_equal(optimiser_state, equiv_state)
    assert tree_equal(optimiser_update, equiv_update)


@pytest.mark.parametrize(
    "manifold",
    [Euclidean(1), Sphere(3)],
)
@pytest.mark.parametrize("curvature", [True, False])
def test_rdog(manifold, curvature):
    """Test RDoG alias."""
    key = random.PRNGKey(123)

    optimiser = rdog(curvature=curvature)
    equiv = scale_by_rdog(curvature=curvature)

    # Check types.
    assert isinstance(optimiser, GradientTransformation)

    # Generate random point on manifold.
    key, subkey = random.split(key)
    point = manifold.random_point(subkey)

    # Test init is the same.
    optimiser_state = optimiser.init(manifold, point)
    equiv_state = equiv.init(manifold, point)

    assert tree_equal(optimiser_state, equiv_state)

    # Test update is the same.
    key, subkey = random.split(key)
    tangent_vector = manifold.log(point, manifold.random_point(subkey))
    optimiser_update, optimiser_state = optimiser.update(
        manifold, tangent_vector, optimiser_state, point
    )
    equiv_update, equiv_state = equiv.update(
        manifold, tangent_vector, equiv_state, point
    )

    assert tree_equal(optimiser_state, equiv_state)
    assert tree_equal(optimiser_update, equiv_update)


@pytest.mark.parametrize(
    "manifold",
    [Euclidean(1), Sphere(3)],
)
@pytest.mark.parametrize("curvature", [True, False])
def test_rdowg(manifold, curvature):
    """Test RDoWG alias."""
    key = random.PRNGKey(123)

    optimiser = rdowg(curvature=curvature)
    equiv = scale_by_rdowg(curvature=curvature)

    # Check types.
    assert isinstance(optimiser, GradientTransformation)

    # Generate random point on manifold.
    key, subkey = random.split(key)
    point = manifold.random_point(subkey)

    # Test init is the same.
    optimiser_state = optimiser.init(manifold, point)
    equiv_state = equiv.init(manifold, point)

    assert tree_equal(optimiser_state, equiv_state)

    # Test update is the same.
    key, subkey = random.split(key)
    tangent_vector = manifold.log(point, manifold.random_point(subkey))
    optimiser_update, optimiser_state = optimiser.update(
        manifold, tangent_vector, optimiser_state, point
    )
    equiv_update, equiv_state = equiv.update(
        manifold, tangent_vector, equiv_state, point
    )

    assert tree_equal(optimiser_state, equiv_state)
    assert tree_equal(optimiser_update, equiv_update)
