"""Test combine transformations for Riemannian optimisers."""
from typing import Tuple

from equinox import tree_equal
from jax import random
from optax import GradientTransformation
import pytest

from manifold.geometry import (
    AbstractManifold,
    Euclidean,
    Sphere,
)
from manifold.optimisers.combine import chain
from manifold.optimisers.transformations import (
    scale_by_learning_rate,
    scale_by_radam,
)


@pytest.mark.parametrize("manifold", [Euclidean(1), Sphere(3)])
@pytest.mark.parametrize(
    "transforms",
    [
        (
            scale_by_radam(),
            scale_by_learning_rate(1e-3),
        ),
        (scale_by_learning_rate(1e-3),),
    ],
)
def test_chain(manifold: AbstractManifold, transforms: Tuple[GradientTransformation]):
    key = random.PRNGKey(123)

    chain_transform = chain(*transforms)

    # Check types.
    assert isinstance(chain_transform, GradientTransformation)

    # Generate random point on manifold.
    key, subkey = random.split(key)
    point = manifold.random_point(subkey)

    init_fns, update_fns = zip(*transforms)

    # Test init is the same.
    chain_state = chain_transform.init(manifold, point)
    equiv_state = tuple(init(manifold, point) for init in init_fns)

    assert tree_equal(chain_state, equiv_state)

    # Test update is the same.
    key, subkey = random.split(key)
    tangent_vector = manifold.log(point, manifold.random_point(subkey))
    chain_update, chain_state = chain_transform.update(
        manifold, tangent_vector, chain_state, point
    )

    new_state = []
    equiv_update = tangent_vector
    for s, fn in zip(equiv_state, update_fns):
        equiv_update, new_s = fn(manifold, equiv_update, s, point)
        new_state.append(new_s)
    equiv_state = tuple(new_state)

    assert tree_equal(chain_update, equiv_update)
    assert tree_equal(chain_state, equiv_state)
