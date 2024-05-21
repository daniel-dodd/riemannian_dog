"""Test objectives. """
from jax import (
    random,
    vmap,
)
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Key,
)
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiment_utils.objectives import (
    frechlet,
    pca,
    rayleigh_quotient,
)
from manifold.geometry import (
    AbstractManifold,
    Euclidean,
    Grassmann,
    Sphere,
)


@pytest.mark.parametrize("manifold", [Sphere(2), Euclidean(3)])
@pytest.mark.parametrize("num_data", [1, 2, 5])
def test_frechlet(manifold: AbstractManifold, num_data: int) -> None:
    """Test frechlet objective."""

    # Generate point on the manifold.
    key, subkey = random.split(random.PRNGKey(123))
    point = manifold.random_point(subkey)

    # Generate a batch of points on the manifold, that will be used as the points we compute the frechlet mean of.
    points = vmap(lambda key: manifold.random_point(key))(random.split(key, num_data))

    # Construct the frechlet objective.
    objective = frechlet(manifold)

    # Check that the objective is positive.
    evaluation = objective(point, points)
    assert evaluation > 0.0
    assert isinstance(evaluation, Float[Array, ""])


@pytest.mark.parametrize("num_components", [1, 2, 3, 4, 5])
def test_pca(num_components: int) -> None:
    """Test PCA objective."""

    # Generate data.
    key = random.PRNGKey(123)
    data = random.normal(key, shape=(100, 5))

    # Standardise the data.
    ss = StandardScaler()
    data = ss.fit_transform(data)

    # Use sklearn to perform PCA.
    sklearn_pca = PCA(num_components)
    sklearn_pca.fit(data)
    optimal_solution = sklearn_pca.components_.T
    mean = jnp.asarray(sklearn_pca.mean_).reshape(1, -1)
    centered_data = data - mean
    minimum_loss = (
        jnp.linalg.norm(
            centered_data.T - optimal_solution @ (optimal_solution.T @ centered_data.T)
        )
        ** 2
    ) / centered_data.shape[0]

    # Define the manifold.
    manifold = Grassmann(centered_data.shape[-1], num_components)

    # Construct the pca objective.
    objective = pca(manifold)

    # Check evaluation against sklearn.
    evaluation = objective(optimal_solution, centered_data)
    assert isinstance(evaluation, Float[Array, ""])
    assert np.isclose(evaluation, minimum_loss)

    # Ensure that we get an error if we try to use a non-Grassmann manifold.
    with pytest.raises(ValueError):
        pca(Sphere(3))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_rayleigh_quotient(dimension: int) -> None:
    """Test rayleigh quotient objective."""
    key = random.PRNGKey(123)
    manifold = Sphere(dimension)

    # Generate point on the manifold.
    key, subkey = random.split(key)
    point = manifold.random_point(subkey)

    # Construct the rayleigh quotient objective.
    objective = rayleigh_quotient(manifold)

    # Generate batch of psd matricies.
    def generate_random_psd(key: Key) -> Array:
        """Generate a random positive definite matrix."""
        matrix = random.normal(key, shape=(dimension, dimension))
        return matrix @ matrix.T

    matrix_batch = vmap(generate_random_psd)(random.split(key, 100))

    # Check evaluation.
    evaluation = objective(point, matrix_batch)
    assert isinstance(evaluation, Float[Array, ""])
    assert (
        jnp.abs(evaluation + 0.5 * jnp.dot(jnp.dot(matrix_batch, point), point).mean())
        < 1e-5
    ).all()

    # Ensure that we get an error if we try to use a non-Sphere manifold.
    with pytest.raises(ValueError):
        rayleigh_quotient(Euclidean(3))
