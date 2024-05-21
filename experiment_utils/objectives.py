"""Common objectives."""
from typing import Callable

from jax import (
    nn,
    vmap,
)
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    PyTree,
)

from manifold.geometry import (
    AbstractManifold,
    Grassmann,
    Sphere,
)

__all__ = [
    "frechlet",
    "pca",
    "rayleigh_quotient",
    "word_embeddings",
]


def frechlet(
    manifold: AbstractManifold,
) -> Callable[[PyTree[Float[Array, "*"]], PyTree[Float[Array, "*"]]], Float[Array, ""]]:
    """Frechlet mean objective.

    Args:
        manifold: Manifold to use.

    Returns:
        Frechlet mean objective.
    """

    def frechlet_fn(
        point: PyTree[Float[Array, "*"]], data_points: PyTree[Float[Array, "*"]]
    ) -> Float[Array, ""]:
        """Frechlet mean objective.

        Args:
            point: Mean of the distribution.
            data_points: Data points to compare to the mean.

        Returns:
            Frechlet mean objective.
        """
        return vmap(lambda data_point: manifold.distance(point, data_point))(
            data_points
        ).mean()

    return frechlet_fn


def pca(
    manifold: Grassmann,
) -> Callable[[PyTree[Float[Array, "*"]], PyTree[Float[Array, "*"]]], Float[Array, ""]]:
    """PCA objective.

    Args:
        manifold: Manifold to use.

    Returns:
        PCA objective.
    """

    if not isinstance(manifold, Grassmann):
        raise ValueError("Manifold must be a Grassmann manifold.")

    def pca_fn(
        components: PyTree[Float[Array, "*"]], data_points: PyTree[Float[Array, "*"]]
    ) -> Float[Array, ""]:
        """PCA objective.

        Args:
            components: Components of the PCA.
            data_points: Data points to compare to the components.

        Returns:
            PCA objective.
        """
        return vmap(
            lambda data_point: jnp.linalg.norm(
                data_point - data_point @ components @ components.T
            )
            ** 2
        )(data_points).mean()

    return pca_fn


def rayleigh_quotient(
    manifold: Sphere,
) -> Callable[[PyTree[Float[Array, "*"]], PyTree[Float[Array, "*"]]], Float[Array, ""]]:
    """Rayleigh quotient objective.

    Args:
        manifold: Manifold to use.

    Returns:
        Rayleigh quotient objective.
    """

    if not isinstance(manifold, Sphere):
        raise ValueError("Manifold must be a Sphere manifold.")

    del manifold

    def rayleigh_quotient_fn(
        point: Float[Array, " N"], matrix_batch: Float[Array, "B N N"]
    ) -> Float[Array, ""]:
        """Rayleigh quotient objective. Takes in a batch of samples from a random matrix to learn the expected value of the Rayleigh quotient.

        Args:
            point: Point to evaluate the Rayleigh quotient at.
            matrix_batch: Batch of matricies to evaluate the Rayleigh quotient objective at.

        Returns:
            Rayleigh quotient objective.
        """
        return -0.5 * jnp.dot(jnp.dot(matrix_batch, point), point).mean()

    return rayleigh_quotient_fn


def word_embeddings(
    embedding_manifold: AbstractManifold,
) -> Callable[[Float[Array, "B N D"]], Float[Array, ""]]:
    r"""Word embeddings loss function of Nickel & Kiela (2017).

    Args:
        embedding_manifold: Manifold to use.

    Returns:
        Word embeddings loss function.
    """

    def loss_fn(embeddings_batch: Float[Array, "B N D"]) -> Float[Array, ""]:
        def compute_loss(embeddings: Float[Array, "N D"]) -> Float[Array, ""]:
            # Compute distances between the first embedding and all others for each batch.
            distances = vmap(embedding_manifold.distance, in_axes=(None, 0))(
                embeddings[0], embeddings[1:]
            )

            # Compute negative log of the softmax of the distances.
            return -jnp.log(nn.softmax(-distances)[0])

        return vmap(compute_loss)(embeddings_batch).mean()

    return loss_fn
