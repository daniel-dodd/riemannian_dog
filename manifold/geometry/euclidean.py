"""Eucidean manifold."""
from jax import random
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Key,
)

from manifold.geometry.base import AbstractManifold

__all__ = [
    "Euclidean",
]


class Euclidean(AbstractManifold):
    """Euclidean manifold."""

    dimension: int

    @property
    def curvature_bound(self) -> Float[Array, ""]:
        """Lower bound on the sectional curvature."""
        return jnp.array(0.0)

    def inner_product(
        self,
        point: Float[Array, " N"],
        tangent_vector_a: Float[Array, " N"],
        tangent_vector_b: Float[Array, " N"],
    ) -> Float[Array, ""]:
        """Inner product between two tangent vectors at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector_a: A tangent vector at `point`.
            tangent_vector_b: A tangent vector at `point`.

        Returns:
            The inner product between `tangent_vector_a` and `tangent_vector_b`.
        """
        return jnp.dot(tangent_vector_a, tangent_vector_b)

    def distance(
        self, point_a: Float[Array, " N"], point_b: Float[Array, " N"]
    ) -> Float[Array, ""]:
        """Distance between two points on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.

        Returns:
            The distance between `point_a` and `point_b`.
        """
        return jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(point_a - point_b)), 1e-32))

    def norm(
        self, point: Float[Array, " N"], tangent_vector: Float[Array, " N"]
    ) -> Float[Array, ""]:
        """Norm of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The norm of `tangent_vector`.
        """
        return jnp.linalg.norm(tangent_vector)

    def exp(
        self, point: Float[Array, " N"], tangent_vector: Float[Array, " N"]
    ) -> Float[Array, " N"]:
        """Exponential map of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The exponential map of `tangent_vector` at `point`.
        """
        return point + tangent_vector

    def log(
        self, point_a: Float[Array, " N"], point_b: Float[Array, " N"]
    ) -> Float[Array, " N"]:
        """Logarithmic map of a point on the manifold.

        Args:
            point_b: A point on the manifold.
            point_b: A point on the manifold.

        Returns:
            The logarithmic map of `point_b` at `point_a`.
        """
        return point_b - point_a

    def parallel_transport(
        self,
        point_a: Float[Array, " N"],
        point_b: Float[Array, " N"],
        tangent_vector: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        """Parallel transport of a tangent vector from a point to another on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.
            tangent_vector: A tangent vector at `point_a`.

        Returns:
            The parallel transport of `tangent_vector` from `point_a` to `point_b`.
        """
        return tangent_vector

    def euclidean_to_riemannian_gradient(
        self, point: Float[Array, " N"], euclidean_gradient: Float[Array, " N"]
    ) -> Float[Array, " N"]:
        """Converts Euclidean gradient to Riemannian gradient.

        Args:
            point: A point on the manifold.
            euclidean_gradient: A Euclidean gradient at `point`.

        Returns:
            The Riemannian gradient of `euclidean_gradient` at `point`.
        """
        return euclidean_gradient

    def random_point(self, key: Key) -> Float[Array, " N"]:
        """Generates a random point on the manifold.

        Args:
            key: A JAX PRNG key.

        Returns:
            A random point on the manifold.
        """
        return random.normal(key, (self.dimension,))
