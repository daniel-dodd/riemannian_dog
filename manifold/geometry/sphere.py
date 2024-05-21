"""Sphere manifold. Adapted from Pymanopt."""
from jax import random
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Key,
)

from manifold.geometry.base import AbstractManifold

__all__ = [
    "Sphere",
]

# Prevents division by zero.
EPSILON = 1e-15


class Sphere(AbstractManifold):
    """Sphere manifold.

    NOTE: The dimension is of the ambient space.
    """

    dimension: int

    @property
    def curvature_bound(self) -> Float[Array, ""]:
        """Lower bound on the sectional curvature."""
        return jnp.array(1.0)

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
        return jnp.arccos(
            jnp.clip(
                jnp.inner(point_a, point_b),
                -1.0,
                1.0,
            )
        )

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
        norm = self.norm(point, tangent_vector)
        exp = point * jnp.cos(norm) + tangent_vector * jnp.sin(norm) / norm
        return exp / jnp.linalg.norm(exp)

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
        vector = self.euclidean_to_riemannian_gradient(point_a, point_b - point_a)
        distance = self.distance(point_a, point_b)
        return (distance + EPSILON) / (self.norm(point_a, vector) + EPSILON) * vector

    def retraction(
        self, point: Float[Array, " N"], tangent_vector: Float[Array, " N"]
    ) -> Float[Array, " N"]:
        """Retraction map of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The retraction map of `tangent_vector` at `point`.
        """
        array = point + tangent_vector

        # We normalize the array to project it onto the sphere.
        return array / jnp.linalg.norm(array)

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
        return self.euclidean_to_riemannian_gradient(point_b, tangent_vector)

    def euclidean_to_riemannian_gradient(
        self, point: Float[Array, " N"], euclidean_gradient: Float[Array, " N"]
    ) -> Float[Array, " N"]:
        """Converts Euclidean gradient - to Riemannian gradient.

        Args:
            point: A point on the manifold.
            euclidean_gradient: A Euclidean gradient at `point`.

        Returns:
            The Riemannian gradient of `euclidean_gradient` at `point`.
        """
        return (
            euclidean_gradient
            - self.inner_product(point, point, euclidean_gradient) * point
        )

    def random_point(self, key: Key) -> Float[Array, " N"]:
        """Generates a random point on the manifold.

        Args:
            key: A JAX PRNG key.

        Returns:
            A random point on the manifold.
        """
        # We generate a random point.
        x = random.normal(key, shape=(self.dimension,))

        # And project it onto the sphere by normalizing!
        return x / jnp.linalg.norm(x)
