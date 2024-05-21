"""Grassmann manifold. Code inspired and adapted from the (excellent) Pymanopt library."""
from jax import random
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Key,
)

from manifold.geometry.base import AbstractManifold
from manifold.geometry.utils import multiqr

__all__ = [
    "Grassmann",
]


class Grassmann(AbstractManifold):
    """Grassmann manifold."""

    row_dimension: int
    column_dimension: int

    @property
    def curvature_bound(self) -> Float[Array, ""]:
        """Lower bound on the sectional curvature."""
        return jnp.array(0.0)

    def inner_product(
        self,
        point: Float[Array, "M R"],
        tangent_vector_a: Float[Array, "M R"],
        tangent_vector_b: Float[Array, "M R"],
    ) -> Float[Array, ""]:
        """Inner product between two tangent vectors at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector_a: A tangent vector at `point`.
            tangent_vector_b: A tangent vector at `point`.

        Returns:
            The inner product between `tangent_vector_a` and `tangent_vector_b`.
        """
        return jnp.tensordot(tangent_vector_a, tangent_vector_b, axes=2)

    def distance(
        self, point_a: Float[Array, "M R"], point_b: Float[Array, "M R"]
    ) -> Float[Array, ""]:
        """Distance between two points on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.

        Returns:
            The distance between `point_a` and `point_b`.
        """
        s = jnp.clip(jnp.linalg.svd(point_a.T @ point_b, compute_uv=False), a_max=1.0)
        return jnp.linalg.norm(jnp.arccos(s))

    def norm(
        self, point: Float[Array, "M R"], tangent_vector: Float[Array, "M R"]
    ) -> Float[Array, ""]:
        """Norm of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The norm of `tangent_vector`.
        """
        return jnp.linalg.norm(tangent_vector, axis=(-1, -2))

    def exp(
        self, point: Float[Array, "M R"], tangent_vector: Float[Array, "M R"]
    ) -> Float[Array, "M R"]:
        """Exponential map of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The exponential map of `tangent_vector` at `point`.
        """
        U, S, VH = jnp.linalg.svd(tangent_vector, full_matrices=False)
        cos_S = jnp.expand_dims(jnp.cos(S), -2)
        sin_S = jnp.expand_dims(jnp.sin(S), -2)
        Y = point @ (VH.T * cos_S) @ VH + (U * sin_S) @ VH

        # Re-orthonormalize as is done in Pymanopt library.
        q, _ = multiqr(Y)
        return q

    def log(
        self, point_a: Float[Array, "M R"], point_b: Float[Array, "M R"]
    ) -> Float[Array, "M R"]:
        """Logarithmic map of a point on the manifold.

        Args:
            point_b: A point on the manifold.
            point_b: A point on the manifold.

        Returns:
            The logarithmic map of `point_b` at `point_a`.
        """
        YHX = point_b.T @ point_a
        AH = point_b.T - YHX @ point_a.T
        BH = jnp.linalg.solve(YHX, AH)
        U, S, VH = jnp.linalg.svd(BH.T, full_matrices=False)
        arctan_S = jnp.expand_dims(jnp.arctan(S), -2)
        return (U * arctan_S) @ VH

    def retraction(
        self, point: Float[Array, "M R"], tangent_vector: Float[Array, "M R"]
    ) -> Float[Array, "M R"]:
        """Retraction map of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The retraction map of `tangent_vector` at `point`.
        """
        u, _, vt = jnp.linalg.svd(point + tangent_vector, full_matrices=False)
        return u @ vt

    def parallel_transport(
        self,
        point_a: Float[Array, "M R"],
        point_b: Float[Array, "M R"],
        tangent_vector: Float[Array, "M R"],
    ) -> Float[Array, "M R"]:
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
        self, point: Float[Array, "M R"], euclidean_gradient: Float[Array, "M R"]
    ) -> Float[Array, "M R"]:
        """Converts Euclidean gradient to Riemannian gradient.

        Args:
            point: A point on the manifold.
            euclidean_gradient: A Euclidean gradient at `point`.

        Returns:
            The Riemannian gradient of `euclidean_gradient` at `point`.
        """
        return euclidean_gradient - point @ (point.T @ euclidean_gradient)

    def random_point(self, key: Key) -> Float[Array, "M R"]:
        """Generates a random point on the manifold.

        Args:
            key: A JAX PRNG key.

        Returns:
            A random point on the manifold.
        """
        q, _ = multiqr(random.normal(key, (self.row_dimension, self.column_dimension)))
        return q
