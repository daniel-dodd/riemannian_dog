"""Poincaré manifold. Implementation thanks to Geoopt and HyperLL (https://github.com/maxvanspengler/hyperbolic_learning_library)"""

from jax import (
    lax,
    random,
)
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Key,
)

from manifold.geometry.base import AbstractManifold

__all__ = [
    "Poincare",
    "conformal_factor",
    "mobius_addition",
    "gyration",
]

EPSILON = 1e-6
MIN_NORM = 1e-15


def project(point: Float[Array, " N"]) -> Float[Array, " N"]:
    """
    Safe projection on the manifold for numerical stability. This was mentioned in [1]

    Args:
        point: point on the manifold.

    Returns:
        Projected vector on the manifold.

    References:
        [1] Hyperbolic Neural Networks, NIPS2018
        https://arxiv.org/abs/1805.09112
    """

    norm = jnp.clip(jnp.linalg.norm(point), MIN_NORM)
    maxnorm = 1 - EPSILON
    projected = point / norm * maxnorm
    return lax.cond(norm > maxnorm, lambda: projected, lambda: point)


def conformal_factor(point: Float[Array, " N"]) -> Float[Array, ""]:
    """Conformal factor.

    Args:
        point: point on the manifold.

    Returns:
        The conformal factor at the given point.
    """
    return 2 / jnp.clip((1 - jnp.sum(point * point)), MIN_NORM)


def recip_conformal_factor(point: Float[Array, " N"]) -> Float[Array, ""]:
    """Reciprocal conformal factor.

    Args:
        point: point on the manifold.

    Returns:
        The reciprocal conformal factor at the given point.
    """
    return (1 - jnp.sum(point * point)) / 2


def mobius_addition(
    point_a: Float[Array, " N"], point_b: Float[Array, " N"]
) -> Float[Array, " N"]:
    """Möbius addition.

    Special non-associative and non-commutative operation which is closed
    in the Poincare ball.

    Args:
        point_a: The first point.
        point_b: The second point.

    Returns:
        The Möbius sum of ``point_a`` and ``point_b``.
    """
    scalar_product = jnp.sum(point_a * point_b)
    norm_point_a = jnp.sum(point_a * point_a)
    norm_point_b = jnp.sum(point_b * point_b)

    return (
        point_a * (1 + 2 * scalar_product + norm_point_b) + point_b * (1 - norm_point_a)
    ) / jnp.clip((1 + 2 * scalar_product + norm_point_a * norm_point_b), MIN_NORM)


def gyration(
    u: Float[Array, " N"], v: Float[Array, " N"], w: Float[Array, " N"]
) -> Float[Array, " N"]:
    """Gyration operator.

    Args:
        u: point on the manifold.
        v: point on the manifold.
        w: point on the manifold.

    Returns:
        The gyration of ``w`` at ``u`` and ``v``.
    """
    u2 = jnp.linalg.norm(u) ** 2
    v2 = jnp.linalg.norm(v) ** 2

    uv = jnp.inner(u, v)
    uw = jnp.inner(u, w)
    vw = jnp.inner(v, w)

    a = -uw * v2 + vw + 2 * uv * vw
    b = -vw * u2 - uw
    d = 1 + 2 * uv + u2 * v2
    return w + 2 * (a * u + b * v) / jnp.clip(d, MIN_NORM)


class Poincare(AbstractManifold):
    """Poincaré manifold."""

    dimension: int

    @property
    def curvature_bound(self) -> Float[Array, ""]:
        """Lower bound on the sectional curvature."""
        return jnp.array(-1.0)

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
        return (conformal_factor(point) ** 2) * jnp.inner(
            tangent_vector_a, tangent_vector_b
        )

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
        return 2 * jnp.arctanh(jnp.linalg.norm(mobius_addition(-point_a, point_b)))

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
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

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
        norm_point = jnp.linalg.norm(tangent_vector)
        return project(
            mobius_addition(
                point,
                tangent_vector
                * (
                    jnp.tanh(norm_point * conformal_factor(point) / 2)
                    / jnp.clip(norm_point, MIN_NORM)
                ),
            )
        )

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
        return project(point + tangent_vector)

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
        w = mobius_addition(-point_a, point_b)
        norm_w = jnp.linalg.norm(w)
        return (
            jnp.arctanh(norm_w)
            * (w / jnp.clip(norm_w, MIN_NORM))
            * (2 / conformal_factor(point_a))
        )

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
        return gyration(point_b, -point_a, tangent_vector) * (
            conformal_factor(point_a) * recip_conformal_factor(point_b)
        )

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
        return euclidean_gradient * (recip_conformal_factor(point) ** 2)

    def random_point(self, key: Key) -> Float[Array, " N"]:
        """Generates a random point on the manifold.

        Args:
            key: A JAX PRNG key.

        Returns:
            A random point on the manifold.
        """
        key, subkey = random.split(key)
        array = random.normal(subkey, (self.dimension,))
        norm = jnp.linalg.norm(array)
        radius = random.uniform(key) ** (1.0 / self.dimension)
        point = array / norm * radius
        return point
