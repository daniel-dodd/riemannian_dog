"""Product manifold when the underlying manifolds in the cartesian product are the same."""
from jax import (
    random,
    vmap,
)
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    Key,
    PyTree,
)

from manifold.geometry import AbstractManifold

__all__ = [
    "ProductSame",
]


class ProductSame(AbstractManifold):
    """Cartesian product manifold same underlying manifold.

    Points on the manifold and tangent vectors are represented as lists of
    points and tangent vectors of the individual manifolds.
    The metric is obtained by element-wise extension of the individual
    manifolds.

    Args:
        manifolds: The collection of manifolds in the product.
    """

    manifold: AbstractManifold
    num_manifolds: int

    def __post_init__(self):
        if not isinstance(self.manifold, AbstractManifold):
            raise TypeError(
                f"Expected `manifold` to be of type `AbstractManifold`, got "
                f"`{type(self.manifold)}` instead."
            )

    @property
    def curvature_bound(self) -> Float[Array, ""]:
        """Lower bound on the sectional curvature."""
        return self.manifold.curvature_bound

    def norm(
        self, point: PyTree[Float], tangent_vector: PyTree[Float]
    ) -> Float[Array, ""]:
        """Norm of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The norm of `tangent_vector`.
        """
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def inner_product(
        self,
        point: PyTree[Float],
        tangent_vector_a: PyTree[Float],
        tangent_vector_b: PyTree[Float],
    ) -> Float[Array, ""]:
        """Inner product between two tangent vectors at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector_a: A tangent vector at `point`.
            tangent_vector_b: A tangent vector at `point`.

        Returns:
            The inner product between `tangent_vector_a` and `tangent_vector_b`.
        """
        return jnp.sum(
            vmap(self.manifold.inner_product, in_axes=(0, 0, 0))(
                point, tangent_vector_a, tangent_vector_b
            )
        )

    def distance(
        self, point_a: PyTree[Float], point_b: PyTree[Float]
    ) -> Float[Array, ""]:
        """Distance between two points on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.

        Returns:
            The distance between `point_a` and `point_b`.
        """
        return jnp.sqrt(
            jnp.sum(
                jnp.square(
                    vmap(self.manifold.distance, in_axes=(0, 0))(point_a, point_b)
                )
            )
        )

    def euclidean_to_riemannian_gradient(
        self, point: PyTree[Float], euclidean_gradient: PyTree[Float]
    ) -> PyTree[Float]:
        """Converts Euclidean gradient to Riemannian gradient.

        Args:
            point: A point on the manifold.
            euclidean_gradient: A Euclidean gradient at `point`.

        Returns:
            The Riemannian gradient of `euclidean_gradient` at `point`.
        """
        return vmap(self.manifold.euclidean_to_riemannian_gradient, in_axes=(0, 0))(
            point, euclidean_gradient
        )

    def exp(self, point: PyTree[Float], tangent_vector: PyTree[Float]) -> PyTree[Float]:
        """Exponential map of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The exponential map of `tangent_vector` at `point`.
        """
        return vmap(self.manifold.exp, in_axes=(0, 0))(point, tangent_vector)

    def retraction(
        self, point: PyTree[Float], tangent_vector: PyTree[Float]
    ) -> PyTree[Float]:
        """Retraction map of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The retraction map of `tangent_vector` at `point`.
        """
        return vmap(self.manifold.retraction, in_axes=(0, 0))(point, tangent_vector)

    def log(self, point_a: PyTree[Float], point_b: PyTree[Float]) -> PyTree[Float]:
        """Logarithmic map of a point on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.

        Returns:
            The logarithmic map of `tangent_vector` at `point_a`.
        """
        return vmap(self.manifold.log, in_axes=(0, 0))(point_a, point_b)

    def random_point(self, key: Key) -> PyTree[Float]:
        """Sample a random point on the manifold.

        Args:
            key: A PRNG key.

        Returns:
            A random point on the manifold.
        """
        keys = random.split(key, self.num_manifolds)
        return vmap(self.manifold.random_point)(keys)

    def parallel_transport(
        self,
        point_a: PyTree[Float],
        point_b: PyTree[Float],
        tangent_vector_a: PyTree[Float],
    ) -> PyTree[Float]:
        """Parallel transport of a tangent vector from a point to another on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.
            tangent_vector_a: A tangent vector at `point_a`.

        Returns:
            The parallel transport of `tangent_vector` from `point_a` to `point_b`.
        """
        return vmap(self.manifold.parallel_transport, in_axes=(0, 0, 0))(
            point_a, point_b, tangent_vector_a
        )
