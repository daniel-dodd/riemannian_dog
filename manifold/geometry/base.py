"""Abstract class for a manifold."""
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

from jaxtyping import (
    Array,
    Float,
    Key,
    PyTree,
)

__all__ = [
    "AbstractManifold",
]


class AbstractManifold(ABC):
    """Abstract class for a manifold."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(cls, eq=True, frozen=True)

    @property
    def curvature_bound(self) -> Float:
        """Lower bound on the sectional curvature."""
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `curvature_bound`."
        )

    @abstractmethod
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
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `inner_product`."
        )

    @abstractmethod
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
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `distance`."
        )

    @abstractmethod
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
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `norm`."
        )

    @abstractmethod
    def exp(self, point: PyTree[Float], tangent_vector: PyTree[Float]) -> PyTree[Float]:
        """Exponential map of a tangent vector at a point on the manifold.

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at `point`.

        Returns:
            The exponential map of `tangent_vector` at `point`.
        """
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `exp`."
        )

    @abstractmethod
    def log(self, point_a: PyTree[Float], point_b: PyTree[Float]) -> PyTree[Float]:
        """Logarithmic map of a point on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.

        Returns:
            The logarithmic map of `tangent_vector` at `point_a`.
        """
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `log`."
        )

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
        return self.exp(point, tangent_vector)

    @abstractmethod
    def parallel_transport(
        self,
        point_a: PyTree[Float],
        point_b: PyTree[Float],
        tangent_vector: PyTree[Float],
    ) -> PyTree[Float]:
        """Parallel transport of a tangent vector from a point to another on the manifold.

        Args:
            point_a: A point on the manifold.
            point_b: A point on the manifold.
            tangent_vector: A tangent vector at `point_a`.

        Returns:
            The parallel transport of `tangent_vector` from `point_a` to `point_b`.
        """
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `parallel_transport`."
        )

    @abstractmethod
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
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `euclidean_to_riemannian_gradient`."
        )

    @abstractmethod
    def random_point(self, key: Key) -> PyTree[Float]:
        """Generates a random point on the manifold.

        Args:
            key: A JAX PRNG key.

        Returns:
            A random point on the manifold.
        """
        raise NotImplementedError(
            f"Class {self.__class__.__name__} does not implement `random_point`."
        )
