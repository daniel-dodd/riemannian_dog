"""Riemannian gradient functions."""

from typing import (
    Any,
    Callable,
    Tuple,
)

from jax import (
    grad,
    value_and_grad,
)
from jaxtyping import (
    Array,
    Float,
    PyTree,
)

from manifold.geometry.base import AbstractManifold

__all__ = [
    "rgrad",
    "value_and_rgrad",
]


def rgrad(
    manifold: AbstractManifold,
    fun: Callable[[PyTree[Float], ...], Float[Array, ""]],
    *args: Any,
    **kwargs: Any,
) -> PyTree[Float]:
    """Computes the Riemannian gradient of a function akin to `jax.grad`.

    Args:
        manifold: A manifold.
        fun: A function.
        *args: Positional arguments to `jax.grad`.
        **kwargs: Keyword arguments to `jax.grad`.

    Returns:
        The Riemannian gradient of `fun` on `manifold`.
    """

    def rgrad_fun(
        point: PyTree[Float], *fun_args: Any, **fun_kwargs: Any
    ) -> PyTree[Float]:
        # Compute the Euclidean gradient of `fun` at `point`.
        euclidean_grad = grad(fun, *args, **kwargs)(point, *fun_args, **fun_kwargs)

        # Convert the Euclidean gradient to the Riemannian gradient.
        riemannian_grad = manifold.euclidean_to_riemannian_gradient(
            point, euclidean_grad
        )

        return riemannian_grad

    return rgrad_fun


def value_and_rgrad(
    manifold: AbstractManifold,
    fun: Callable[[PyTree[Float], ...], Float[Array, ""]],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Float[Array, ""], PyTree[Float]]:
    """Computes the value and Riemannian gradient of a function akin to `jax.value_and_grad`.

    Args:
        manifold: A manifold.
        fun: A function.
        *args: Positional arguments to `jax.grad`.
        **kwargs: Keyword arguments to `jax.grad`.

    Returns:
        The value and Riemannian gradient of `fun` on `manifold`.
    """

    def value_and_rgrad_fun(
        point: PyTree[Float], *fun_args: Any, **fun_kwargs: Any
    ) -> Tuple[Float[Array, ""], PyTree[Float]]:
        # Compute the value and Euclidean gradient of `fun` at `point`.
        value, euclidean_grad = value_and_grad(fun, *args, **kwargs)(
            point, *fun_args, **fun_kwargs
        )

        # Convert the Euclidean gradient to the Riemannian gradient.
        riemannian_grad = manifold.euclidean_to_riemannian_gradient(
            point, euclidean_grad
        )

        return value, riemannian_grad

    return value_and_rgrad_fun
