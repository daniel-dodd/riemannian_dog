"""Utility functions for working with arrays of matrices. These have been adapted from the (excellent) Pymanopt library."""

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

__all__ = [
    "multitransp",
    "multihconj",
    "multiqr",
]


def multitransp(A: Float[Array, "M N P"]) -> Float[Array, "M P N"]:
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.

    Args:
        A: An array of matrices.

    Returns:
        An array of transposed matrices.
    """
    if A.ndim == 2:
        return A.T
    return jnp.transpose(A, (0, 2, 1))


def multihconj(A: Float[Array, "M N P"]) -> Float[Array, "M N P"]:
    """Vectorized matrix conjugate transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.

    That is, ``A`` is an ``M x N x P`` array. Multihconj then returns an array
    containing the ``M`` conjugate transpose of the matrices in ``A``, each of
    which will be ``P x N``.

    Args:
        A: An array of matrices.

    Returns:
        An array of conjugate transposed matrices.
    """
    return jnp.conjugate(multitransp(A))


def multiqr(A: Float[Array, "M N N"]) -> Float[Array, "M N N"]:
    """Vectorized QR decomposition.

    Args:
        A: An array of matrices.

    Returns:
        An array of Q matrices and an array of R matrices.
    """
    if A.ndim not in (2, 3):
        raise ValueError("Input must be a matrix or a stacked matrix")

    q, r = jnp.vectorize(jnp.linalg.qr, signature="(m,n)->(m,k),(k,n)")(A)

    # Compute signs or unit-modulus phase of entries of diagonal of r.
    s = jnp.diagonal(r, axis1=-2, axis2=-1).copy()
    s = jnp.where(s == 0.0, 1.0, s)
    s = s / jnp.abs(s)

    s = jnp.expand_dims(s, axis=-1)
    q = q * multitransp(s)
    r = r * jnp.conjugate(s)
    return q, r
