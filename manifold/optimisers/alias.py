"""Our Riemannian optimisers!"""
from optax import GradientTransformation

from manifold.optimisers.combine import chain
from manifold.optimisers.transformations import (
    scale_by_learning_rate,
    scale_by_nrdog,
    scale_by_radam,
    scale_by_rdog,
    scale_by_rdowg,
)

__all__ = [
    "rsgd",
    "radam",
    "rdog",
    "rdowg",
    "nrdog",
]


def rsgd(learning_rate: float) -> GradientTransformation:
    """Riemannian (stochastic) gradient descent.

    References:
        [1] Bonnabel, S. (2013). Stochastic gradient descent on Riemannian manifolds.
        IEEE Transactions on Automatic Control, 58(9), 2217-2229.

    Args:
        learning_rate: The learning rate.

    Returns:
        A `GradientTransformation` object.
    """
    return scale_by_learning_rate(learning_rate)


def radam(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 1e-8,
    ams_grad: bool = False,
) -> GradientTransformation:
    """Riemannian adaptive moment estimation (RADAM).

    References:
      [Gary Becigneul et al, 2019](https://arxiv.org/abs/1810.00760)

    Args:
        learning_rate: The learning rate.
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the second moment estimates.
        eps: A small constant for numerical stability.
        eps_root: A small constant for numerical stability.
        ams_grad: Whether to use the AMSGrad variant of this algorithm.

    Returns:
        A `GradientTransformation` object.
    """
    return chain(
        scale_by_radam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            ams_grad=ams_grad,
        ),
        scale_by_learning_rate(learning_rate),
    )


def rdog(
    reps: float = 1e-6,
    eps: float = 1e-8,
    curvature: bool = True,
) -> GradientTransformation:
    """RDoG learning-rate-free optimiser.

    Args:
      reps: Small loading term to avoid zero learning rates and divide-by-zero
        errors.
      eps: Small constant for numerical stability.
      curvature: Whether to use the curvature-dependent version of RDoG.

    Returns:
      A `GradientTransformation` object.
    """

    return scale_by_rdog(reps=reps, eps=eps, curvature=curvature)


def rdowg(
    reps: float = 1e-6,
    eps: float = 1e-8,
    curvature: bool = True,
) -> GradientTransformation:
    """RDoWG learning-rate-free optimiser.

    Args:
      reps: Small loading term to avoid zero learning rates and divide-by-zero
        errors.
      eps: Small constant for numerical stability.

    Returns:
      A `GradientTransformation` object.
    """

    return scale_by_rdowg(reps=reps, eps=eps, curvature=curvature)


def nrdog(
    reps: float = 1e-6,
    eps: float = 1e-8,
    curvature: bool = True,
) -> GradientTransformation:
    """NRDoG learning-rate-free optimiser.

    Args:
      reps: Small loading term to avoid zero learning rates and divide-by-zero
        errors.
      eps: Small constant for numerical stability.
      curvature: Whether to use the curvature-dependent version of RDoG.

    Returns:
      A `GradientTransformation` object.
    """

    return scale_by_nrdog(reps=reps, eps=eps, curvature=curvature)
