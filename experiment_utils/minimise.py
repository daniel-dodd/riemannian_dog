"""Minimise objective functions on manifolds."""
from typing import Tuple

from gpjax.scan import vscan
from jax import random
from jaxtyping import (
    Key,
    PyTree,
)
from optax import GradientTransformation

from experiment_utils.dataset import get_batch
from manifold.geometry import AbstractManifold
from manifold.gradient import value_and_rgrad
from manifold.optimisers.averagers import default_averager


def minimise(
    key: Key,
    manifold: AbstractManifold,
    data,
    objective,
    optimiser: GradientTransformation,
    num_steps,
    batch_size=32,
    log_rate=1,
    metrics: dict = None,
    retraction: bool = False,
) -> Tuple[PyTree, dict]:
    """Optimise a cost function on a manifold using a Riemannian optimiser.

    Args:
        init_point: Initial point on the manifold.
        data: Data for the objective function fit.
        objective: Objective function to minimise.
        optimiser: Riemannian optimiser.
        num_steps: Number of optimisation steps.
        batch_size: Batch size for optimisation.
        key: Random key for optimisation.
        log_rate: Log rate for progress bar.
        metrics: Metrics to evaluate during optimisation.
        retraction: Whether to use retractions or not.

    Returns:
        Optimal point on the manifold and optimisation history (and metrics if specified).
    """

    # Initial point on the manifold.
    init_point = manifold.random_point(key)

    cost = objective(manifold)

    # Initialise the optimiser.
    opt_state = optimiser.init(manifold, init_point)

    # Initialise the weighted average.
    averager = default_averager(opt_state)
    weighted_average, averager_state = averager.init(manifold, init_point)

    # Optimize!
    def step(carry, _=None):
        point, opt_state, weighted_average, averager_state, key = carry

        # Get the batch.
        if batch_size == -1:
            batch = data

        else:
            key, subkey = random.split(key)
            batch = get_batch(data, subkey, batch_size)

        # Evaluate loss and gradients.
        value, riem_grads = value_and_rgrad(manifold, cost)(point, batch)

        # Record the training loss.
        output = dict(
            iterate=dict(train_loss=value),
            average=dict(train_loss=cost(weighted_average, batch)),
        )

        # Evaluate metrics and append to output.
        if metrics is not None:
            for m in metrics:
                output["iterate"][m] = metrics[m](point)
                output["average"][m] = metrics[m](weighted_average)

        # Update the iterate.
        updates, opt_state = optimiser.update(manifold, riem_grads, opt_state, point)
        point = (
            manifold.retraction(point, updates)
            if retraction
            else manifold.exp(point, updates)
        )

        # Update the weighted average.
        weighted_average, averager_state = averager.update(
            manifold,
            params=point,
            optim_state=opt_state,
            averager_state=averager_state,
            weighted_average=weighted_average,
        )

        return (point, opt_state, weighted_average, averager_state, key), output

    (point, opt_state, weighted_average, averager_state, key), output = vscan(
        step,
        (init_point, opt_state, weighted_average, averager_state, key),
        None,
        length=num_steps + 1,  # +1 for initial point!
        log_value=False,
        log_rate=log_rate,
    )

    return dict(iterate=point, average=weighted_average, opt_state=opt_state), output
