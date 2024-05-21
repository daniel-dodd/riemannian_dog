"""Mammals subtree. """
import os
import pickle
from typing import (
    Callable,
    NamedTuple,
)

from jax import (
    config,
    random,
)
import jax.numpy as jnp
import jax.tree_util as jtu

config.update("jax_enable_x64", True)

from experiment_data.wordnet import mammal_relations
from experiment_utils.wordnet import minimise
from manifold.geometry import Poincare
from manifold.optimisers import (
    nrdog,
    radam,
    rdog,
    rdowg,
    rsgd,
)

# Settings.
NREPS = 5
NUM_DIMENSIONS = 5
NUM_EPOCH = 1000
NUM_BURNIN = 10
NUM_NEGATIVES = 50
BATCH_SIZE = 10
SEED = 42

# Define key.
key = random.PRNGKey(SEED)

# Load data.
data = mammal_relations()

# Define experiments.
LEARNING_RATES = jnp.logspace(-2, 2, num=10)
SENSITIVITY = jnp.logspace(-10, -6, num=5)


class Experiment(NamedTuple):
    main_optim: Callable
    burnin_optim: Callable = None
    burnin_epochs: int = 0


experiment_tree = dict(
    sensitivity=dict(
        rdog=[Experiment(main_optim=rdog(s)) for s in SENSITIVITY],
        rdowg=[Experiment(main_optim=rdowg(s)) for s in SENSITIVITY],
        nrdog=[Experiment(main_optim=nrdog(s)) for s in SENSITIVITY],
    ),
    sensitivity_noc=dict(
        rdog=[Experiment(main_optim=rdog(s, curvature=False)) for s in SENSITIVITY],
        rdowg=[Experiment(main_optim=rdowg(s, curvature=False)) for s in SENSITIVITY],
        nrdog=[Experiment(main_optim=nrdog(s, curvature=False)) for s in SENSITIVITY],
    ),
    learning_rate=dict(
        rsgd=[
            Experiment(
                main_optim=rsgd(lr),
                burnin_optim=rsgd(lr / 10),
                burnin_epochs=NUM_BURNIN,
            )
            for lr in LEARNING_RATES
        ],
        radam=[
            Experiment(
                main_optim=radam(lr),
                burnin_optim=rsgd(lr / 10),
                burnin_epochs=NUM_BURNIN,
            )
            for lr in LEARNING_RATES
        ],
    ),
)

experiments, experiment_treedef = jtu.tree_flatten(
    experiment_tree, is_leaf=lambda x: isinstance(x, Experiment)
)

if __name__ == "__main__":
    experiment_data = []

    for _ in range(NREPS):
        replication_leaves = []

        key, subkey = random.split(key)

        for experiment in experiments:
            output = minimise(
                subkey,
                relations=data,
                embedding_manifold=Poincare(NUM_DIMENSIONS),
                main_optim=experiment.main_optim,
                main_epochs=NUM_EPOCH,
                burnin_optim=experiment.burnin_optim,
                burnin_epochs=experiment.burnin_epochs,
                num_negatives=NUM_NEGATIVES,
                batch_size=BATCH_SIZE,
            )

            replication_leaves.append(output)

        experiment_data.append(experiment_treedef.unflatten(replication_leaves))

    # Save the experiment data.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = os.path.join(dir_path, "results")

    # Create the directory
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # Pickle the data in the directory.
    with open(dir_name + "/mammals.pkl", "wb") as f:
        pickle.dump(experiment_data, f)
