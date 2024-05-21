import os
import pickle

from jax import (
    config,
    random,
)
import jax.numpy as jnp
import jax.tree_util as jtu

config.update("jax_enable_x64", True)

from experiment_data.pca import wine
from experiment_utils.minimise import minimise
from experiment_utils.objectives import pca
from manifold.geometry import Grassmann
from manifold.optimisers import (
    nrdog,
    radam,
    rdog,
    rdowg,
    rsgd,
)

# Settings.
SEED = 42
NUM_COMPONENTS = 1
NUM_STEPS = 5000
NREPS = 5
BATCH_SIZE = -1
LEARNING_RATES = jnp.logspace(-8, 6, num=20)
SENSITIVITY = jnp.logspace(-8, 6, num=20)

key = random.PRNGKey(SEED)

# Load the data.
data = wine(NUM_COMPONENTS)

minimiser = data.minimiser
num_rows = data.X.shape[-1]
num_columns = data.num_components

# Define the geometry and cost function.
geom = Grassmann(num_rows, num_columns)


# Distance to the minimiser.
metrics = dict(
    distance=lambda point: geom.distance(point, minimiser),
    regret=lambda point: pca(geom)(point, data.X) - pca(geom)(minimiser, data.X),
)


# Define the optimisers.
def scale_by_root(learning_rate: float):
    """Returns a schedule that scales the learning rate by 1/sqrt(t + 1)."""

    def schedule(t: int):
        return learning_rate / jnp.sqrt(t + 1)

    return schedule


# Define optimisers to run.
experiment_tree = {
    "parameter_free": {"rdog": rdog(), "rdowg": rdowg(), "nrdog": nrdog()},
    "learning_rate": {
        "rsgd": [rsgd(lr) for lr in LEARNING_RATES],
        "radam": [radam(lr) for lr in LEARNING_RATES],
    },
    "scaled_learning_rate": {
        "rsgd": [rsgd(scale_by_root(lr)) for lr in LEARNING_RATES],
        "radam": [radam(scale_by_root(lr)) for lr in LEARNING_RATES],
    },
    "sensitivity": {
        "rdog": [rdog(s) for s in SENSITIVITY],
        "rdowg": [rdowg(s) for s in SENSITIVITY],
        "nrdog": [nrdog(s) for s in SENSITIVITY],
    },
}

optimisers, experiment_treedef = jtu.tree_flatten(
    experiment_tree, is_leaf=lambda x: isinstance(x, tuple)
)


# Run the experiment.
if __name__ == "__main__":
    experiment_data = []

    for _ in range(NREPS):
        key, subkey = random.split(key)
        replication_leaves = []
        for opt in optimisers:
            _, output = minimise(
                subkey,
                geom,
                data.X,
                pca,
                opt,
                NUM_STEPS,
                BATCH_SIZE,
                log_rate=1,
                metrics=metrics,
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
    with open(dir_name + "/wine.pkl", "wb") as f:
        pickle.dump(experiment_data, f)
