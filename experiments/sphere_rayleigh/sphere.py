import os
import pickle

from jax import (
    config,
    random,
)
import jax.numpy as jnp
import jax.tree_util as jtu

config.update("jax_enable_x64", True)

from experiment_utils.minimise import minimise
from experiment_utils.objectives import rayleigh_quotient
from manifold.geometry import Sphere
from manifold.optimisers import (
    nrdog,
    radam,
    rdog,
    rdowg,
    rsgd,
)

# Settings.
SEED = 42
DIM = 1000
NUM_STEPS = 5000
NREPS = 10
BATCH_SIZE = -1
LEARNING_RATES = jnp.logspace(-8, 6, num=20)
SENSITIVITY = jnp.logspace(-8, 6, num=20)

key = random.PRNGKey(SEED)

# Create dataset.
key, subkey = random.split(key)
sqrt = random.normal(subkey, shape=(DIM, DIM + 100)) / jnp.sqrt(DIM)
matrix = sqrt @ sqrt.T

# Compute the first eigenvector numerically.
eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
eigenvalues = jnp.real(eigenvalues)
eigenvectors = jnp.real(eigenvectors)
sol = eigenvectors[:, jnp.argmax(eigenvalues)] / jnp.linalg.norm(
    eigenvectors[:, jnp.argmax(eigenvalues)]
)

# Define the geometry and cost function.
geom = Sphere(DIM)

# Metrics to track. `distance` from a minimiser (this example has two) and `regret` from the minimal value.
metrics = dict(
    distance=lambda point: jnp.minimum(
        geom.distance(point, sol), geom.distance(point, -sol)
    ),
    regret=lambda point: rayleigh_quotient(geom)(point, matrix)
    - rayleigh_quotient(geom)(sol, matrix),
)


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
                matrix,
                rayleigh_quotient,
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
    with open(dir_name + "/sphere.pkl", "wb") as f:
        pickle.dump(experiment_data, f)
