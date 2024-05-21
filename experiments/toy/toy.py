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
    rdog,
    rsgd,
)

# Settings.
SEED = 123
DIM = 3
NUM_STEPS = 100
NREPS = 1
BATCH_SIZE = -1

key = random.PRNGKey(SEED)

# Create dataset.
key, subkey = random.split(key)
sqrt = jnp.pi * random.normal(subkey, shape=(DIM, DIM + 1))
matrix = sqrt @ sqrt.T / jnp.sqrt(DIM)

# Compute the first eigenvector numerically.
eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
max_eig = jnp.max(eigenvalues)
min_eig = jnp.min(eigenvalues)
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
    trajectory=lambda point: point,
)


# Define optimisers to run.
experiment_tree = {
    "rdog": rdog(),
    "big_rsgd": rsgd(1e-1),
    "small_rsgd": rsgd(1e-3),
}

optimisers, experiment_treedef = jtu.tree_flatten(
    experiment_tree, is_leaf=lambda x: isinstance(x, tuple)
)


if __name__ == "__main__":
    # Run the experiment.
    replication_leaves = []
    for opt in optimisers:
        _, output = minimise(
            key,
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

    experiment_data = experiment_treedef.unflatten(replication_leaves)

    experiment_data["sol"] = sol

    # Save the experiment data.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = os.path.join(dir_path, "results")

    # Create the directory
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # Pickle the data in the directory.
    with open(dir_name + "/toy.pkl", "wb") as f:
        pickle.dump(experiment_data, f)
