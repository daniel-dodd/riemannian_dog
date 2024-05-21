import os
import pickle
from typing import Iterable

from jax import tree_util as jtu
import jax.numpy as jnp
from jaxtyping import PyTree
from matplotlib import ticker
import matplotlib.pyplot as plt

# Set the style and font.
from experiment_utils.plots import (
    get_cols,
    set_style_and_font,
)

set_style_and_font()
cols = get_cols()

# Settings.
WHAT = "iterate"
LEARNING_RATES = jnp.logspace(-2, 2, num=10)
SENSITIVITY = jnp.logspace(-10, -6, num=5)

OPTIMS_TO_PLOT = {
    "rdog": dict(color=cols[0], label="RDoG (Ours)"),
    "rdowg": dict(color=cols[1], label="RDoWG (Ours)"),
    "rsgd": dict(color=cols[2], label="RSGD"),
    "radam": dict(color=cols[3], label="RADAM"),
    "nrdog": dict(color=cols[4], label="NRDoG (Ours)"),
}

# Load experiment data.
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "results/mammals.pkl"), "rb") as f:
    experiment_data_all = pickle.load(f)

# with open(os.path.join(dir_path, "results/mammals_productwise.pkl"), "rb") as f:
#     experiment_data_pw = pickle.load(f)


# Create plots directory.
dir_name = os.path.join(dir_path, "plots")

# Create the directory
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


# Compute mean and standard error.
def _standard_error(*replication_leaves: Iterable[PyTree]):
    leaves_as_array = jnp.array([*replication_leaves])
    return jnp.std(leaves_as_array, axis=0) / jnp.sqrt(leaves_as_array.shape[0])


def _mean(*replication_leaves: Iterable[PyTree]):
    return jnp.array([*replication_leaves]).mean(axis=0)


mean = jtu.tree_map(_mean, *experiment_data_all)
standard_error = jtu.tree_map(_standard_error, *experiment_data_all)

# mean_pw = jtu.tree_map(_mean, *experiment_data_pw)
# standard_error_pw = jtu.tree_map(_standard_error, *experiment_data_pw)

# Plot the results at the final iteration for each metric.
METRICS = {
    "map": dict(label=r"Mean average precision", yscale="linear"),
}


for WHICH in ["sensitivity_noc", "sensitivity"]:
    for m in METRICS:
        fig, ax = plt.subplots()
        ax_top = ax.twiny()

        ax.set_xscale("log")
        ax_top.set_xscale("log")
        ax.set_xlabel(r"Learning rate $\eta$")
        ax_top.set_xlabel(r"Initial distance $\epsilon$")
        ax.set_ylim(0.0, 1.0)

        for opt in mean[WHICH]:
            # if opt == "nrdog":
            #     continue

            ax_top.plot(
                SENSITIVITY,
                [mean[WHICH][opt][_][WHAT][m] for _ in range(len(SENSITIVITY))],
                ".-",
                label=OPTIMS_TO_PLOT[opt]["label"],
                color=OPTIMS_TO_PLOT[opt]["color"],
            )
            ax_top.fill_between(
                SENSITIVITY,
                [
                    mean[WHICH][opt][_][WHAT][m]
                    - standard_error[WHICH][opt][_][WHAT][m]
                    for _ in range(len(SENSITIVITY))
                ],
                [
                    mean[WHICH][opt][_][WHAT][m]
                    + standard_error[WHICH][opt][_][WHAT][m]
                    for _ in range(len(SENSITIVITY))
                ],
                alpha=0.15,
                color=OPTIMS_TO_PLOT[opt]["color"],
            )

        for opt in mean["learning_rate"]:
            if opt == "radam":
                # ax.plot(
                #     LEARNING_RATES,
                #     [mean_pw["learning_rate"][opt][_]["iterate"][m] for _ in range(len(LEARNING_RATES))],
                #     ".-",
                #     label=OPTIMS_TO_PLOT[opt]["label"],
                #     color=OPTIMS_TO_PLOT[opt]["color"],
                # )

                # ax.fill_between(
                #     LEARNING_RATES,
                #     [mean_pw["learning_rate"][opt][_]["iterate"][m]
                #     - standard_error["learning_rate"][opt][_][WHAT][m] for _ in range(len(LEARNING_RATES))],
                #     [mean_pw["learning_rate"][opt][_]["iterate"][m]
                #     + standard_error_pw["learning_rate"][opt][_]["iterate"][m] for _ in range(len(LEARNING_RATES))],
                #     alpha=0.15,
                #     color=OPTIMS_TO_PLOT[opt]["color"],
                # )
                continue

            ax.plot(
                LEARNING_RATES,
                [
                    mean["learning_rate"][opt][_]["iterate"][m]
                    for _ in range(len(LEARNING_RATES))
                ],
                ".-",
                label=OPTIMS_TO_PLOT[opt]["label"],
                color=OPTIMS_TO_PLOT[opt]["color"],
            )

            ax.fill_between(
                LEARNING_RATES,
                [
                    mean["learning_rate"][opt][_]["iterate"][m]
                    - standard_error["learning_rate"][opt][_][WHAT][m]
                    for _ in range(len(LEARNING_RATES))
                ],
                [
                    mean["learning_rate"][opt][_]["iterate"][m]
                    + standard_error["learning_rate"][opt][_]["iterate"][m]
                    for _ in range(len(LEARNING_RATES))
                ],
                alpha=0.15,
                color=OPTIMS_TO_PLOT[opt]["color"],
            )

        ax.set_ylabel(METRICS[m]["label"])
        ax.set_yscale(METRICS[m]["yscale"])

        # Get the legend handles and labels from both axes
        handles, labels = ax.get_legend_handles_labels()
        handles_top, labels_top = ax_top.get_legend_handles_labels()

        # Combine the handles and labels
        handles += handles_top
        labels += labels_top

        # Create a single legend
        ax.legend(handles, labels, loc="lower left")
        # ax.grid(True, alpha=0.15)

        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax_top.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        fig.tight_layout()

        # Save the figure.
        plt.savefig(os.path.join(dir_name, f"mammals_{WHICH}.pdf"))
