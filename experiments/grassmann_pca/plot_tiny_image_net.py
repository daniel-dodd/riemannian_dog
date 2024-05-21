import os
import pickle
from typing import Iterable

from jax import tree_util as jtu
import jax.numpy as jnp
from jaxtyping import PyTree
from matplotlib import (
    cm,
    colors,
    lines,
    ticker,
)
import matplotlib.pyplot as plt
import numpy as np

# Set the style and font.
from experiment_utils.plots import (
    get_cols,
    set_style_and_font,
)

set_style_and_font()
cols = get_cols()

# Settings.
WHAT = "average"
ITERATION = 2000

LEARNING_RATES = jnp.logspace(-8, 6, num=20)
SENSITIVITY = jnp.logspace(-8, 6, num=20)

OPTIMS_TO_PLOT = {
    "rdog": dict(color=cols[0], label="RDoG (Ours)"),
    "rdowg": dict(color=cols[1], label="RDoWG (Ours)"),
    "rsgd": dict(color=cols[2], label="RSGD"),
    "radam": dict(color=cols[3], label="RADAM"),
    "nrdog": dict(color=cols[4], label="NRDoG (Ours)"),
}

# Load experiment data.
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_name = os.path.join(dir_path, "results/tiny_image_net.pkl")

with open(dir_name, "rb") as f:
    experiment_data_all = pickle.load(f)

# Create plots directory.
dir_name = os.path.join(dir_path, "plots_tiny_image_net")

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

# Plot the results at the final iteration for each metric.
METRICS = {
    # "regret": dict(label=rf"Regret", yscale="linear"),
    "distance": dict(label=r"Geodesic distance to optima", yscale="linear"),
}

MAX_ITER = 500

for ITERATION in [100, 500, 1000, 2000, 5000]:
    for m in METRICS:
        fig, ax = plt.subplots()
        ax_top = ax.twiny()

        ax.set_xscale("log")
        ax_top.set_xscale("log")

        ax.set_xlabel(r"Learning rate $\eta$")
        ax_top.set_xlabel(r"Initial distance $\epsilon$")
        ax_top.set_xlim(1e-8, 1e-1)

        # Plot parameter-free optimisers over the range of initial distance estimates.
        for opt in mean["sensitivity"]:
            ax_top.plot(
                SENSITIVITY,
                [
                    mean["sensitivity"][opt][_][WHAT][m][ITERATION]
                    for _ in range(len(SENSITIVITY))
                ],
                ".-",
                label=OPTIMS_TO_PLOT[opt]["label"],
                color=OPTIMS_TO_PLOT[opt]["color"],
            )
            ax_top.fill_between(
                SENSITIVITY,
                [
                    mean["sensitivity"][opt][_][WHAT][m][ITERATION]
                    - standard_error["sensitivity"][opt][_][WHAT][m][ITERATION]
                    for _ in range(len(SENSITIVITY))
                ],
                [
                    mean["sensitivity"][opt][_][WHAT][m][ITERATION]
                    + standard_error["sensitivity"][opt][_][WHAT][m][ITERATION]
                    for _ in range(len(SENSITIVITY))
                ],
                alpha=0.15,
                color=OPTIMS_TO_PLOT[opt]["color"],
            )

        # Plot optimisers over the range of learning rates.
        for opt in mean["learning_rate"]:
            ax.plot(
                LEARNING_RATES,
                [
                    mean["learning_rate"][opt][_]["iterate"][m][ITERATION]
                    for _ in range(len(LEARNING_RATES))
                ],
                ".-",
                label=OPTIMS_TO_PLOT[opt]["label"],
                color=OPTIMS_TO_PLOT[opt]["color"],
            )

            ax.fill_between(
                LEARNING_RATES,
                [
                    mean["learning_rate"][opt][_]["iterate"][m][ITERATION]
                    - standard_error["learning_rate"][opt][_][WHAT][m][ITERATION]
                    for _ in range(len(LEARNING_RATES))
                ],
                [
                    mean["learning_rate"][opt][_]["iterate"][m][ITERATION]
                    + standard_error["learning_rate"][opt][_]["iterate"][m][ITERATION]
                    for _ in range(len(LEARNING_RATES))
                ],
                alpha=0.15,
                color=OPTIMS_TO_PLOT[opt]["color"],
            )

        # Set the y-axis label.
        ax.set_ylabel(METRICS[m]["label"])
        ax.set_yscale(METRICS[m]["yscale"])

        # Get the legend handles and labels from both axes
        handles, labels = ax.get_legend_handles_labels()
        handles_top, labels_top = ax_top.get_legend_handles_labels()

        # Combine the handles and labels
        handles += handles_top
        labels += labels_top

        # Create a single legend
        ax.legend(handles, labels, loc="center left")

        # Set the major ticks for both axes
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax_top.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        fig.tight_layout()

        # Save the figure.
        plt.savefig(os.path.join(dir_name, f"tiny_image_net_{m}_{ITERATION}.pdf"))


# Plot trajectories of the optimisers for regret.
WHAT = "average"
for METRIC in METRICS:
    for WHICH in ["rdowg", "rdog", "nrdog"]:
        fig, ax = plt.subplots()  # Create a Figure and an Axes object
        cmap = cm.get_cmap("viridis")  # Choose a colormap

        # Normalize the log of the sensitivity values to [0, 1]
        normalize = colors.LogNorm(
            vmin=np.min(SENSITIVITY[SENSITIVITY > 0]),
            vmax=np.max(SENSITIVITY[SENSITIVITY < 2.0]),
        )

        for i, s in enumerate(SENSITIVITY):
            if s < 2.0:
                # Plot mean.
                ax.plot(
                    mean["sensitivity"][WHICH][i][WHAT][METRIC][:MAX_ITER],
                    "-.",
                    color=cmap(normalize(s)),
                )  # Color the line based on the normalized log sensitivity value
                # Plot standard error.
                ax.fill_between(
                    range(len(mean["sensitivity"][WHICH][i][WHAT][METRIC][:MAX_ITER])),
                    mean["sensitivity"][WHICH][i][WHAT][METRIC][:MAX_ITER]
                    - standard_error["sensitivity"][WHICH][i][WHAT][METRIC][:MAX_ITER],
                    mean["sensitivity"][WHICH][i][WHAT][METRIC][:MAX_ITER]
                    + standard_error["sensitivity"][WHICH][i][WHAT][METRIC][:MAX_ITER],
                    alpha=0.15,
                    color=cmap(normalize(s)),
                )

        # Plot best rsgd.

        best = jnp.array(
            [
                mean["learning_rate"]["rsgd"][_][WHAT][METRIC][-1]
                for _ in range(len(LEARNING_RATES))
            ]
        ).argmin()

        ax.plot(
            mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER],
            "--",
            color="black",
            label="Best RSGD",
        )  # Color the line based on the normalized log sensitivity value
        # Plot standard error.
        ax.fill_between(
            range(len(mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER])),
            mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER]
            - standard_error["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER],
            mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER]
            + standard_error["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER],
            alpha=0.15,
            color="black",
        )

        # plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=ax)  # Add a colorbar
        cbar = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=ax)
        cbar.set_label(r"Initial distance $\epsilon$")
        cbar.locator = ticker.LogLocator()  # Set the ticks to a log scale
        cbar.update_ticks()

        # else:
        ax.set_xlim(0, 500)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(METRICS[METRIC]["label"])

        # Create a custom legend entry
        rdog_line = lines.Line2D(
            [], [], linestyle="-.", color="black", label=OPTIMS_TO_PLOT[WHICH]["label"]
        )

        # Add the custom legend entry to the existing legend
        handles, labels = ax.get_legend_handles_labels()
        handles.append(rdog_line)
        ax.legend(handles=handles)
        fig.tight_layout()
        plt.savefig(
            os.path.join(
                dir_name, f"tiny_image_net_sensitivity_trace_{WHICH}_{METRIC}.pdf"
            )
        )


# Plot trajectories of the optimisers for regret.
WHAT = "average"
for METRIC in METRICS:
    for WHICH in ["radam", "rsgd"]:
        fig, ax = plt.subplots()  # Create a Figure and an Axes object
        cmap = cm.get_cmap("viridis")  # Choose a colormap

        # Normalize the log of the sensitivity values to [0, 1]
        normalize = colors.LogNorm(
            vmin=np.min(LEARNING_RATES),
            vmax=np.max(LEARNING_RATES[LEARNING_RATES < 2.0]),
        )

        for i, lr in enumerate(LEARNING_RATES):
            if lr < 2.0:
                # Plot mean.
                ax.plot(
                    mean["learning_rate"][WHICH][i][WHAT][METRIC][:MAX_ITER],
                    "-.",
                    color=cmap(normalize(lr)),
                )  # Color the line based on the normalized log sensitivity value
                # Plot standard error.
                ax.fill_between(
                    range(
                        len(mean["learning_rate"][WHICH][i][WHAT][METRIC][:MAX_ITER])
                    ),
                    mean["learning_rate"][WHICH][i][WHAT][METRIC][:MAX_ITER]
                    - standard_error["learning_rate"][WHICH][i][WHAT][METRIC][
                        :MAX_ITER
                    ],
                    mean["learning_rate"][WHICH][i][WHAT][METRIC][:MAX_ITER]
                    + standard_error["learning_rate"][WHICH][i][WHAT][METRIC][
                        :MAX_ITER
                    ],
                    alpha=0.15,
                    color=cmap(normalize(lr)),
                )

        # Plot best rsgd.
        best = jnp.array(
            [
                mean["learning_rate"]["rsgd"][_][WHAT][METRIC][:MAX_ITER][-1]
                for _ in range(len(LEARNING_RATES))
            ]
        ).argmin()

        ax.plot(
            mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER],
            "--",
            color="black",
            label="Best RSGD",
        )  # Color the line based on the normalized log sensitivity value
        # Plot standard error.
        ax.fill_between(
            range(len(mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER])),
            mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER]
            - standard_error["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER],
            mean["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER]
            + standard_error["learning_rate"]["rsgd"][best][WHAT][METRIC][:MAX_ITER],
            alpha=0.15,
            color="black",
        )

        # plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=ax)  # Add a colorbar
        cbar = plt.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=ax)
        cbar.set_label(r"Learning rate $\eta$")
        cbar.locator = ticker.LogLocator()  # Set the ticks to a log scale
        cbar.update_ticks()

        ax.set_xlim(0, 500)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(METRICS[METRIC]["label"])

        # Create a custom legend entry
        rdog_line = lines.Line2D(
            [], [], linestyle="-.", color="black", label=OPTIMS_TO_PLOT[WHICH]["label"]
        )

        # Add the custom legend entry to the existing legend
        handles, labels = ax.get_legend_handles_labels()
        handles.append(rdog_line)
        ax.legend(handles=handles)
        fig.tight_layout()
        plt.savefig(
            os.path.join(
                dir_name, f"tiny_image_net_sensitivity_trace_{WHICH}_{METRIC}.pdf"
            )
        )
