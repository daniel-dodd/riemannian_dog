import os
import pickle

from adjustText import adjust_text
import jax.numpy as jnp
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from experiment_data.wordnet import mammal_relations

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


# Create plots directory.
dir_name = os.path.join(dir_path, "plots")

# Create the directory
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


with open(os.path.join(dir_path, "results/mammals.pkl"), "rb") as f:
    experiment_data_all = pickle.load(f)

# Plot the results at the final iteration for each metric.
METRICS = {
    "map": dict(label=r"Mean average precision", yscale="linear"),
}

NUM_POINTS = 50

data = mammal_relations()
ids = data.ids
degrees = data.degrees
pairs = data.pairs


for WHICH in ["sensitivity_noc", "sensitivity"]:
    for opt in experiment_data_all[0][WHICH]:
        embeddings = experiment_data_all[0][WHICH][opt][0]["iterate"]["embeddings"]

        fig, ax = plt.subplots(figsize=(10, 10))

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Add a unit circle
        circle = Circle(
            (0, 0), 1, edgecolor="black", facecolor="none", linewidth=1, alpha=0.9
        )
        ax.add_patch(circle)

        texts = []  # list to store the labels
        for pair in pairs:
            x_values = [embeddings[pair[0]][0], embeddings[pair[1]][0]]
            y_values = [embeddings[pair[0]][1], embeddings[pair[1]][1]]
            ax.plot(
                x_values,
                y_values,
                ".-",
                alpha=0.1,
                linewidth=0.2,
                markersize=0.2,
                color="black",
                zorder=-1,
                rasterized=True,
            )

        # Plot each point
        for i in range(NUM_POINTS):
            x, y, *_ = embeddings[i]
            ax.scatter(x, y, s=50, alpha=0.6)
            text = ids[i].split("_")
            text[0] = text[0].title()
            transformed_text = " ".join(text)
            texts.append(ax.text(x, y, transformed_text, fontsize=14, zorder=10))

        # Adjust the labels to avoid overlaps
        adjust_text(texts)

        # Set the aspect of the plot to be equal, so the circle will appear as a circle
        ax.set_aspect("equal")

        plt.savefig(
            os.path.join(dir_name, f"embeddings_{opt}_{WHICH}.pdf"),
            dpi=450,
            bbox_inches="tight",
        )


best = jnp.array(
    [
        experiment_data_all[0]["learning_rate"]["rsgd"][_][WHAT]["map"]
        for _ in range(len(LEARNING_RATES))
    ]
).argmax()


print(best)

embeddings = experiment_data_all[0]["learning_rate"]["rsgd"][best]["iterate"][
    "embeddings"
]

fig, ax = plt.subplots(figsize=(10, 10))

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")

# Add a unit circle
circle = Circle((0, 0), 1, edgecolor="black", facecolor="none", linewidth=1, alpha=0.9)
ax.add_patch(circle)

texts = []  # list to store the labels
for pair in pairs:
    x_values = [embeddings[pair[0]][0], embeddings[pair[1]][0]]
    y_values = [embeddings[pair[0]][1], embeddings[pair[1]][1]]
    ax.plot(
        x_values,
        y_values,
        ".-",
        alpha=0.1,
        linewidth=0.2,
        markersize=0.2,
        color="black",
        zorder=-1,
        rasterized=True,
    )

# Plot each point
for i in range(NUM_POINTS):
    x, y, *_ = embeddings[i]
    ax.scatter(x, y, s=50, alpha=0.6)
    text = ids[i].split("_")
    text[0] = text[0].title()
    transformed_text = " ".join(text)
    texts.append(ax.text(x, y, transformed_text, fontsize=14, zorder=10))

# Adjust the labels to avoid overlaps
adjust_text(texts)

# Set the aspect of the plot to be equal, so the circle will appear as a circle
ax.set_aspect("equal")

plt.savefig(os.path.join(dir_name, "embeddings_rsgd.pdf"), dpi=450, bbox_inches="tight")
