import os
import pickle

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Settings.
RESOLUTION = 25
RADIUS = 1.0

# Load experiment data.
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_name = os.path.join(dir_path, "results/toy.pkl")

with open(dir_name, "rb") as f:
    experiment_data = pickle.load(f)

sol = experiment_data["sol"]

# Create plots directory.
dir_name = os.path.join(dir_path, "plots")

print(experiment_data["rdog"])

# Create the directory
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

###################
# Make the Sphere #
###################

# Define the range for theta and phi
theta = np.linspace(0, 2.0 * np.pi, RESOLUTION)
phi = np.linspace(0, np.pi, RESOLUTION)

# Create a meshgrid for theta and phi
theta, phi = np.meshgrid(theta, phi)

# Calculate the Cartesian coordinates of each point on the sphere
x = RADIUS * np.sin(phi) * np.cos(theta)
y = RADIUS * np.sin(phi) * np.sin(theta)
z = RADIUS * np.cos(phi)

# Create a surface plot
surface = go.Surface(
    x=x,
    y=y,
    z=z,
    colorscale="Viridis",
    opacity=0.1,
    name="Surface",
    colorbar=dict(tickvals=[-1, 0, 1], title="f(x, y, z)"),
    showscale=False,
)


# Create line plots for the trajectories with numbered markers
rdog = go.Scatter3d(
    x=experiment_data["rdog"]["iterate"]["trajectory"][:, 0],
    y=experiment_data["rdog"]["iterate"]["trajectory"][:, 1],
    z=experiment_data["rdog"]["iterate"]["trajectory"][:, 2],
    mode="lines+markers+text",
    line=dict(color="blue", width=3),
    marker=dict(size=4),
    textposition="bottom center",
    textfont=dict(size=5),  # Adjust the text size here
    name="RDoG",
)

big_rsgd = go.Scatter3d(
    x=experiment_data["big_rsgd"]["iterate"]["trajectory"][:, 0],
    y=experiment_data["big_rsgd"]["iterate"]["trajectory"][:, 1],
    z=experiment_data["big_rsgd"]["iterate"]["trajectory"][:, 2],
    mode="lines+markers+text",
    line=dict(color="red", width=3),
    marker=dict(size=4),
    textposition="bottom center",
    textfont=dict(size=5),  # Adjust the text size here
    name="RSGD",
)

small_rsgd = go.Scatter3d(
    x=experiment_data["small_rsgd"]["iterate"]["trajectory"][:, 0],
    y=experiment_data["small_rsgd"]["iterate"]["trajectory"][:, 1],
    z=experiment_data["small_rsgd"]["iterate"]["trajectory"][:, 2],
    mode="lines+markers+text",
    line=dict(color="red", width=3),
    marker=dict(size=4),
    textposition="bottom center",
    textfont=dict(size=5),  # Adjust the text size here
    name="RSGD",
)


# Make small_rsgd plot:

layout = go.Layout(
    scene=dict(
        camera=dict(
            eye=dict(x=float(2 * sol[0]), y=float(2 * sol[1]), z=float(2 * sol[2])),
            center=dict(x=float(sol[0]), y=float(sol[1]), z=float(sol[2])),
            up=dict(x=2, y=2, z=2),
        ),
    ),
    legend=dict(
        x=0.27, y=0.9, orientation="h"
    ),  # Adjust x and y to position the legend, #dict(x=0.5, y=0.8),
    autosize=False,
    margin=dict(l=0, r=0, b=0, t=0),
)

# Create a marker for x_sol
x_sol_marker = go.Scatter3d(
    x=[sol[0], -sol[0]],
    y=[sol[1], -sol[1]],
    z=[sol[2], -sol[2]],
    mode="markers",
    marker=dict(color="black", size=8, symbol="cross"),
    showlegend=True,
    name="Optima",
)

# Create a figure and add the surface, lines, and marker
fig = go.Figure(data=[surface, rdog, small_rsgd, x_sol_marker], layout=layout)

# Create line plots for each row of the grid
for i in range(x.shape[0]):
    fig.add_trace(
        go.Scatter3d(
            x=x[i, :],
            y=y[i, :],
            z=z[i, :],
            mode="lines",
            line=dict(color="black", width=0.5),
            showlegend=False,
        )
    )

# Create line plots for each column of the grid
for i in range(x.shape[1]):
    fig.add_trace(
        go.Scatter3d(
            x=x[:, i],
            y=y[:, i],
            z=z[:, i],
            mode="lines",
            line=dict(color="black", width=0.5),
            showlegend=False,
        )
    )

fig.update_layout(
    scene=dict(
        xaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        ),
        zaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        ),
    ),
    font=dict(family="Ogg", color="black", size=20),
)

# Show the plot
fig.show()
pio.write_image(fig, dir_name + "/sphere_small.pdf", scale=5)


# Make big_rsgd plot:
layout = go.Layout(
    scene=dict(
        xaxis=dict(
            title=r"",
            tickvals=[-1, 1],
            showgrid=False,
            showticklabels=False,
            showbackground=False,
        ),
        yaxis=dict(
            title=r"",
            tickvals=[1],
            showgrid=False,
            showticklabels=False,
            showbackground=False,
        ),
        zaxis=dict(
            title=r"",
            tickvals=[-1, 1],
            showgrid=False,
            showticklabels=False,
            showbackground=False,
        ),
        camera=dict(
            eye=dict(x=float(2 * sol[0]), y=float(2 * sol[1]), z=float(2 * sol[2])),
            center=dict(x=float(sol[0]), y=float(sol[1]), z=float(sol[2])),
            up=dict(x=2, y=2, z=2),
        ),
    ),
    legend=dict(x=0.5, y=0.8),
    autosize=False,
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=False,
)

# Create a figure and add the surface, lines, and marker
fig = go.Figure(data=[surface, rdog, big_rsgd, x_sol_marker], layout=layout)

# Create line plots for each row of the grid
for i in range(x.shape[0]):
    fig.add_trace(
        go.Scatter3d(
            x=x[i, :],
            y=y[i, :],
            z=z[i, :],
            mode="lines",
            line=dict(color="black", width=0.5),
            showlegend=False,
        )
    )

# Create line plots for each column of the grid
for i in range(x.shape[1]):
    fig.add_trace(
        go.Scatter3d(
            x=x[:, i],
            y=y[:, i],
            z=z[:, i],
            mode="lines",
            line=dict(color="black", width=0.5),
            showlegend=False,
        )
    )

fig.update_layout(
    scene=dict(
        xaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        ),
        zaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        ),
    ),
    font=dict(family="Ogg", color="black", size=20),
)

# Show the plot
fig.show()
pio.write_image(fig, dir_name + "/sphere_big.pdf", scale=5)
