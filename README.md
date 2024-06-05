# Learning-Rate-Free Stochastic Riemannian Optimization in JAX

This repository contains the implementation of the algorithms in the paper [Learning-Rate-Free Stochastic Optimization over Riemannian Manifolds](http://www.arxiv.org/pdf/2406.02296) by Daniel Dodd, Louis Sharrock, and Christopher Nemeth.

## 🛠️ Install via poetry
```
git clone https://github.com/daniel-dodd/riemannian_dog.git
cd riemannian_dog
poetry install
```

## 🧪Run tests with Pytest
```
cd riemannian_dog
pytest
```

## 🧐 Example
We consider the problem of maximizing the Rayleigh quotient $\frac{x^TAx}{2\|x\|_2^2}$ over $\mathbb{R}^d$, i.e. of finding the dominant eigenvector of $A\in\mathbb{R}^{d\times d}$. This non-convex problem can be written on the open hemisphere $\mathbb{S}^{d-1}$ and is known to be geodesically gradient-dominated.

We begin by importing necessary dependencies and define a key for reproducibility.
```python
import jax.numpy as jnp
from jaxtyping import Float, Array, Key
from jax import random, config, vmap
config.update("jax_enable_x64", True)

from manifold.geometry import Sphere
from manifold.optimisers import rdog
from manifold.gradient import rgrad

# Settings.
SEED = 123
DIMENSION = 10

key = random.PRNGKey(SEED)
```

Now we generate a positive definite matrix that we wish to compute the Rayleigh quotient of.
```python
key, subkey = random.split(key)
sqrt = random.normal(subkey, shape = (DIMENSION, DIMENSION))
matrix = sqrt @ sqrt.T
```

Next we write our loss function.
```python
def loss(point: Float[Array, "N"], matrix: Float[Array, "N N"]) -> Float[Array, ""]:
    """Rayleigh quotient objective.

    Args:
        point: Point to evaluate the Rayleigh quotient at.
        matrix: Matrix we are evaluating the Rayleigh quotient of.

    Returns:
        Rayleigh quotient objective.
    """
    return -0.5 * jnp.dot(point, jnp.dot(matrix, point))
```

Now we produce to define our `Sphere` manifold and generate an initial point via the `.random_point` method.
```python
manifold = Sphere(DIMENSION)
key, subkey = random.split(key)
point = manifold.random_point(subkey)
```

We define our parameter-free optimiser.
```python
opt = rdog()
opt_state = opt.init(manifold, point)
```
And optimise!
```python
for _ in range(500):
    rg = rgrad(manifold, loss)(point, matrix) # I'm a Riemannian gradient!
    updates, opt_state = opt.update(manifold, rg, opt_state, point)
    point = manifold.exp(point, updates)
```
Finally to sense check what we have run, we compare to the traditional numerical solution.
```python
eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
max_eig = jnp.max(eigenvalues)
min_eig = jnp.min(eigenvalues)
sol = eigenvectors[:, jnp.argmax(eigenvalues)]/jnp.linalg.norm(eigenvectors[:, jnp.argmax(eigenvalues)])
```
Computing the distance between the numerical solution and a local optima.
```python
jnp.mininum(manifold.distance(point, sol), manifold.distance(point, -sol))
```
We see that we obtain the same answer!

## 🔬 Run experiments
The experiment scipts are contained in the `experiments` directory.

- `experiments/toy` is code to reproduce Figure 1. Please run `experiments/toy/toy.py`to run the experiments and cache the results, followed by `experiments/toy/plot_toy.py` to generate the plot.
- `experiments/sphere_rayleigh` is code to reproduce Figure 2. Please run `experiments/sphere_rayleigh/sphere.py` to run the experiments and cache the results, followed by `experiments/sphere_rayleigh/plot_sphere.py` to generate the plot.
- `experiments/grassmann_pca` is code to reproduce Figure 3. For (a) please run `experiments/grassmann_pca/wine.py` to run the experiments and cache the results, followed by `experiments/grassmann_pca/plot_wine.py` to generate the plot. For (b) please run `experiments/grassmann_pca/waveform.py` to run the experiments and cache the results, followed by `experiments/grassmann_pca/plot_waveform.py` to generate the plot. For (c) please run `experiments/grassmann_pca/tiny_image_net.py` to run the experiments and cache the results, followed by `experiments/grassmann_pca/plot_tiny_image_net.py` to generate the plot.
- `experiments/poincare_wordnet` is code to reproduce Figure 4. For (a) please run the files in `experiments/five_dimensional` followed by the plotting scripts. For (b) please run the files in `experiments/two_dimensional` followed by the plotting scripts.
