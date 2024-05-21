"""Utilities for working with WordNet. """
from gpjax.scan import vscan
from jax import (
    random,
    value_and_grad,
)
import jax.numpy as jnp
from jaxtyping import Key

from experiment_data.wordnet import RelationsDataset
from experiment_utils.batch import (
    batch_transform,
    gather_slice,
    update_slice,
)
from experiment_utils.embeddings.metrics import reconstruction_metrics
from experiment_utils.embeddings.sampling import construct_sampler
from experiment_utils.objectives import word_embeddings
from manifold.geometry import (
    AbstractManifold,
    ProductSame,
)
from manifold.optimisers.averagers import default_averager


def minimise(
    key: Key,
    relations: RelationsDataset,
    embedding_manifold: AbstractManifold,
    main_optim=None,
    main_epochs: int = 500,
    burnin_optim: int = None,
    burnin_epochs: int = 0,
    num_negatives: int = 50,
    batch_size: int = 10,
):
    if main_optim is None:
        raise ValueError("Main optimiser must be specified.")

    # Define product manifold.
    manifold = ProductSame(embedding_manifold, len(relations.ids))
    loss = word_embeddings(embedding_manifold)

    # Initialise embeddings.
    init = random.uniform(
        key,
        minval=-1e-3,
        maxval=1e-3,
        shape=(len(relations.ids), embedding_manifold.dimension),
    )

    def build_step_fn(optim, averager, burnin: bool):
        sample_indicies = construct_sampler(
            relations, batch_size, num_negatives, burnin
        )

        def step_fn(carry, key):
            embeddings, state, average, average_state = carry

            # Sample indices.
            batch_idx = sample_indicies(key)

            # Compute value and padded Euclidean gradient.
            egrad = jnp.zeros_like(embeddings)
            value, egrad_idx = value_and_grad(loss)(embeddings[batch_idx])
            egrad = egrad.at[batch_idx].add(egrad_idx)

            # Compute Riemannian gradient.
            rg = manifold.euclidean_to_riemannian_gradient(embeddings, egrad)

            # Compute updates.
            updates, state = optim.update(manifold, rg, state, embeddings)

            # Compute exponential map.
            updates = manifold.exp(embeddings, updates)

            # Update average
            average, average_state = averager.update(
                manifold, updates, state, average_state, average
            )

            return (updates, state, average, average_state), value

        return step_fn

    steps_per_epoch = int(len(relations) / batch_size)

    # Run burnin if required.
    if burnin_epochs > 0:
        if burnin_optim is None:
            raise ValueError("Burnin optimiser must be specified if burnin_epochs > 0.")

        # Initialise burnin optimiser state and averager.
        state = burnin_optim.init(manifold, init)
        burnin_averager = default_averager(state)
        average, average_state = burnin_averager.init(manifold, init)

        # Build burnin step function.
        step_fn = build_step_fn(burnin_optim, burnin_averager, burnin=True)

        # Run burnin.
        key, subkey = random.split(key)
        (init, _, _, _), _ = vscan(
            step_fn,
            (init, state, average, average_state),
            random.split(subkey, steps_per_epoch * burnin_epochs),
            log_rate=100,
        )

    # Initialise optimiser state and averager.
    state = main_optim.init(manifold, init)
    averager = default_averager(state)
    average, average_state = averager.init(manifold, init)

    # Build main step function.
    step_fn = build_step_fn(main_optim, averager, burnin=False)

    # Run main optimisation.
    key, subkey = random.split(key)
    (embeddings, state, average, average_state), value = vscan(
        step_fn,
        (init, state, average, average_state),
        random.split(subkey, steps_per_epoch * main_epochs),
        log_rate=100,
    )

    # Evaluate metrics.
    iterate_rank, iterate_map = reconstruction_metrics(
        embedding_manifold, relations.adjacency, embeddings
    )
    average_rank, average_map = reconstruction_metrics(
        embedding_manifold, relations.adjacency, average
    )

    return {
        "iterate": dict(embeddings=embeddings, map=iterate_map, rank=iterate_rank),
        "average": dict(embeddings=average, map=average_map, rank=average_rank),
        "training_loss": value.reshape(main_epochs, -1).mean(axis=1),
    }


def minimise_sparse_productwise(  # noqa: PLR0915
    key: Key,
    relations: RelationsDataset,
    embedding_manifold: AbstractManifold,
    main_optim=None,
    main_epochs: int = 500,
    burnin_optim: int = None,
    burnin_epochs: int = 0,
    num_negatives: int = 50,
    batch_size: int = 10,
):
    if main_optim is None:
        raise ValueError("Main optimiser must be specified.")

    # Define product manifold.
    manifold = ProductSame(embedding_manifold, len(relations.ids))
    loss = word_embeddings(embedding_manifold)

    # Initialise embeddings.
    init = random.uniform(
        key,
        minval=-1e-3,
        maxval=1e-3,
        shape=(len(relations.ids), embedding_manifold.dimension),
    )

    def build_step_fn(optim, averager, burnin: bool):
        sample_indicies = construct_sampler(
            relations, batch_size, num_negatives, burnin
        )

        def step_fn(carry, key):
            embeddings, state, average, average_state = carry

            # Sample indices.
            batch_idx = sample_indicies(key)

            idx = batch_idx.flatten()

            state_idx = gather_slice(state, idx)
            average_state_idx = gather_slice(average_state, idx)

            # Compute value and padded Euclidean gradient.
            egrad = jnp.zeros_like(embeddings)
            value, egrad_idx = value_and_grad(loss)(embeddings[batch_idx])
            egrad_idx = egrad.at[batch_idx].add(egrad_idx)[idx]

            # Compute Riemannian gradient.
            rg_idx = manifold.euclidean_to_riemannian_gradient(
                embeddings[idx], egrad_idx
            )

            # Compute updates.
            updates_idx, state_idx = optim.update(
                embedding_manifold, rg_idx, state_idx, embeddings[idx]
            )

            # Compute exponential map.
            updates = embeddings.at[idx].set(manifold.exp(embeddings[idx], updates_idx))

            # Update average
            average_idx, average_state_idx = averager.update(
                embedding_manifold,
                updates[idx],
                state_idx,
                average_state_idx,
                average[idx],
            )

            average = average.at[idx].set(average_idx)

            # Update state.
            state = update_slice(state, idx, state_idx)
            average_state = update_slice(average_state, idx, average_state_idx)

            return (updates, state, average, average_state), value

        return step_fn

    steps_per_epoch = int(len(relations) / batch_size)

    # Run burnin if required.
    if burnin_epochs > 0:
        if burnin_optim is None:
            raise ValueError("Burnin optimiser must be specified if burnin_epochs > 0.")

        # Initialise burnin optimiser state and averager.
        burnin_optim = batch_transform(burnin_optim)
        state = burnin_optim.init(manifold, init)
        burnin_averager = batch_transform(
            default_averager(default_averager(gather_slice(state, 0)))
        )
        average, average_state = burnin_averager.init(manifold, init)

        # Build burnin step function.
        step_fn = build_step_fn(burnin_optim, burnin_averager, burnin=True)

        # Run burnin.
        key, subkey = random.split(key)
        (init, _, _, _), _ = vscan(
            step_fn,
            (init, state, average, average_state),
            random.split(subkey, steps_per_epoch * burnin_epochs),
            log_rate=100,
        )

    # Initialise optimiser state and averager.
    main_optim = batch_transform(main_optim)
    state = main_optim.init(embedding_manifold, init)
    averager = batch_transform(default_averager(gather_slice(state, 0)))
    average, average_state = averager.init(manifold, init)

    # Build main step function.
    step_fn = build_step_fn(main_optim, averager, burnin=False)

    # Run main optimisation.
    key, subkey = random.split(key)
    (embeddings, state, average, average_state), value = vscan(
        step_fn,
        (init, state, average, average_state),
        random.split(subkey, steps_per_epoch * main_epochs),
        log_rate=100,
    )

    # Evaluate metrics.
    iterate_rank, iterate_map = reconstruction_metrics(
        embedding_manifold, relations.adjacency, embeddings
    )
    average_rank, average_map = reconstruction_metrics(
        embedding_manifold, relations.adjacency, average
    )

    return {
        "iterate": dict(embeddings=embeddings, map=iterate_map, rank=iterate_rank),
        "average": dict(embeddings=average, map=average_map, rank=average_rank),
        "training_loss": value.reshape(main_epochs, -1).mean(axis=1),
    }
