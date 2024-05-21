"""Wordnet data. Adapted from code at https://github.com/facebookresearch/poincare-embeddings/tree/main/wordnet ."""
from dataclasses import dataclass
import os
import re

import jax.numpy as jnp
from nltk.corpus import wordnet as wn
import pandas
from tqdm import tqdm

try:
    wn.all_synsets  # noqa: B018
except LookupError:
    import nltk

    nltk.download("wordnet")

from experiment_utils.dataset import Dataset

__all__ = [
    "mammal_relations",
    "knowledge_relations",
    "mathematics_relations",
    "relations_from_nouns",
]

dir, _ = os.path.split(__file__)


def _local_path(filename: str) -> str:
    """Return the local path of a file."""
    return os.path.join(dir, filename)


def _process_closure():
    # make sure each edge is included only once
    edges = set()
    for synset in tqdm(wn.all_synsets(pos="n")):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))

        # also write transitive closure for all instances of a synset
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))

    nouns = pandas.DataFrame(list(edges), columns=["id1", "id2"])
    nouns["weight"] = 1

    # Extract the set of nouns that have "mammal.n.01" as a hypernym
    mammal_set = set(nouns[nouns.id2 == "mammal.n.01"].id1.unique())
    mammal_set.add("mammal.n.01")

    # Select relations that have a mammal as hypo and hypernym
    mammals = nouns[nouns.id1.isin(mammal_set) & nouns.id2.isin(mammal_set)]

    with open(_local_path("mammals_filter.txt"), "r") as fin:
        filt = re.compile(f'({"|".join([line.strip() for line in fin.readlines()])})')

    filtered_mammals = mammals[~mammals.id1.str.cat(" " + mammals.id2).str.match(filt)]

    if not os.path.exists(_local_path("_cache")):
        os.makedirs(_local_path("_cache"))

    nouns.to_csv(_local_path("_cache/noun_closure.csv"), index=False)
    filtered_mammals.to_csv(_local_path("_cache/mammal_closure.csv"), index=False)
    return filtered_mammals


def _process_knowledge_closure():
    # make sure each edge is included only once
    edges = set()
    for synset in tqdm(wn.all_synsets(pos="n")):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))

        # also write transitive closure for all instances of a synset
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))

    nouns = pandas.DataFrame(list(edges), columns=["id1", "id2"])
    nouns["weight"] = 1

    # Extract the set of nouns that have "knowledge.n.01" as a hypernym
    knowledge_set = set(nouns[nouns.id2 == "knowledge_domain.n.01"].id1.unique())
    knowledge_set.add("knowledge_domain.n.01")

    # Select relations that have a knowledge as hypo and hypernym
    knowledge = nouns[nouns.id1.isin(knowledge_set) & nouns.id2.isin(knowledge_set)]

    if not os.path.exists(_local_path("_cache")):
        os.makedirs(_local_path("_cache"))

    knowledge.to_csv(_local_path("_cache/knowledge_closure.csv"), index=False)
    return knowledge


def _process_mathematics_closure():
    # make sure each edge is included only once
    edges = set()
    for synset in tqdm(wn.all_synsets(pos="n")):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))

        # also write transitive closure for all instances of a synset
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))

    nouns = pandas.DataFrame(list(edges), columns=["id1", "id2"])
    nouns["weight"] = 1

    # Extract the set of nouns that have "knowledge.n.01" as a hypernym
    knowledge_set = set(nouns[nouns.id2 == "knowledge_domain.n.01"].id1.unique())
    knowledge_set.add("knowledge_domain.n.01")

    # Select relations that have a knowledge as hypo and hypernym
    knowledge = nouns[nouns.id1.isin(knowledge_set) & nouns.id2.isin(knowledge_set)]

    if not os.path.exists(_local_path("_cache")):
        os.makedirs(_local_path("_cache"))

    # Extract the set of nouns that have "mathematics.n.01" as a hypernym
    mathematics_set = set(knowledge[knowledge.id2 == "mathematics.n.01"].id1.unique())
    mathematics_set.add("mathematics.n.01")
    mathematics = knowledge[
        knowledge.id1.isin(mathematics_set) & knowledge.id2.isin(mathematics_set)
    ]

    mathematics.to_csv(_local_path("_cache/mathematics_closure.csv"), index=False)
    return mathematics


def _load_edge_list(path):
    df = pandas.read_csv(path, usecols=["id1", "id2", "weight"], engine="c")
    df.dropna(inplace=True)

    # Split id1 and id2 strings by the `.` character and take the first element.
    df.id1 = df.id1.str.split(".").str[0]
    df.id2 = df.id2.str.split(".").str[0]

    # Find index pairs of relations, `idx` and the unique objects, `objects`.
    idx, objects = pandas.factorize(df[["id1", "id2"]].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype("int")

    return jnp.array(idx), tuple(objects.tolist())


@dataclass
class RelationsDataset(Dataset):
    """Relations dataset."""

    ids: tuple = None
    adjacency: set = None

    @property
    def pairs(self):
        """Return the relation pairs."""
        return self.X

    @property
    def relations(self):
        """Return the relation pairs."""
        return self.X

    @property
    def valid_negatives(self):
        """Return the valid negatives as a matrix."""
        matrix = jnp.ones((len(self.ids), len(self.ids)))

        for k, v in self.adjacency.items():
            matrix = matrix.at[k, list(v)].set(0.0)

        matrix = matrix.at[jnp.diag_indices_from(matrix)].set(0.0)

        return matrix

    @property
    def degrees(self):
        """Return the degrees of the nodes."""
        return jnp.array([len(v) for v in self.adjacency.values()])


@dataclass
class MammalRelations(RelationsDataset):
    """Mammal relations dataset."""

    def __init__(self):
        """Initialize the dataset."""

        path = _local_path("_cache/mammal_closure.csv")

        # Check if the file exists
        if not os.path.exists(path):
            _process_closure()

        # Load the mammal relations.
        relations, ids = _load_edge_list(path)

        # Determine adjecency matrix.
        node_relations = {_: set() for _ in range(len(ids))}

        for u, v in relations:
            node_relations[int(u)].add(int(v))

        # Set the attributes.
        self.X = relations
        self.ids = ids
        self.adjacency = node_relations


@dataclass
class KnowledgeRelations(RelationsDataset):
    """Knowledge relations dataset."""

    def __init__(self):
        """Initialize the dataset."""

        path = _local_path("_cache/knowledge_closure.csv")

        # Check if the file exists
        if not os.path.exists(path):
            _process_knowledge_closure()

        # Load the knowledge relations.
        relations, ids = _load_edge_list(path)

        # Determine adjecency matrix.
        node_relations = {_: set() for _ in range(len(ids))}

        for u, v in relations:
            node_relations[int(u)].add(int(v))

        # Set the attributes.
        self.X = relations
        self.ids = ids
        self.adjacency = node_relations


@dataclass
class MathemathicsRelations(RelationsDataset):
    """Mathematics relations dataset."""

    def __init__(self):
        """Initialize the dataset."""

        path = _local_path("_cache/mathematics_closure.csv")

        # Check if the file exists
        if not os.path.exists(path):
            _process_mathematics_closure()

        # Load the knowledge relations.
        relations, ids = _load_edge_list(path)

        # Determine adjecency matrix.
        node_relations = {_: set() for _ in range(len(ids))}

        for u, v in relations:
            node_relations[int(u)].add(int(v))

        # Set the attributes.
        self.X = relations
        self.ids = ids
        self.adjacency = node_relations


def mammal_relations() -> MammalRelations:
    """Return the mammal relations dataset."""
    return MammalRelations()


def knowledge_relations() -> KnowledgeRelations:
    """Return the knowledge relations dataset."""
    return KnowledgeRelations()


def mathematics_relations() -> MathemathicsRelations:
    """Return the mathematics relations dataset."""
    return MathemathicsRelations()


def relations_from_nouns(hypernym: str) -> RelationsDataset:
    """Return the relations dataset."""

    @dataclass
    class RelationsFromNouns(RelationsDataset):
        """Relations dataset."""

        def __init__(self):
            """Initialize the dataset."""

            path = _local_path("_cache/noun_closure.csv")

            # Check if the file exists
            if not os.path.exists(path):
                _process_mathematics_closure()

            nouns = pandas.read_csv(path, usecols=["id1", "id2", "weight"], engine="c")

            filter_set = set(nouns[nouns.id2 == hypernym].id1.unique())
            filter_set.add(hypernym)

            df = nouns[nouns.id1.isin(filter_set) & nouns.id2.isin(filter_set)]
            df.dropna(inplace=True)

            # Split id1 and id2 strings by the `.` character and take the first element.
            df.id1 = df.id1.str.split(".").str[0]
            df.id2 = df.id2.str.split(".").str[0]

            # Find index pairs of relations, `idx` and the unique objects, `objects`.
            idx, objects = pandas.factorize(df[["id1", "id2"]].values.reshape(-1))
            idx = idx.reshape(-1, 2).astype("int")

            relations, ids = jnp.array(idx), tuple(objects.tolist())

            # Determine adjecency matrix.
            node_relations = {_: set() for _ in range(len(ids))}

            for u, v in relations:
                node_relations[int(u)].add(int(v))

            # Set the attributes.
            self.X = relations
            self.ids = ids
            self.adjacency = node_relations

    return RelationsFromNouns()
