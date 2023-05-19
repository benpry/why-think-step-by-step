"""
This file contains an implementation of the important functionality for Bayes nets.
"""
from typing import Iterable, Optional
import numpy as np
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node
from itertools import product
import networkx as nx

class BayesNet:
    """
    A Bayes Net for autoregressive inference
    """

    def __init__(self, model: BayesianNetwork, graph: nx.DiGraph):
        self.model = model
        self.graph = graph
        self.distances = None
        self.node_order = None

    def sample(self, n: int, evidences: Optional[Iterable] = None):
        if evidences is None:
            raw_samples = self.model.sample(n)
        else:
            raw_samples = self.model.sample(n, evidences=evidences, algorithm="gibbs")

        node_order = self.get_node_order()
        samples = [dict(zip(node_order, [int(x) for x in raw_sample])) for raw_sample in raw_samples]
        return samples

    def get_distances(self):
        if self.distances is None:
            distances = nx.all_pairs_shortest_path_length(self.graph.to_undirected())
            self.distances = {}
            for d in distances:
                self.distances[d[0]] = d[1]

        return self.distances

    def check_d_separated(self, query1, query2, conditioned_vars):
        distances = self.get_distances()

        # if they are infinite distance apart, they must be d-separated
        if query2 not in distances[query1]:
            return True

        # step 1: construct the ancestral graph
        ancestral_nodes = nx.ancestors(self.graph, query1) | nx.ancestors(self.graph, query2) | {query1, query2}
        for node in conditioned_vars:
            ancestral_nodes |= nx.ancestors(self.graph, node) | {node}
        ancestral_graph = self.graph.subgraph(ancestral_nodes)

        # step 2: moralize the ancestral graph
        moralized_graph = ancestral_graph.copy()
        edges_to_add = []
        for node in ancestral_graph.nodes:
            parents = [p for p in ancestral_graph.predecessors(node)]
            if len(parents) > 1:
                for p1, p2 in product(parents, parents):
                    if p1 != p2:
                        edges_to_add.append((p1, p2))

        moralized_graph.add_edges_from(edges_to_add)

        disoriented_graph = moralized_graph.to_undirected()

        # step 4: remove the conditions
        for node in conditioned_vars:
            if node in disoriented_graph:
                disoriented_graph.remove_node(node)

        # step 5: check if the query nodes are connected
        return not nx.has_path(disoriented_graph, query1, query2)

    def get_local_joint(self, center, distance) -> set:
        """
        Return a set of variables that are at most distance away from center
        """
        distances = self.get_distances()
        nodes = {n for n in distances[center] if distances[center][n] <= distance}

        return nodes

    def get_node_order(self):
        if (not hasattr(self, "node_order")) or (self.node_order is None):
            self.node_order = [x.name for x in self.model.graph.states if "joint" not in x.name]

        return self.node_order

    def predict_proba(self, evidence: dict):
        node_order = self.get_node_order()
        probs = self.model.predict_proba(evidence)
        prob_dict = dict(zip(node_order, probs))
        return prob_dict

    def predict_proba_with_intervention(self, intervention: dict):
        """
        Get probabilities of other variables given do(x)=i
        """
        topo_sort = list(nx.topological_sort(self.graph))
        curr_probs = self.predict_proba({})
        probs = {}
        for node in topo_sort:
            if node in intervention:
                curr_probs = self.predict_proba({node: intervention[node]})

            probs[node] = curr_probs[node]

        return probs

    def plot(self, *args, **kwargs):

        return self.model.plot(*args, **kwargs)

    def to_pickle(self):
        self.model = self.model.to_json()

    def from_pickle(self):
        self.model = BayesianNetwork.from_json(self.model)


def generate_random_edges(n_vars: int, n_edges: int):
    """
    Generate a random set of edges
    """
    all_pairs = [(i, j) for i in range(n_vars) for j in range(n_vars) if i < j]
    edges = [all_pairs[i] for i in np.random.choice(len(all_pairs), n_edges, replace=False)]
    # randomly flip half the edges
    edges = [(edge[1], edge[0]) if np.random.randint(0, 2) == 1 else edge for edge in edges]
    return edges


def contains_cycle(edges):
    """
    Check if the edges contain a cycle
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    try:
        nx.find_cycle(graph)
        return True
    except nx.NetworkXNoCycle:
        return False


def get_conditional_probability():
    return np.random.beta(0.2, 0.2)


def create_random_bayes_net(
    n_vars: int, n_edges: int
) -> BayesNet:
    """
    Generate a random Bayes net
    """

    edges = generate_random_edges(n_vars, n_edges)
    # randomly regenerate the edges until we have a DAG
    while contains_cycle(edges):
        edges = generate_random_edges(n_vars, n_edges)

    node_names = [f"X{i}" for i in range(n_vars)]
    # create the model graph
    graph = nx.DiGraph()
    for node_name in node_names:
        graph.add_node(node_name)
    for edge in edges:
        graph.add_edge(node_names[edge[0]], node_names[edge[1]])

    bayes_net_nodes = {}
    # make the probability tables
    for node in nx.topological_sort(graph):
        parents = [p for p in graph.predecessors(node)]
        n_parents = len(parents)
        # make the probability table
        if n_parents == 0:
            p = np.random.random()
            dist = DiscreteDistribution({True: p, False: 1 - p})
        else:
            conditions = [list(x) for x in product([True, False], repeat=n_parents + 1)]
            for i in range(0, len(conditions), 2):
                p = get_conditional_probability()
                not_p = 1 - p
                conditions[i].append(p)
                conditions[i + 1].append(not_p)

            dist = ConditionalProbabilityTable(
                conditions, [bayes_net_nodes[parent].distribution for parent in parents]
            )

        bayes_net_nodes[node] = Node(dist, name=node)

    # add the edges to the bayes net
    model = BayesianNetwork("Bayes Net")
    for n in node_names:
        model.add_state(bayes_net_nodes[n])
    for edge in graph.edges:
        model.add_edge(bayes_net_nodes[edge[0]], bayes_net_nodes[edge[1]])

    model.bake()
    bayes_net = BayesNet(model, graph)

    return bayes_net
