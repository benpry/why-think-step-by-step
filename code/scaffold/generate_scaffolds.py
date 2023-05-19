"""
This file computes scaffolds to reason through for trained language models
"""
import pickle
import pandas as pd
from argparse import ArgumentParser
from pyprojroot import here
from itertools import product
import networkx as nx
import sys
sys.path.extend(["../core", "./code/core"])


def get_scaffold(net, target_var, condition_var):
    """
    Get the order in which to generate variables in the scaffolded reasoning condition
    """
    distances = net.get_distances()

    # if they are infinite distance apart, they must already be d-separated
    if target_var not in distances[condition_var]:
        return []

    # step 1: construct the ancestral graph
    ancestral_nodes = nx.ancestors(net.graph, target_var) | nx.ancestors(net.graph, condition_var) | {target_var, condition_var}
    ancestral_graph = net.graph.subgraph(ancestral_nodes)

    # step 2: moralize the ancestral graph
    moralized_graph = ancestral_graph.copy()
    edges_to_add = []
    for node in ancestral_graph.nodes:
        parents = [p for p in ancestral_graph.predecessors(node)]
        if len(parents) > 1:
            for p1, p2 in product(parents, repeat=2):
                if p1 != p2:
                    edges_to_add.append((p1, p2))

    moralized_graph.add_edges_from(edges_to_add)

    disoriented_graph = moralized_graph.to_undirected()

    # return the minimum set of nodes that d-separates the target and condition
    scaffold_vars = nx.minimum_node_cut(disoriented_graph, s=target_var, t=condition_var)

    vars_with_distances = {}
    for var in scaffold_vars:
        vars_with_distances[var] = distances[condition_var][var]

    sorted_scaffold = sorted(vars_with_distances.keys(), key=vars_with_distances.get)

    return sorted_scaffold

parser = ArgumentParser()
parser.add_argument("--net-idx", type=int)
parser.add_argument("--num-scaffolds", type=int, default=50)
parser.add_argument("--bayes-net-file", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    net = pickle.load(open(here(args.bayes_net_file), "rb"))[args.net_idx]
    net.from_pickle()

    df_probs = pd.read_csv(here(f"data/evaluation/true-probs/true-probabilities-net-{args.net_idx}.csv"))

    df_selected_pairs = pd.read_csv(here(f"data/training-data/selected-pairs/selected-pairs-net-{args.net_idx}.csv"))
    # get half of the most influential univariate conditionals for scaffolding
    df_selected = pd.DataFrame()
    for i, row in df_selected_pairs.iterrows():
        df_selected = pd.concat((df_selected, df_probs[((df_probs["target_var"] == row["var1"]) &
                                                  (df_probs["condition_var"] == row["var2"])) |
                                                  ((df_probs["target_var"] == row["var2"]) &
                                                  (df_probs["condition_var"] == row["var1"]))]))

    all_orders = []
    for index, row in df_selected.iterrows():
        target_var, condition_var = row["target_var"], row["condition_var"]
        order = get_scaffold(net, target_var, condition_var)
        all_orders.append(order)

    df_selected["scaffold_order"] = all_orders
    df_selected.to_csv(here(f"data/scaffolds/scaffolds-net-{args.net_idx}.csv"), index=False)
