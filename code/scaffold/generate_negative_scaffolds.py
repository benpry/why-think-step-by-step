"""
This file creates negative scaffolds which are like scaffolds but with irrelevant variables.
"""
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pyprojroot import here
from itertools import product
from generate_scaffolds import get_scaffold
import networkx as nx
import sys
sys.path.extend(["../core", "./code/core"])


def get_negative_scaffold(net, target_var, condition_var):
    """
    Get the order in which to generate variables in the scaffolded reasoning condition
    """
    scaffold_vars = get_scaffold(net, target_var, condition_var)

    negative_scaffold_size = len(scaffold_vars)
    potential_negative_scaffold_vars = list(set(net.graph.nodes) - {target_var, condition_var, *scaffold_vars})

    return np.random.choice(potential_negative_scaffold_vars, negative_scaffold_size)

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

    all_negative_scaffolds = []
    for index, row in df_selected.iterrows():
        target_var, condition_var = row["target_var"], row["condition_var"]
        order = get_negative_scaffold(net, target_var, condition_var)
        all_negative_scaffolds.append(order)

    df_selected["scaffold_order"] = all_negative_scaffolds
    df_selected.to_csv(here(f"data/scaffolds/negative-scaffolds-net-{args.net_idx}.csv"), index=False)
