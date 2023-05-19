"""
This file computes the mutual information between every pair of variables in the specificed Bayes net
"""
import pandas as pd
import numpy as np
import pickle
from itertools import product
from argparse import ArgumentParser
from tqdm import tqdm
import sys
sys.path.extend(["../core", "./code/core"])
from pyprojroot import here

def compute_mutual_information(x, y, net, marginals):
    """compute the mutual information between two variables in the bayes net"""
    kl_divergence = 0
    for x_val, y_val in product((False,True), repeat=2):
        joint_logprob = marginals[x][x_val] + \
            net.predict_proba({x: x_val})[y].log_probability(y_val)

        x_marginal = marginals[x][x_val]
        y_marginal = marginals[y][y_val]

        kl_divergence += np.exp(joint_logprob) * (joint_logprob - x_marginal - y_marginal)

    return kl_divergence

parser = ArgumentParser()
parser.add_argument("--bayes-net-file", type=str)
if __name__ == "__main__":

    args = parser.parse_args()

    nets = pickle.load(open(here(args.bayes_net_file), "rb"))

    for net_idx, net in enumerate(nets):
        net.from_pickle()
        net.node_order = None
        all_vars = net.graph.nodes
        distances = net.get_distances()

        marginals = {}
        for variable in all_vars:
            marginals[variable] = {}
            for val in (False, True):
                marginals[variable][val] = net.predict_proba({})[variable].log_probability(val)

        rows = []
        for var1 in tqdm(all_vars):
            for var2 in all_vars:
                if var1 == var2:
                    break

                if var2 in distances[var1]:
                    distance = int(distances[var1][var2])
                else:
                    distance = -1

                if net.check_d_separated(var1, var2, set()):
                    mi = 0
                else:
                    mi = compute_mutual_information(var1, var2, net, marginals)
                rows.append({
                    "var1": var1,
                    "var2": var2,
                    "distance": distance,
                    "mutual_information": mi,
                })

        df_mi = pd.DataFrame(rows)
        df_mi.to_csv(here(f"data/evaluation/mutual-informations/mutual-informations-net-{net_idx}.csv"))
