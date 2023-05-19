"""
This file computes the true conditional probabilities in a Bayes net.
"""
import pandas as pd
import numpy as np
import pickle
from itertools import product
from argparse import ArgumentParser
import sys
sys.path.extend(["../core", "./code/core"])
from pyprojroot import here

def get_probs_from_samples(samples, target_var, condition_var, condition_val):
    condition_samples = [s for s in samples if s[condition_var] == condition_val]
    target_prob = np.mean([s[target_var] for s in condition_samples])

    # get the difference between the true marginal and the conditional
    marginal = np.mean([s[target_var] for s in samples])

    return target_prob, marginal

def get_probs_analytically(net, target_var, condition_var, condition_val):
    probs = net.predict_proba({condition_var: condition_val})
    target_dist = probs[target_var]
    target_prob = target_dist.probability(True)

    # get the difference between the true marginal and the conditional
    marginal = net.predict_proba({})[target_var].probability(True)

    return target_prob, marginal

parser = ArgumentParser()
parser.add_argument("--net_idx", type=int)
parser.add_argument("--bayes-net-file", type=str)

if __name__ == "__main__":

    args = parser.parse_args()

    nets = pickle.load(open(here(args.bayes_net_file), "rb"))
    net = nets[args.net_idx]
    net.from_pickle()
    net.node_order = None
    all_vars = net.graph.nodes

    distances = net.get_distances()

    rows = []
    for target, condition in product(all_vars, all_vars):

        if target == condition:
            continue

        if target in distances[condition]:
            distance = int(distances[condition][target])
        else:
            distance = -1

        for condition_val in (False, True):

            target_prob, marginal = get_probs_analytically(net, target, condition, condition_val)
            condition_influence = abs(target_prob - marginal)

            rows.append({
                "target_var": target,
                "condition_var": condition,
                "condition_val": condition_val,
                "cond_target_dist": distance,
                "prob": target_prob,
                "marginal": marginal,
                "condition_influence": condition_influence
            })

    df_true = pd.DataFrame(rows)
    df_true.to_csv(here(f"data/evaluation/true-probs/true-probabilities-net-{args.net_idx}.csv"))

