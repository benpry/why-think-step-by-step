"""
This file generates a training dataset which contains variable values from a bayes net.
"""
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from pyprojroot import here
from argparse import ArgumentParser
import sys
sys.path.extend(["./core", "../core", "./code/core"])
from create_sample_strs import create_target_sample_str
from collections import defaultdict
from bayes_net import BayesNet
from utils import pom_to_pgm

def generate_causal_samples(net: BayesNet, n: int):

    pgm_net = pom_to_pgm(net.model)
    # decide how many of each variable to make the intervention
    n_observational = 0
    # the number of causal samples to generate for each variable, value pair
    n_causal_samples = defaultdict(int)
    # decide how many of each sample type to take
    for i in range(n):
        # probability of generating an observational sample is 1
        if random.getrandbits(1):
            n_observational += 1
        else:
            # we're generating a causal sample, choose a variable and value to intervene on
            var = np.random.choice(net.graph.nodes)
            val = random.getrandbits(1)
            n_causal_samples[(var, val)] += 1

    # generate the observational samples
    observational_samples = net.sample(n_observational)
    observational_samples = [{"type": "observational", "sample": s} for s in observational_samples]

    causal_samples = []
    for (var, val), n_causal in n_causal_samples.items():
        # generate the causal samples
        condition_samples = pgm_net.simulate(n_causal, do={var: val}).to_dict(orient="records")
        # condition_samples = [{**s, var: val} for s in condition_samples]
        condition_samples = [{"type": "causal", "intervention_var": var, "intervention_val": val, "sample": s} for s in condition_samples]
        causal_samples.extend(condition_samples)

    all_samples = observational_samples + causal_samples
    random.shuffle(all_samples)

    return all_samples

def get_random_local_joint_vars(net: BayesNet, max_dist: int = 0):
    target = np.random.choice(net.graph.nodes)
    local_joint = net.get_local_joint(target, max_dist)
    return list(local_joint)

def get_all_vars_randomized(net: BayesNet):
    return np.random.choice(net.graph.nodes, size=len(net.graph.nodes), replace=False)

def get_random_vars(net: BayesNet, n: int):
    return np.random.choice(net.graph.nodes, size=n, replace=False)

def sample_vars_with_dropout(sample_fn, given_net, hidden_pairs, dropout_p, *args, min_length=1):
    """Sample avoiding held-out pairs"""
    selected_vars = []
    while len(selected_vars) < min_length:
        sample_vars = sample_fn(given_net, *args)
        selected_vars = [v for v in sample_vars if random.random() > dropout_p]
        # make sure there aren't any forbidden pairs
        for v1, v2 in hidden_pairs:
            if v1 in selected_vars and v2 in selected_vars:
                var_to_drop = np.random.choice((v1, v2))
                selected_vars.remove(var_to_drop)

    random.shuffle(selected_vars)
    return selected_vars

def format_sample(
        sample: dict,
        sample_format: str,
        net: BayesNet,
        local_joint_size=1,
        dropout_p=0.2,
        exp_p=0.5,
        zipf_k=2,
        pairs_to_avoid=(),
        other_net: Optional[BayesNet] = None,
        intervention_var = None
) -> str:
    if intervention_var is None:
        min_length=1
    else:
        min_length=2

    if sample_format == "fully_observed":
        target_var = np.random.choice(list(sample.keys()))
        return create_target_sample_str(sample, target_var)
    elif sample_format == "fully_observed_held_out":
        selected_vars = sample_vars_with_dropout(get_all_vars_randomized, net, pairs_to_avoid, 0, min_length=min_length)
    elif sample_format == "local_joint_exp":
        local_joint_size = np.random.geometric(exp_p)
        selected_vars = sample_vars_with_dropout(get_random_local_joint_vars, net, pairs_to_avoid,
                                                 dropout_p, local_joint_size, min_length=min_length)
    elif sample_format == "local_joint_zipf":
        local_joint_size = np.random.zipf(zipf_k)
        selected_vars = sample_vars_with_dropout(get_random_local_joint_vars, net, pairs_to_avoid,
                                                 dropout_p, local_joint_size, min_length=min_length)
    elif sample_format == "non_local_zipf":
        local_joint_size = np.random.zipf(zipf_k)
        local_joint = get_random_local_joint_vars(net, local_joint_size)
        local_joint_n_vars = len(local_joint)
        selected_vars = sample_vars_with_dropout(get_random_vars, net, pairs_to_avoid,
                                                 dropout_p, local_joint_n_vars, min_length=min_length)
    elif sample_format == "wrong_local_joint_exp":
        local_joint_size = np.random.geometric(exp_p)
        selected_vars = sample_vars_with_dropout(get_random_local_joint_vars,
                                                 other_net, pairs_to_avoid,
                                                 dropout_p, local_joint_size, min_length=min_length)
    elif sample_format == "wrong_local_joint_zipf":
        local_joint_size = np.random.zipf(zipf_k)
        selected_vars = sample_vars_with_dropout(get_random_local_joint_vars,
                                                 other_net, pairs_to_avoid,
                                                 dropout_p, local_joint_size, min_length=min_length)
    else:
        raise ValueError(f"Unknown sample format: {sample_format}")

    if intervention_var is not None:
        return create_target_sample_str(sample, selected_vars[-1], condition_var=intervention_var,
                                        var_order=selected_vars, intervention=True)
    else:
        return create_target_sample_str(sample, selected_vars[-1], var_order=selected_vars)

parser = ArgumentParser()
parser.add_argument("-n", type=int, default=1000000)
parser.add_argument("--sample-format", default="fully_observed")
parser.add_argument("--sample-format-str")
parser.add_argument("--bayes-net-file")
parser.add_argument("--net-id", type=int, default=0)
parser.add_argument("--dropout-p", type=float, default=0.2)
parser.add_argument("--exp-p", type=float, default=0.33)
parser.add_argument("--zipf-k", type=float, default=2)
parser.add_argument("--causal", action="store_true")

if __name__ == "__main__":

    args = parser.parse_args()
    nets = pickle.load(open(here(args.bayes_net_file), "rb"))
    for net in nets:
        net.from_pickle()
    net = nets[args.net_id]
    # the other net for the wrong local joints
    other_net_id = (args.net_id + 1) % len(nets)
    other_net = nets[other_net_id]

    df_pairs = pd.read_csv(here(f"data/training-data/selected-pairs/selected-pairs-net-{args.net_id}.csv"))
    pairs_to_avoid = set(zip(df_pairs["var1"].tolist(), df_pairs["var2"].tolist())) | set(zip(df_pairs["var2"].tolist(), df_pairs["var1"].tolist()))

    datasets = {"train": [], "eval": []}
    for dataset in datasets:
        print(f"Generating {dataset} dataset")
        if args.causal: # generate all samples at once if we don't need causal samples
            all_samples = generate_causal_samples(net, args.n)
        else:
            all_samples = net.sample(args.n)

        for sample in tqdm(all_samples):

            if args.causal:
                if sample["type"] == "causal":
                    intervention_var = sample["intervention_var"]
                else:
                    intervention_var = None
                # we don't need the sample meta-information anymore
                sample = sample["sample"]
            else:
                intervention_var = None

            sample_str = format_sample(
                sample,
                args.sample_format,
                net=net,
                dropout_p=args.dropout_p,
                exp_p=args.exp_p,
                zipf_k=args.zipf_k,
                pairs_to_avoid=pairs_to_avoid,
                other_net=other_net,
                intervention_var=intervention_var
            )

            datasets[dataset].append(sample_str)

    for dataset in datasets:
        if args.causal:
            pd.DataFrame({"text": datasets[dataset]}).to_csv(here(f"data/training-data/samples/{dataset}_samples_{args.sample_format_str}_net_{args.net_id}_causal.csv"), index=False)
        else:
            pd.DataFrame({"text": datasets[dataset]}).to_csv(here(f"data/training-data/samples/{dataset}_samples_{args.sample_format_str}_net_{args.net_id}.csv"), index=False)
