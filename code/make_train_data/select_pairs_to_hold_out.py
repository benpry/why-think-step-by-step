"""
This file selects the pairs of variables to hold out from the training set.
"""
from argparse import ArgumentParser
import pandas as pd
from pyprojroot import here
import sys
sys.path.extend(["../core", "./code/core"])

parser = ArgumentParser()
parser.add_argument("--n_nets", type=int)
parser.add_argument("--n_pairs", type=int)

if __name__ == "__main__":

    args = parser.parse_args()

    mean_mi_rows = []

    for net_idx in range(args.n_nets):

        df_net = pd.read_csv(here(f"data/evaluation/mutual-informations/mutual-informations-net-{net_idx}.csv"))

        # filter out adjacent pairs
        df_distant = df_net[df_net["distance"] > 1]

        df_largest_effects = df_distant.sort_values("mutual_information", ascending=False).head(2 * args.n_pairs)
        df_selected_pairs = df_largest_effects.sample(args.n_pairs)

        mean_mi = df_selected_pairs['mutual_information'].mean()

        mean_mi_rows.append({
            "net_idx": net_idx,
            "mean_mi": mean_mi
        })

        df_selected_pairs.to_csv(here(f"data/training-data/selected-pairs/selected-pairs-net-{net_idx}.csv"), index=False)

    df_mean_mi = pd.DataFrame(mean_mi_rows)
    df_mean_mi = df_mean_mi.sort_values(by="mean_mi", ascending=False)
    df_mean_mi.to_csv(here(f"data/training-data/mean-mis-selected-pairs.csv"))
