"""
This file generates a bunch of random Bayes nets
"""
import pickle
from pyprojroot import here
from argparse import ArgumentParser
import sys
sys.path.extend(["./core", "../core", "./code/core"])
from bayes_net import create_random_bayes_net


parser = ArgumentParser()
parser.add_argument("--n_nets", type=int, default=10)
parser.add_argument("--n_nodes", type=int, default=10)
parser.add_argument("--n_edges", type=int, default=10)

if __name__ == "__main__":

    args = parser.parse_args()

    # create the bayes nets
    nets = []
    for i in range(args.n_nets):
        nets.append(create_random_bayes_net(args.n_nodes, args.n_edges))

    # pickle and save the nets
    for net in nets:
        net.to_pickle()
    with open(here(f"data/bayes_nets/nets_n-{args.n_nets}_nodes-{args.n_nodes}_edges-{args.n_edges}.pkl"), "wb") as f:
        pickle.dump(nets, f)
