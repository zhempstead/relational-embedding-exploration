"""
Reference implementation of node2vec.
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
"""

from collections import defaultdict
import random

from numpy.random import choice

# import pandas as pd
from tqdm import tqdm

from relational_embeddings.lib.graph_utils import read_graph

def node2vec_graph2text(indir, outdir, cfg):
    infile = indir / "edgelist"
    outfile = outdir / "text.txt"

    nx_G = read_graph(infile, cfg.weighted)
    print("Reading Done!")
    G = Graph(nx_G, cfg.p, cfg.q, cfg.weighted)
    print("Creation Done!")
    G.preprocess_transition_probs()
    print("Preprocess Done!")
    with open(outfile, "w") as f:
        print("Walk iteration:")
        for walk_iter in range(cfg.num_walks):
            print(walk_iter + 1, "/", cfg.num_walks)
            walks = G.simulate_walk(cfg.walk_length)
            for walk in walks:
                f.write(" ".join(walk) + "\n")
    print("Walking Done!")


class Graph:
    def __init__(self, nx_G, p, q, is_weighted, limit=1000):
        self.G = nx_G
        self.p = p
        self.q = q
        self.weighted = is_weighted
        self.adjList = [list(nx_G.neighbors(x)) for x in nx_G.nodes()]
        self.adjList_prob = [
            [nx_G[y][x]["weight"] for y in nx_G.neighbors(x)] for x in nx_G.nodes()
        ]
        self.adjList_prob = [
            [float(i) / sum(prob_vector) for i in prob_vector]
            for prob_vector in self.adjList_prob
        ]

        self.limit_dict = defaultdict(int)
        self.limit = limit

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        walk = [start_node]
        curr = walk[0]
        for i in range(walk_length):
            if self.adjList[curr] == []:
                break
            if self.weighted:
                nxt = choice(self.adjList[curr], p=self.adjList_prob[curr])
            else:
                nxt = choice(self.adjList[curr])
            self.limit_dict[nxt] += 1
            if self.limit_dict[nxt] >= self.limit:
                pass
                # idx = self.adjList[curr].index(nxt)
                # self.adjList[curr].pop(idx)
                # self.adjList_prob[curr].pop(idx)
                # norm_sum = sum(self.adjList_prob[curr])
                # self.adjList_prob[curr] = [float(i) / norm_sum for i in self.adjList_prob[curr]]
            else:
                walk.append(nxt)
            curr = nxt
        return list(map(lambda x: str(x), walk))

    def simulate_walk(self, walk_length, nodes=None):
        """
        Repeatedly simulate random walks from each node.
        """
        if nodes is None:
            nodes = list(self.G.nodes())
        random.shuffle(nodes)
        walks = [
            self.node2vec_walk(walk_length=walk_length, start_node=int(node))
            for node in tqdm(nodes)
        ]
        return walks

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        return
