import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scipy as sp

from typing import Callable, Dict, List, Optional

from cassiopeia.solver import GreedySolver
from cassiopeia.solver import graph_utils

class SpectralSolver(GreedySolver.GreedySolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: str,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        threshold: Optional[int] = 0,
        weights: Optional[Dict] = None
    ):

        super().__init__(character_matrix, missing_char, meta_data, priors)

        self.threshold = threshold
        self.weights = weights
    
    def perform_split(self, samples: List[int] = None, display=False):

        G = graph_utils.construct_similarity_graph(self.prune_cm, samples, self.missing_char, threshold = self.threshold, w = self.weights)

        L = nx.normalized_laplacian_matrix(G).todense()
        diag = sp.linalg.eig(L)
        v2 = diag[1][:, 1] 
        x = {}
        vertices = list(G.nodes())
        for i in range(len(vertices)):
            x[vertices[i]] = v2[i]
        vertices.sort(key=lambda v: x[v])
        total_weight = 2*sum([G[e[0]][e[1]]['weight'] for e in G.edges()])
        S = set()
        num = 0
        denom = 0
        best_score = 10000000
        best_index = 0
        for i in range(len(vertices) - 1):
            v = vertices[i]
            S.add(v)
            cut_edges = 0
            neighbor_weight = 0
            for w in G.neighbors(v):
                neighbor_weight += G[v][w]['weight']
                if w in S:
                    cut_edges += G[v][w]['weight']
            denom += neighbor_weight
            num += neighbor_weight - 2*cut_edges
            if num == 0:
                best_index = i
                break
            if num/min(denom, total_weight-denom) < best_score:
                best_score = num/min(denom, total_weight-denom)
                best_index = i
        if display:
            print("number of samples = ", len(v2))
            print("lambda2 = ", diag[0][1])
            plt.hist(v2, density=True, bins=30)
            plt.hist([x[v] for v in vertices[:best_index+1]], density=True, bins=30)
            plt.show()

        ret_set = graph_utils.spectral_improve_cut(vertices[:best_index+1], G)

        return ret_set
    

    