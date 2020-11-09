"""
"""
import itertools
import networkx as nx
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple

from cassiopeia.solver import GreedySolver
from cassiopeia.solver import graph_utilities as g_utils


class MaxCutSolver(GreedySolver.GreedySolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: str,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        sdimension: Optional[int] = 3,
        iterations: Optional[int] = 50,
        weights: Optional[Dict] = None,
    ):

        super().__init__(character_matrix, missing_char, meta_data, priors)
        self.sdimension = sdimension
        self.iterations = iterations
        self.weights = weights

    def perform_split(
        self, samples: List[int] = None
    ) -> Tuple[List[int], List[int]]:
        """"""
        G = g_utils.construct_connectivity_graph(
            self.prune_cm, samples, self.missing_char, w=self.weights
        )

        d = self.sdimension + 1
        emb = {}
        for i in G.nodes():
            x = np.random.normal(size=d)
            x = x / np.linalg.norm(x)
            emb[i] = x

        for k in range(self.iterations):
            new_emb = {}
            for i in G.nodes:
                cm = np.zeros(d, dtype=float)
                for j in G.neighbors(i):
                    cm -= (
                        G[i][j]["weight"]
                        * np.linalg.norm(emb[i] - emb[j])
                        * emb[j]
                    )
                cm = cm / np.linalg.norm(cm)
                new_emb[i] = cm
            emb = new_emb

        return_set = set()
        best_score = 0
        for k in range(3 * d):
            b = np.random.normal(size=d)
            b = b / np.linalg.norm(b)
            S = set()
            for i in G.nodes():
                if np.dot(emb[i], b) > 0:
                    S.add(i)
            this_score = self.evaluate_cut(S, G)
            if this_score > best_score:
                return_set = S
                best_score = this_score

        improved_S = g_utils.max_cut_improve_cut(G, return_set)

        rest = set(samples) - improved_S

        return list(improved_S), list(rest)

    def evaluate_cut(self, S, G, B=None):
        cut_score = 0
        total_good = 0
        total_bad = 0
        for e in G.edges():
            u = e[0]
            v = e[1]
            w_uv = G[u][v]["weight"]
            total_good += float(w_uv)
            if g_utils.check_if_cut(u, v, S):
                cut_score += float(w_uv)

        if B:
            for e in B.edges():
                u = e[0]
                v = e[1]
                w_uv = B[u][v]["weight"]
                total_bad += float(w_uv)
                if g_utils.check_if_cut(u, v, S):
                    cut_score -= float(w_uv)

        return cut_score
