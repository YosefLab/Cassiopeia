"""
Test the ccphylo solver implementations against the standard NJ and UPGMA
"""
import unittest
from typing import Dict, Optional
from unittest import mock

import os

import configparser
import itertools
import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas


def find_triplet_structure(triplet, T):
    a, b, c = triplet[0], triplet[1], triplet[2]
    a_ancestors = [node for node in nx.ancestors(T, a)]
    b_ancestors = [node for node in nx.ancestors(T, b)]
    c_ancestors = [node for node in nx.ancestors(T, c)]
    ab_common = len(set(a_ancestors) & set(b_ancestors))
    ac_common = len(set(a_ancestors) & set(c_ancestors))
    bc_common = len(set(b_ancestors) & set(c_ancestors))
    structure = "-"
    if ab_common > bc_common and ab_common > ac_common:
        structure = "ab"
    elif ac_common > bc_common and ac_common > ab_common:
        structure = "ac"
    elif bc_common > ab_common and bc_common > ac_common:
        structure = "bc"
    return structure

# specify dissimilarity function for solvers to use
def delta_fn(
    x: np.array,
    y: np.array,
    missing_state: int,
    priors: Optional[Dict[int, Dict[int, float]]],
):
    d = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            d += 1
    return d

# only run test if ccphylo_path is specified in config.ini
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__),
                         "..","..","cassiopeia","config.ini"))
CCPHYLO_CONFIGURED = (config.get("Paths","ccphylo_path") != 
                      "/path/to/ccphylo/ccphylo")
print(CCPHYLO_CONFIGURED)


class TestCCPhyloSolver(unittest.TestCase):
    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def setUp(self):

        # --------------------- General NJ ---------------------
        cm = pd.DataFrame.from_dict(
            {
                "a": [0, 1, 2, 1, 0, 0, 2, 0, 0, 0],
                "b": [1, 1, 2, 1, 0, 0, 2, 0, 0, 0],
                "c": [2, 2, 2, 1, 0, 0, 2, 0, 0, 0],
                "d": [1, 1, 1, 1, 0, 0, 2, 0, 0, 0],
                "e": [0, 0, 0, 0, 1, 2, 1, 0, 2, 0],
                "f": [0, 0, 0, 0, 2, 2, 1, 0, 2, 0],
                "g": [0, 2, 0, 0, 1, 1, 1, 0, 2, 0],
                "h": [0, 2, 0, 0, 1, 0, 0, 1, 2, 1],
                "i": [1, 2, 0, 0, 1, 0, 0, 2, 2, 1],
                "j": [1, 2, 0, 0, 1, 0, 0, 1, 1, 1],
            },
            orient="index",
            columns=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", 
                        "x9", "x10"],
        )

        self.cm = cm
        self.basic_tree = cas.data.CassiopeiaTree(
            character_matrix=cm
        )

        self.nj_solver = cas.solver.NeighborJoiningSolver(
            add_root=True,fast=False)
        self.ccphylo_nj_solver = cas.solver.NeighborJoiningSolver(
            add_root=True,fast=True,implementation="ccphylo_nj")
        self.ccphylo_dnj_solver = cas.solver.NeighborJoiningSolver(
            add_root=True,fast = True, implementation="ccphylo_dnj")
        self.ccphylo_hnj_solver = cas.solver.NeighborJoiningSolver(
            add_root=True, fast = True, implementation="ccphylo_hnj")
        
        self.ccphylo_upgma_solver = cas.solver.UPGMASolver(fast=True)
        self.upgma_solver = cas.solver.UPGMASolver(fast=False)
        

        # ------------- CM with Duplictes -----------------------
        duplicates_cm = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
                "f": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        self.duplicate_tree = cas.data.CassiopeiaTree(
            character_matrix=duplicates_cm
        )

    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def test_ccphylo_invalid_input(self):
        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            nothing_solver = cas.solver.NeighborJoiningSolver(fast = True,
                implementation="invalid")
       
        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            nothing_solver = cas.solver.UPGMASolver(fast = True,
                implementation="invalid")

    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def test_ccphylo_nj_solver(self):
        # NJ Solver
        nj_tree = self.basic_tree.copy()
        self.nj_solver.solve(nj_tree)

        # CCPhylo Fast NJ Solver
        ccphylo_nj_tree = self.basic_tree.copy()
        self.ccphylo_nj_solver.solve(ccphylo_nj_tree)

        # test for expected number of edges
        self.assertEqual(len(nj_tree.edges), len(ccphylo_nj_tree.edges))

        triplets = itertools.combinations(["a", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, 
                                    nj_tree.get_tree_topology())
            observed_triplet = find_triplet_structure(triplet, 
                                    ccphylo_nj_tree.get_tree_topology())
            self.assertEqual(expected_triplet, observed_triplet)

    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def test_ccphylo_dnj_solver(self):
        # NJ Solver
        nj_tree = self.basic_tree.copy()
        self.nj_solver.solve(nj_tree)

        # CCPhylo DNJ Solver
        dnj_tree = self.basic_tree.copy()
        self.ccphylo_dnj_solver.solve(dnj_tree)

        # test for expected number of edges
        self.assertEqual(len(nj_tree.edges), len(dnj_tree.edges))

        triplets = itertools.combinations(["a", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, 
                                    nj_tree.get_tree_topology())
            observed_triplet = find_triplet_structure(triplet, 
                                    dnj_tree.get_tree_topology())
            self.assertEqual(expected_triplet, observed_triplet)

    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def test_ccphylo_hnj_solver(self):
        # NJ Solver
        nj_tree = self.basic_tree.copy()
        self.nj_solver.solve(nj_tree)

        # CCPhylo HNJ Solver
        hnj_tree = self.basic_tree.copy()
        self.ccphylo_hnj_solver.solve(hnj_tree)

        # test for expected number of edges
        self.assertEqual(len(nj_tree.edges), len(hnj_tree.edges))


        triplets = itertools.combinations(["a", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, 
                                    nj_tree.get_tree_topology())
            observed_triplet = find_triplet_structure(triplet, 
                                    hnj_tree.get_tree_topology())
            self.assertEqual(expected_triplet, observed_triplet)

    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def test_ccphylo_upgma_solver(self):
        # UPGMA Solver
        upgma_tree = self.basic_tree.copy()
        self.upgma_solver.solve(upgma_tree)

        # CCPhylo Fast UPGMA Solver
        ccphylo_upgma_tree = self.basic_tree.copy()
        self.ccphylo_upgma_solver.solve(ccphylo_upgma_tree)

        # test for expected number of edges
        self.assertEqual(len(upgma_tree.edges), len(ccphylo_upgma_tree.edges))


        triplets = itertools.combinations(["a", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, 
                                    upgma_tree.get_tree_topology())
            observed_triplet = find_triplet_structure(triplet, 
                                    ccphylo_upgma_tree.get_tree_topology())
            self.assertEqual(expected_triplet, observed_triplet)

    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def test_collapse_mutationless_edges_ccphylo(self):
        # NJ Solver
        nj_tree = self.basic_tree.copy()
        self.nj_solver.solve(nj_tree, collapse_mutationless_edges=True)

        # Fast NJ Solver
        ccphylo_nj_tree = self.basic_tree.copy()
        self.ccphylo_nj_solver.solve(ccphylo_nj_tree, 
                                        collapse_mutationless_edges=True)

        # test for expected number of edges
        self.assertEqual(len(nj_tree.edges), len(ccphylo_nj_tree.edges))

        triplets = itertools.combinations(["a", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, 
                                    nj_tree.get_tree_topology())
            observed_triplet = find_triplet_structure(triplet, 
                                    ccphylo_nj_tree.get_tree_topology())
            self.assertEqual(expected_triplet, observed_triplet)

    @unittest.skipUnless(CCPHYLO_CONFIGURED, "CCPhylo not configured.")
    def test_duplicate_sample_ccphylo(self):
        # NJ Solver
        nj_tree = self.duplicate_tree.copy()
        self.nj_solver.solve(nj_tree)

        # Fast NJ Solver
        ccphylo_nj_tree = self.duplicate_tree.copy()
        self.ccphylo_nj_solver.solve(ccphylo_nj_tree)

        # test for expected number of edges
        self.assertEqual(len(nj_tree.edges), len(ccphylo_nj_tree.edges))

        triplets = itertools.combinations(["a", "b", "c", "d", "e", "f"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, 
                                    nj_tree.get_tree_topology())
            observed_triplet = find_triplet_structure(triplet, 
                                    ccphylo_nj_tree.get_tree_topology())
            self.assertEqual(expected_triplet, observed_triplet)

if __name__ == "__main__":
    unittest.main()