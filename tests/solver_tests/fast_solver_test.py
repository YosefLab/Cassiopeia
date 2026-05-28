"""Tests for the functional nj() and upgma() API and the Cython backend."""

import itertools
import unittest
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

import cassiopeia as cas
from cassiopeia.solver import solver_utilities


def find_triplet_structure(triplet, T):
    a, b, c = triplet[0], triplet[1], triplet[2]
    a_anc = set(nx.ancestors(T, a))
    b_anc = set(nx.ancestors(T, b))
    c_anc = set(nx.ancestors(T, c))
    ab = len(a_anc & b_anc)
    ac = len(a_anc & c_anc)
    bc = len(b_anc & c_anc)
    if ab > bc and ab > ac:
        return "ab"
    if ac > bc and ac > ab:
        return "ac"
    if bc > ab and bc > ac:
        return "bc"
    return "-"


CM = pd.DataFrame.from_dict(
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
    columns=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"],
)

SMALL_CM = pd.DataFrame.from_dict(
    {
        "a": [1, 1, 0],
        "b": [1, 2, 0],
        "c": [1, 2, 1],
        "d": [2, 0, 0],
        "e": [2, 0, 2],
    },
    orient="index",
    columns=["x1", "x2", "x3"],
)


class TestFastSolverCassiopeiaTree(unittest.TestCase):
    def setUp(self):
        self.tree = cas.data.CassiopeiaTree(character_matrix=CM)
        self.small_tree = cas.data.CassiopeiaTree(character_matrix=SMALL_CM)
        self.duplicate_cm = pd.DataFrame.from_dict(
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
        self.duplicate_tree = cas.data.CassiopeiaTree(character_matrix=self.duplicate_cm)

    def test_nj_cassiopeia_tree(self):
        cas.solver.nj(self.tree, root="outgroup")

        leaves = self.tree.leaves
        self.assertEqual(len(leaves), len(CM))

        topology = self.tree.get_tree_topology()
        triplets = list(itertools.combinations(["a", "b", "c", "d"], 3))
        structures = [find_triplet_structure(t, topology) for t in triplets]
        self.assertGreater(sum(s != "-" for s in structures), 0)

    def test_nj_cassiopeia_tree_small(self):
        cas.solver.nj(self.small_tree, root="outgroup")
        self.assertEqual(len(self.small_tree.leaves), len(SMALL_CM))
        self.assertIsNotNone(self.small_tree.get_tree_topology())

    def test_upgma_cassiopeia_tree(self):
        cas.solver.upgma(self.tree)

        leaves = self.tree.leaves
        self.assertEqual(len(leaves), len(CM))

        topology = self.tree.get_tree_topology()
        triplets = list(itertools.combinations(["a", "b", "c", "d"], 3))
        structures = [find_triplet_structure(t, topology) for t in triplets]
        self.assertGreater(sum(s != "-" for s in structures), 0)

    def test_nj_and_upgma_agree_on_known_groups(self):
        nj_tree = cas.data.CassiopeiaTree(character_matrix=CM)
        upgma_tree = cas.data.CassiopeiaTree(character_matrix=CM)
        cas.solver.nj(nj_tree, root="outgroup")
        cas.solver.upgma(upgma_tree)

        nj_topo = nj_tree.get_tree_topology()
        upgma_topo = upgma_tree.get_tree_topology()

        for triplet in itertools.combinations(["a", "b", "c", "d"], 3):
            nj_s = find_triplet_structure(triplet, nj_topo)
            upgma_s = find_triplet_structure(triplet, upgma_topo)
            if nj_s != "-" and upgma_s != "-":
                self.assertEqual(nj_s, upgma_s)

    def test_collapse_mutationless_edges(self):
        cas.solver.nj(self.small_tree, root="outgroup")
        solver_utilities.collapse_mutationless_edges(self.small_tree)
        self.assertEqual(len(self.small_tree.leaves), len(SMALL_CM))

    def test_duplicate_sample(self):
        cas.solver.nj(self.duplicate_tree, root="outgroup")
        self.assertEqual(len(self.duplicate_tree.leaves), len(self.duplicate_cm))

    def test_nj_no_root_raises(self):
        tree = cas.data.CassiopeiaTree(character_matrix=SMALL_CM)
        with self.assertRaises(cas.mixins.DistanceSolverError):
            cas.solver.nj(tree)  # root=None, no root_sample_name set

    def test_nj_unknown_root_raises(self):
        tree = cas.data.CassiopeiaTree(character_matrix=SMALL_CM)
        with self.assertRaises(ValueError):
            cas.solver.nj(tree, root="nonexistent_procedure")

    def test_nj_with_explicit_root(self):
        tree = cas.data.CassiopeiaTree(character_matrix=SMALL_CM, root_sample_name="a")
        cas.solver.nj(tree)  # root=None, uses root_sample_name="a"
        self.assertIsNotNone(tree.get_tree_topology())


class TestFastSolverTreeData(unittest.TestCase):
    def setUp(self):
        samples = list("abcde")
        dist = pd.DataFrame(
            [
                [0.0, 0.1, 0.8, 0.8, 0.8],
                [0.1, 0.0, 0.8, 0.8, 0.8],
                [0.8, 0.8, 0.0, 0.1, 0.7],
                [0.8, 0.8, 0.1, 0.0, 0.7],
                [0.8, 0.8, 0.7, 0.7, 0.0],
            ],
            index=samples,
            columns=samples,
        )
        obs = pd.DataFrame(index=samples)
        self.tdata = td.TreeData(obs=obs)
        self.tdata.obsp["distances"] = dist.to_numpy()
        self.tdata.obsm["characters"] = SMALL_CM

    def test_nj_treedata_dist_key(self):
        cas.solver.nj(self.tdata, dist_key="distances", tree_key="nj")
        self.assertIn("nj", self.tdata.obst)
        tree = self.tdata.obst["nj"]
        self.assertIsInstance(tree, nx.DiGraph)
        self.assertGreater(len(tree.nodes), 0)

    def test_upgma_treedata_dist_key(self):
        cas.solver.upgma(self.tdata, dist_key="distances", tree_key="upgma")
        self.assertIn("upgma", self.tdata.obst)
        self.assertIsInstance(self.tdata.obst["upgma"], nx.DiGraph)

    def test_nj_treedata_groups_cluster_correctly(self):
        # Named outgroup: dist_key is used and "e" is inserted as outgroup leaf
        cas.solver.nj(self.tdata, dist_key="distances", root="outgroup", outgroup="e", tree_key="nj")
        tree = self.tdata.obst["nj"]
        structure = find_triplet_structure(("a", "b", "c"), tree)
        self.assertEqual(structure, "ab")

    def test_upgma_treedata_groups_cluster_correctly(self):
        cas.solver.upgma(self.tdata, dist_key="distances", tree_key="upgma")
        tree = self.tdata.obst["upgma"]
        ab_anc = set(nx.ancestors(tree, "a")) & set(nx.ancestors(tree, "b"))
        ac_anc = set(nx.ancestors(tree, "a")) & set(nx.ancestors(tree, "c"))
        self.assertGreater(len(ab_anc), len(ac_anc))

    def test_nj_treedata_no_dist_key(self):
        cas.solver.nj(self.tdata, root="outgroup", tree_key="nj_chars")
        self.assertIn("nj_chars", self.tdata.obst)
        self.assertIsInstance(self.tdata.obst["nj_chars"], nx.DiGraph)

    def test_nj_treedata_default_tree_key(self):
        cas.solver.nj(self.tdata, dist_key="distances")
        self.assertIn("nj", self.tdata.obst)

    def test_upgma_treedata_default_tree_key(self):
        cas.solver.upgma(self.tdata, dist_key="distances")
        self.assertIn("upgma", self.tdata.obst)


class TestBackwardCompatWrappers(unittest.TestCase):
    def setUp(self):
        self.tree = cas.data.CassiopeiaTree(character_matrix=SMALL_CM)

    def test_deprecated_implementation_kwarg_nj(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = cas.solver.NeighborJoiningSolver(
                add_root=True, implementation="ccphylo_dnj"
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))

        solver.solve(self.tree)
        self.assertIsNotNone(self.tree.get_tree_topology())

    def test_deprecated_implementation_kwarg_upgma(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = cas.solver.UPGMASolver(implementation="ccphylo_upgma")
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))

    def test_nj_solver_fast_false_raises(self):
        with self.assertRaises(NotImplementedError):
            cas.solver.NeighborJoiningSolver(fast=False)

    def test_upgma_solver_fast_false_raises(self):
        with self.assertRaises(NotImplementedError):
            cas.solver.UPGMASolver(fast=False)

    def test_nj_solver_wrapper_solves(self):
        solver = cas.solver.NeighborJoiningSolver(add_root=True)
        solver.solve(self.tree)
        self.assertIsNotNone(self.tree.get_tree_topology())
        self.assertEqual(len(self.tree.leaves), len(SMALL_CM))

    def test_upgma_solver_wrapper_solves(self):
        solver = cas.solver.UPGMASolver()
        tree = cas.data.CassiopeiaTree(character_matrix=SMALL_CM)
        solver.solve(tree)
        self.assertIsNotNone(tree.get_tree_topology())
        self.assertEqual(len(tree.leaves), len(SMALL_CM))

    def test_collapse_mutationless_edges_via_wrapper(self):
        solver = cas.solver.NeighborJoiningSolver(add_root=True)
        solver.solve(self.tree, collapse_mutationless_edges=True)
        self.assertIsNotNone(self.tree.get_tree_topology())

    def test_collapse_mutationless_edges_treedata_raises(self):
        obs = pd.DataFrame(index=list("ab"))
        tdata = td.TreeData(obs=obs)
        with self.assertRaises(NotImplementedError):
            solver_utilities.collapse_mutationless_edges(tdata)


class TestRootingModule(unittest.TestCase):
    def test_register_custom_procedure(self):
        @cas.solver.rooting.register("test_first_obs")
        def _first_obs(graph, **kwargs):
            root = list(graph.nodes)[0]
            rooted = nx.DiGraph()
            for e in nx.dfs_edges(graph, source=root):
                rooted.add_edge(e[0], e[1])
            return rooted

        self.assertIn("test_first_obs", cas.solver.rooting._PROCEDURES)

    def test_unknown_procedure_raises(self):
        import cassiopeia as cas
        tree = cas.data.CassiopeiaTree(character_matrix=SMALL_CM)
        with self.assertRaises(ValueError):
            cas.solver.nj(tree, root="does_not_exist")

    def test_outgroup_available(self):
        self.assertIn("outgroup", cas.solver.rooting._PROCEDURES)


if __name__ == "__main__":
    unittest.main()
