"""Tests for birth_death_process()."""

import networkx as nx
import numpy as np
import pytest

from cassiopeia.mixins import TreeSimulatorError
from cassiopeia.simulator import birth_death_process


def _tree_stats(tdata) -> tuple[list[float], int, bool]:
    """Extract (leaf times, num leaves, all internal degrees in {0,2}) from a TreeData."""
    tree = tdata.obst["tree"]
    times = []
    out_degrees = []
    for node in tree.nodes:
        if tree.out_degree(node) == 0:
            times.append(tree.nodes[node]["time"])
        else:
            out_degrees.append(tree.out_degree(node))
    # Exclude root from degree check (root is first in topological order)
    root = next(n for n in tree if tree.in_degree(n) == 0)
    degrees_no_root = [tree.out_degree(n) for n in tree if n != root and tree.out_degree(n) != 0]
    correct_degrees = all(d == 2 for d in degrees_no_root)
    return times, len(times), correct_degrees


# --- Validation errors ---

def test_bad_waiting_distributions():
    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: -1, 1, experiment_time=1)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 0, 1, num_extant=4)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: -1, num_extant=1)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: 0, experiment_time=1)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(
            lambda _: 1, 1, lambda: 0,
            mutation_distribution=lambda: -1,
            fitness_distribution=lambda: 1,
            experiment_time=1,
        )


def test_bad_stopping_conditions():
    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: 2)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: 2, num_extant=0.5)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: 2, num_extant=-1)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: 2, num_extant=0)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: 2, experiment_time=-1)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 1, 1, lambda: 2, experiment_time=0)


def test_dead_at_start():
    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 2, 1, lambda: 1, num_extant=4)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(lambda _: 2, 1, lambda: 1, experiment_time=4)


def test_dead_before_end():
    birth_wd = lambda scale: np.random.exponential(scale)
    death_wd = lambda: np.random.exponential(0.6)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(birth_wd, 0.5, death_wd, num_extant=8, random_seed=5)

    with pytest.raises(TreeSimulatorError):
        birth_death_process(birth_wd, 0.5, death_wd, experiment_time=2, random_seed=5)


# --- Correct simulation results ---

def test_single_lineage():
    tdata = birth_death_process(lambda _: 1, 1, num_extant=1)
    times, n_leaves, _ = _tree_stats(tdata)
    tree = tdata.obst["tree"]
    root = next(n for n in tree if tree.in_degree(n) == 0)
    leaf = next(n for n in tree if tree.out_degree(n) == 0)
    assert n_leaves == 1
    branch_len = tree.nodes[leaf]["time"] - tree.nodes[root]["time"]
    assert branch_len == 1.0
    assert times == [1]

    tdata = birth_death_process(lambda _: 1, 1, experiment_time=1)
    times, n_leaves, _ = _tree_stats(tdata)
    tree = tdata.obst["tree"]
    root = next(n for n in tree if tree.in_degree(n) == 0)
    leaf = next(n for n in tree if tree.out_degree(n) == 0)
    assert n_leaves == 1
    assert tree.nodes[leaf]["time"] - tree.nodes[root]["time"] == 1.0
    assert times == [1]


def test_constant_yule():
    tdata = birth_death_process(lambda _: 1, 1, num_extant=32)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(t == 6 for t in times)
    assert n_leaves == 32
    assert correct

    tdata = birth_death_process(lambda _: 1, 1, experiment_time=6)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(t == 6 for t in times)
    assert n_leaves == 32
    assert correct


def test_nonconstant_yule():
    birth_wd = lambda scale: np.random.exponential(scale)

    tdata = birth_death_process(birth_wd, 1, num_extant=16, random_seed=54)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, times[0]) for t in times)
    assert n_leaves == 16
    assert correct
    tree = tdata.obst["tree"]
    assert max(int(n) for n in tree.nodes if n != "root") == 31

    tdata = birth_death_process(birth_wd, 1, experiment_time=2, random_seed=54)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(t == 2 for t in times)
    assert correct


def test_nonconstant_birth_death_process():
    birth_wd = lambda scale: np.random.exponential(scale)
    death_wd = lambda: np.random.exponential(1.5)

    tdata = birth_death_process(birth_wd, 0.5, death_wd, num_extant=8, random_seed=1234)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, times[0]) for t in times)
    assert n_leaves == 8
    assert correct

    tdata = birth_death_process(birth_wd, 0.5, death_wd, experiment_time=2, random_seed=1234)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, 2) for t in times)
    assert correct


def test_nonconstant_birth_death_process_no_unifurcation_collapsing():
    birth_wd = lambda scale: np.random.exponential(scale)
    death_wd = lambda: np.random.exponential(1.5)

    tdata = birth_death_process(
        birth_wd, 0.5, death_wd, num_extant=8, collapse_unifurcations=False, random_seed=12
    )
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, times[0]) for t in times)
    assert n_leaves == 8
    assert not correct

    tdata = birth_death_process(
        birth_wd, 0.5, death_wd,
        experiment_time=1.3, collapse_unifurcations=False, random_seed=12
    )
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, 1.3) for t in times)
    assert not correct


def test_nonconstant_birth_death_process_both_stopping_conditions():
    birth_wd = lambda scale: np.random.exponential(scale)
    death_wd = lambda: np.random.exponential(1.5)

    tdata = birth_death_process(birth_wd, 0.5, death_wd, num_extant=8, experiment_time=2, random_seed=17)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, times[0]) for t in times)
    assert all(t > 1 for t in times)
    assert n_leaves == 8
    assert correct

    tdata = birth_death_process(birth_wd, 0.5, death_wd, num_extant=8, experiment_time=1, random_seed=17)
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, 1) for t in times)
    assert n_leaves == 3
    assert correct


def test_nonconstant_yule_with_predictable_fitness():
    multiplier = 0.98 ** 2  # 2 mutations per division, fitness_base=0.98

    def check_fitness(tree: nx.DiGraph):
        """Check birth_scale = 0.5 * multiplier^d (internal) or ^(d-1) (leaf)
        where d is edges from root. The root's single child is preserved (not
        collapsed), so each edge corresponds to exactly one division."""
        root = next(n for n in tree if tree.in_degree(n) == 0)
        depth = {root: 0}
        for u, v in nx.dfs_edges(tree, source=root):
            depth[v] = depth[u] + 1
        leaves = {n for n in tree if tree.out_degree(n) == 0}
        for node in tree.nodes:
            d = depth[node]
            if node in leaves:
                expected = 0.5 * multiplier ** (d - 1)
            else:
                expected = 0.5 * multiplier ** d
            assert np.isclose(tree.nodes[node]["birth_scale"], expected), (
                f"node={node} depth={d} leaf={node in leaves} "
                f"actual={tree.nodes[node]['birth_scale']:.6f} expected={expected:.6f}"
            )

    birth_wd = lambda scale: np.random.exponential(scale)

    tdata = birth_death_process(
        birth_wd, 0.5,
        mutation_distribution=lambda: 2,
        fitness_distribution=lambda: 1,
        fitness_base=0.98,
        num_extant=8,
        random_seed=1234,
    )
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, times[0]) for t in times)
    assert n_leaves == 8
    assert correct
    check_fitness(tdata.obst["tree"])

    tdata = birth_death_process(
        birth_wd, 0.5,
        mutation_distribution=lambda: 2,
        fitness_distribution=lambda: 1,
        fitness_base=0.98,
        experiment_time=0.6,
        random_seed=1234,
    )
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, 0.6) for t in times)
    assert correct
    check_fitness(tdata.obst["tree"])


def test_nonconstant_birth_death_process_with_variable_fitness():
    birth_wd = lambda scale: np.random.exponential(scale)
    death_wd = lambda: np.random.exponential(0.6)
    mut_dist = lambda: 1 if np.random.uniform() < 0.2 else 0
    fit_dist = lambda: np.random.uniform(-1, 1)

    tdata = birth_death_process(
        birth_wd, 0.5, death_wd, mut_dist, fit_dist, 1.5, num_extant=8, random_seed=12364
    )
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, times[0]) for t in times)
    assert n_leaves == 8
    assert correct

    tdata = birth_death_process(
        birth_wd, 0.5, death_wd, mut_dist, fit_dist, 1.5,
        experiment_time=3, random_seed=12364
    )
    times, n_leaves, correct = _tree_stats(tdata)
    assert all(np.isclose(t, 3) for t in times)
    assert correct


def test_no_initial_birth_scale():
    topology = nx.DiGraph()
    topology.add_edges_from([
        ("0", "1"), ("0", "2"),
        ("1", "3"), ("1", "4"),
        ("2", "5"), ("2", "6"),
    ])
    birth_wd = lambda scale: np.random.exponential(scale)

    tdata = birth_death_process(birth_wd, 1, num_extant=16, random_seed=54, initial_tree=topology)
    tree = tdata.obst["tree"]
    assert len([n for n in tree if tree.out_degree(n) == 0]) == 16

    initial_leaves = {"3", "4", "5", "6"}
    for leaf in initial_leaves:
        if leaf in tree.nodes:
            assert tree.nodes[leaf]["birth_scale"] == 1


def test_birth_scale_chaining():
    birth_wd = lambda scale: np.random.exponential(scale)

    tdata1 = birth_death_process(birth_wd, 1, num_extant=16, random_seed=54)
    initial_graph = tdata1.obst["tree"]
    initial_leaves = {n for n in initial_graph if initial_graph.out_degree(n) == 0}

    tdata2 = birth_death_process(birth_wd, 1, num_extant=100, random_seed=54, initial_tree=initial_graph)
    tree2 = tdata2.obst["tree"]

    assert len([n for n in tree2 if tree2.out_degree(n) == 0]) == 100

    for leaf in initial_leaves:
        orig_scale = initial_graph.nodes[leaf]["birth_scale"]
        final_scale = tree2.nodes[leaf]["birth_scale"]
        assert orig_scale == final_scale


if __name__ == "__main__":
    pytest.main([__file__])
