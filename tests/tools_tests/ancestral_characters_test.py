"""Tests for tools.ancestral_characters, collapse_edges, count_edge_mutations (TreeData)."""

import importlib
import itertools

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas
from cassiopeia.mixins.errors import CassiopeiaError


def make_tdata():
    g = nx.DiGraph()
    for u, v in [("4", "0"), ("4", "1"), ("5", "2"), ("5", "3"), ("6", "4"), ("6", "5")]:
        g.add_edge(u, v)
    cm = pd.DataFrame.from_dict(
        {
            "0": [1, 0, 3, 4, 5],
            "1": [1, 0, 3, 3, -1],
            "2": [1, 2, 3, 0, -1],
            "3": [1, 0, 3, 0, -1],
        },
        orient="index",
        columns=["a", "b", "c", "d", "e"],
    )
    return td.TreeData(
        obs=pd.DataFrame(index=["0", "1", "2", "3"]),
        obst={"tree": g},
        obsm={"characters": cm},
        uns={"missing_state": -1, "unmodified_state": 0},
    )


def leaves(g):
    return [n for n in g.nodes if g.out_degree(n) == 0]


def test_ancestral_characters_infers_internal_states():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    g = tdata.obst["tree"]
    # internal node 6 is the LCA of all four leaves under Camin-Sokal
    assert g.nodes["6"]["characters"] == [1, 0, 3, 0, 5]
    # leaves are seeded under the same key
    assert g.nodes["0"]["characters"] == [1, 0, 3, 4, 5]


def test_ancestral_characters_copy_returns_new():
    tdata = make_tdata()
    out = cas.tl.ancestral_characters(tdata, tree_key="tree", copy=True)
    assert isinstance(out, td.TreeData)
    # original untouched
    assert "characters" not in tdata.obst["tree"].nodes["6"]


def test_ancestral_characters_key_added():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree", key_added="ancestral")
    g = tdata.obst["tree"]
    assert g.nodes["6"]["ancestral"] == [1, 0, 3, 0, 5]
    assert g.nodes["0"]["ancestral"] == [1, 0, 3, 4, 5]
    # the default characters attribute is not written when key_added is given
    assert "characters" not in g.nodes["6"]


def test_ancestral_characters_unmodified_state():
    g = nx.DiGraph()
    g.add_edge("p", "0")
    g.add_edge("p", "1")
    cm = pd.DataFrame.from_dict({"0": [1], "1": [2]}, orient="index", columns=["a"])
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["0", "1"]),
        obst={"tree": g},
        obsm={"characters": cm},
        uns={"missing_state": -1},
    )
    cas.tl.ancestral_characters(tdata, tree_key="tree", unmodified_state=9)
    # leaves disagree at character a, so the LCA takes the unmodified state
    assert tdata.obst["tree"].nodes["p"]["characters"] == [9]


def leaf_tdata(edges, leaf_states: dict, **uns) -> td.TreeData:
    g = nx.DiGraph(edges)
    leaves = [n for n in g if g.out_degree(n) == 0]
    return td.TreeData(
        obs=pd.DataFrame(index=leaves),
        obst={"tree": g},
        obsm={"characters": pd.DataFrame([leaf_states[n] for n in leaves], index=leaves)},
        uns={"missing_state": -1, "unmodified_state": 0, **uns},
    )


def states_of(tdata: td.TreeData) -> dict:
    return dict(tdata.obst["tree"].nodes(data="characters"))


@pytest.mark.filterwarnings("ignore:Parameter")
def test_camin_sokal_writes_states_of_string_matrix():
    # Without uns, the fallback missing/unmodified values matching the matrix are used.
    edges = [("r", "a"), ("r", "b"), ("a", "1"), ("a", "2"), ("b", "3"), ("b", "4")]
    tdata = leaf_tdata(edges, {"1": ["1", "-"], "2": ["2", "-"], "3": ["*", "-"], "4": ["*", "-"]})
    tdata.uns.clear()
    cas.tl.ancestral_characters(tdata)
    states = states_of(tdata)
    assert states["a"] == ["*", "-"]
    assert states["r"] == ["*", "-"]


# Conservative maximum parsimony. The first two trees are ConvexML's tests; there the
# root is fixed to unmodified, here it sits below an implicit unedited founder.
BINARY_16 = [("0", "1")] + [(str(i), str(2 * i + j)) for i in range(1, 8) for j in (0, 1)]
LEAVES_16 = {
    "8": [99, 99, 0, 0, 99, 99, 99],
    "9": [-1, -1, 11, -1, -1, -1, -1],
    "10": [99, -1, 22, -1, 99, 99, 99],
    "11": [-1, -1, 33, 99, -1, -1, 11],
    "12": [-1, 99, 44, -1, 99, 22, 99],
    "13": [-1, -1, -1, 99, -1, -1, -1],
    "14": [-1, -1, 55, 99, -1, -1, 11],
    "15": [-1, -1, -1, -1, -1, -1, 22],
}
EXPECTED_16 = {
    # ConvexML has the root at [0] * 7; its edits are carried by its only child.
    "0": [-1, -1, 0, 0, -1, 0, 0],
    "1": [-1, 99, 0, 0, 99, 0, 0],
    "2": [99, 99, 0, 0, 99, 99, 0],
    "4": [99, 99, 0, 0, 99, 99, -1],
    "5": [99, -1, 0, -1, 99, 99, 0],
    "3": [-1, 99, 0, 99, 99, -1, 0],
    "6": [-1, 99, -1, 99, 99, -1, -1],
    "7": [-1, -1, -1, 99, -1, -1, 0],
}


def test_conservative_rooted_binary_tree():
    tdata = leaf_tdata(BINARY_16, LEAVES_16)
    cas.tl.ancestral_characters(tdata, method="conservative")
    states = states_of(tdata)
    for node, expected in EXPECTED_16.items():
        assert states[node] == expected, node
    for leaf, expected in LEAVES_16.items():
        assert states[leaf] == expected


def test_conservative_perfect_binary_tree():
    edges = [(str(i), str(2 * i + j)) for i in range(7) for j in (1, 2)]
    leaves = {"7": [0, 99], "8": [0, -1], "9": [0, 99], "10": [0, -1],
              "11": [0, 99], "12": [0, -1], "13": [0, -1], "14": [0, -1]}  # fmt: skip
    tdata = leaf_tdata(edges, leaves)
    cas.tl.ancestral_characters(tdata, method="conservative")
    states = states_of(tdata)
    # Both children of the root carry 99, so it was edited above the root and 2 and
    # 5 inherit it (ConvexML fixes the root to unmodified and leaves them missing).
    expected = {"0": [0, 99], "1": [0, 99], "3": [0, 99], "4": [0, 99], "2": [0, 99],
                "5": [0, 99], "6": [0, -1]}  # fmt: skip
    for node, state in expected.items():
        assert states[node] == state, node


def conservative_oracle(g: nx.DiGraph, leaf_states: dict) -> dict:
    """Conservative MP by brute force (ConvexML's test oracle).

    Sankoff over all optimal reconstructions under an irreversible model in which
    edited and unmodified states may become missing (heritable dropout). A node keeps
    a state only if it is the same in every optimal reconstruction. An unedited
    founder sits above the root.
    """
    root = next(n for n in g if g.in_degree(n) == 0)
    h = g.copy()
    h.add_edge("founder", root)

    def cost(x, y):
        if x == y:
            return 0
        if (x == 0 and y != 0) or (x >= 1 and y == -1):
            return 1
        return np.inf

    out = {n: [] for n in g}
    for i in range(len(next(iter(leaf_states.values())))):
        alphabet = sorted({-1, 0} | {s[i] for s in leaf_states.values()})
        dp = {}
        for node in nx.dfs_postorder_nodes(h, "founder"):
            if h.out_degree(node) == 0:
                dp[node] = {x: 0 if x == leaf_states[node][i] else np.inf for x in alphabet}
                continue
            dp[node] = {
                x: sum(min(dp[c][y] + cost(x, y) for y in alphabet) for c in h.successors(node))
                if node != "founder" or x == 0
                else np.inf
                for x in alphabet
            }
        # Collect every state a node takes in some optimal reconstruction.
        optimal = {n: set() for n in h}
        stack = [("founder", 0)]
        while stack:
            node, x = stack.pop()
            if x in optimal[node]:
                continue
            optimal[node].add(x)
            for c in h.successors(node):
                best = min(dp[c][y] + cost(x, y) for y in alphabet)
                stack.extend((c, y) for y in alphabet if dp[c][y] + cost(x, y) == best)
        for node in g:
            out[node].append(next(iter(optimal[node])) if len(optimal[node]) == 1 else -1)
    return out


def random_tree(rng: np.random.Generator, n_leaves: int) -> nx.DiGraph:
    """Random rooted tree with multifurcations and unifurcations."""
    g = nx.DiGraph()
    nodes = [f"l{i}" for i in range(n_leaves)]
    g.add_nodes_from(nodes)
    counter = itertools.count()
    while len(nodes) > 1:
        k = min(len(nodes), int(rng.integers(1, 5)))
        picked = rng.choice(len(nodes), size=k, replace=False)
        parent = f"n{next(counter)}"
        for j in picked:
            g.add_edge(parent, nodes[j])
        nodes = [n for j, n in enumerate(nodes) if j not in set(picked)] + [parent]
    return g


def random_leaf_states(rng, g, n_chars=6, missing=0.3) -> dict:
    leaves = [n for n in g if g.out_degree(n) == 0]
    states = rng.choice([0, 1, 2, 3], size=(len(leaves), n_chars), p=[0.4, 0.3, 0.2, 0.1])
    states[rng.random(states.shape) < missing] = -1
    return {leaf: states[i].tolist() for i, leaf in enumerate(leaves)}


@pytest.mark.parametrize("seed", range(20))
def test_conservative_matches_oracle(seed):
    rng = np.random.default_rng(seed)
    g = random_tree(rng, int(rng.integers(3, 14)))
    leaf_states = random_leaf_states(rng, g)
    tdata = leaf_tdata(g.edges, leaf_states)
    cas.tl.ancestral_characters(tdata, method="conservative")
    assert states_of(tdata) == conservative_oracle(g, leaf_states)


def test_conservative_and_sankoff_string_states():
    edges = [("r", "a"), ("r", "3"), ("a", "1"), ("a", "2")]
    leaves = {"1": ["1", "-", "*"], "2": ["1", "-", "*"], "3": ["*", "-", "2"]}
    tdata = leaf_tdata(edges, leaves, missing_state="-", unmodified_state="*")
    out = cas.tl.ancestral_characters(tdata, method="conservative", copy=True)
    assert states_of(out)["a"] == ["1", "-", "*"]
    assert states_of(out)["r"] == ["*", "-", "*"]
    out = cas.tl.ancestral_characters(tdata, method="sankoff", copy=True)
    assert states_of(out)["a"] == ["1", "-", "*"]


# Weighted Sankoff.
def clade_tdata(clade_states: list, outgroup=0) -> td.TreeData:
    """Root r with a leaf q and a clade p whose leaves have the given states."""
    leaves = {f"l{i}": [state] for i, state in enumerate(clade_states)}
    edges = [("r", "p"), ("r", "q")] + [("p", leaf) for leaf in leaves]
    return leaf_tdata(edges, {"q": [outgroup], **leaves})


def test_sankoff_keeps_edit_shared_despite_miscall():
    # One leaf of eight carries another edit: Camin-Sokal drops the shared edit
    # from p, Sankoff keeps it and re-edits that leaf.
    tdata = clade_tdata([1] * 7 + [2])
    out = cas.tl.ancestral_characters(tdata, copy=True)
    assert states_of(out)["p"] == [0]
    cas.tl.ancestral_characters(tdata, method="sankoff")
    assert states_of(tdata)["p"] == [1]
    assert states_of(tdata)["r"] == [0]
    assert states_of(tdata)["l7"] == [2]


def test_sankoff_weighting():
    # Four leaves with an edit and one unmodified leaf. Unweighted: one edit at p
    # and one reversion (1 + 2.5) beat four leaf edits. Weighted: the edit at p
    # costs log2(1 + 5), and four leaf edits are cheaper.
    tdata = clade_tdata([1, 1, 1, 1, 0])
    out = cas.tl.ancestral_characters(tdata, method="sankoff", weighted=False, copy=True)
    assert states_of(out)["p"] == [1]
    out = cas.tl.ancestral_characters(tdata, method="sankoff", copy=True)
    assert states_of(out)["p"] == [0]
    out = cas.tl.ancestral_characters(tdata, method="sankoff", weighted=False, reedit_cost=4,
                                      copy=True)  # fmt: skip
    assert states_of(out)["p"] == [0]


def test_sankoff_missing_subtrees_and_root():
    edges = [("r", "a"), ("r", "b"), ("a", "1"), ("a", "2"), ("b", "3"), ("b", "4")]
    leaves = {"1": [5, -1], "2": [5, -1], "3": [5, 0], "4": [5, -1]}
    tdata = leaf_tdata(edges, leaves)
    cas.tl.ancestral_characters(tdata, method="sankoff")
    states = states_of(tdata)
    # The edit shared by all leaves is placed above the root (founder -> root).
    assert states["r"] == [5, 0]
    # No leaf below a has data at the second character.
    assert states["a"] == [5, -1]
    assert states["b"] == [5, 0]
    assert states["1"] == [5, -1]


def sankoff_cost(g, states: dict, weighted: bool, reedit_cost: float, i: int) -> float:
    """Weighted Sankoff cost of an assignment, including the founder -> root edge."""
    root = next(n for n in g if g.in_degree(n) == 0)
    n_below = {}
    for node in nx.dfs_postorder_nodes(g, root):
        kids = list(g.successors(node))
        n_below[node] = sum(n_below[c] for c in kids) if kids else 1

    def change(x, y, node):
        if x == y:
            return 0.0
        w = np.log2(1 + n_below[node]) if weighted else 1.0
        return w * (1.0 if x == 0 else reedit_cost)

    # Nodes without data take their parent's state (cost 0).
    effective = {}
    total = 0.0
    for parent, node in [("founder", root), *nx.dfs_edges(g, root)]:
        x = 0 if parent == "founder" else effective[parent]
        y = states[node][i]
        effective[node] = x if y == -1 else y
        if y != -1:
            total += change(x, y, node)
    return total


def sankoff_optimum(g, leaf_states: dict, weighted: bool, reedit_cost: float, i: int) -> float:
    root = next(n for n in g if g.in_degree(n) == 0)
    alphabet = sorted({0} | {s[i] for s in leaf_states.values() if s[i] > 0})
    n_below, dp = {}, {}
    for node in nx.dfs_postorder_nodes(g, root):
        kids = list(g.successors(node))
        n_below[node] = sum(n_below[c] for c in kids) if kids else 1
        if not kids:
            s = leaf_states[node][i]
            dp[node] = {x: 0.0 if s in (-1, x) else np.inf for x in alphabet}
            continue
        dp[node] = {
            x: sum(
                min(dp[c][y] + (0 if x == y else (np.log2(1 + n_below[c]) if weighted else 1.0)
                                * (1.0 if x == 0 else reedit_cost)) for y in alphabet)
                for c in kids
            )
            for x in alphabet
        }  # fmt: skip
    w = np.log2(1 + n_below[root]) if weighted else 1.0
    return min(dp[root][y] + (0 if y == 0 else w) for y in alphabet)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("weighted, reedit_cost", [(True, 2.5), (False, 2.5), (True, 0.5)])
def test_sankoff_is_optimal(seed, weighted, reedit_cost):
    rng = np.random.default_rng(seed)
    g = random_tree(rng, int(rng.integers(3, 20)))
    leaf_states = random_leaf_states(rng, g)
    tdata = leaf_tdata(g.edges, leaf_states)
    cas.tl.ancestral_characters(tdata, method="sankoff", weighted=weighted, reedit_cost=reedit_cost)
    states = states_of(tdata)
    for i in range(len(next(iter(leaf_states.values())))):
        assert sankoff_cost(g, states, weighted, reedit_cost, i) == pytest.approx(
            sankoff_optimum(g, leaf_states, weighted, reedit_cost, i)
        )


@pytest.mark.parametrize("seed", range(5))
def test_sankoff_costs_matrix_matches_default(seed):
    rng = np.random.default_rng(seed)
    g = random_tree(rng, 15)
    leaf_states = random_leaf_states(rng, g)
    tdata = leaf_tdata(g.edges, leaf_states)
    matrix = np.full((4, 4), 2.5)
    matrix[0] = 1.0
    np.fill_diagonal(matrix, 0.0)
    costs = pd.DataFrame(matrix, index=[0, 1, 2, 3], columns=[0, 1, 2, 3])
    default = cas.tl.ancestral_characters(tdata, method="sankoff", copy=True)
    custom = cas.tl.ancestral_characters(tdata, method="sankoff", costs=costs, copy=True)
    assert states_of(default) == states_of(custom)
    # States are also matched to the cost labels as strings.
    custom = cas.tl.ancestral_characters(tdata, method="sankoff", costs=costs.rename(index=str, columns=str),
                                         copy=True)  # fmt: skip
    assert states_of(default) == states_of(custom)


def test_sankoff_custom_costs():
    # Re-editing 1 -> 2 is free, so p takes edit 1 and its leaves with 2 are re-edited.
    costs = pd.DataFrame([[0, 1, 1], [9, 0, 0], [9, 9, 0]], index=[0, 1, 2], columns=[0, 1, 2])
    tdata = clade_tdata([1, 2, 2, 1, 2])
    cas.tl.ancestral_characters(tdata, method="sankoff", costs=costs)
    assert states_of(tdata)["p"] == [1]


def test_ancestral_characters_errors():
    tdata = clade_tdata([1, 1, 2])
    with pytest.raises(ValueError, match="Unknown method"):
        cas.tl.ancestral_characters(tdata, method="fitch")
    with pytest.raises(ValueError, match="reedit_cost"):
        cas.tl.ancestral_characters(tdata, method="sankoff", reedit_cost=0)
    costs = pd.DataFrame(1.0, index=[0, 1], columns=[0, 1])
    with pytest.raises(CassiopeiaError, match="missing from the index"):
        cas.tl.ancestral_characters(tdata, method="sankoff", costs=costs)
    ambiguous = clade_tdata([1, 1, 2], outgroup=(1, 2))
    for method in ["conservative", "sankoff"]:
        with pytest.raises(CassiopeiaError, match="Ambiguous"):
            cas.tl.ancestral_characters(ambiguous, method=method)


def test_sankoff_chunks_characters(monkeypatch):
    # A tiny memory budget forces one character per chunk; results must not change.
    module = importlib.import_module("cassiopeia.tools.ancestral_characters")
    rng = np.random.default_rng(0)
    g = random_tree(rng, 25)
    tdata = leaf_tdata(g.edges, random_leaf_states(rng, g, n_chars=8))
    whole = cas.tl.ancestral_characters(tdata, method="sankoff", copy=True)
    monkeypatch.setattr(module, "_SANKOFF_MEMORY", 1)
    chunked = cas.tl.ancestral_characters(tdata, method="sankoff", copy=True)
    assert states_of(whole) == states_of(chunked)


def test_count_edge_mutations():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    cas.tl.count_edge_mutations(tdata, tree_key="tree")
    g = tdata.obst["tree"]
    # 6->4 is mutationless; 4->0 has one real mutation (character d, 0 -> 4)
    assert g.edges["6", "4"]["n_mutations"] == 0
    assert g.edges["4", "0"]["n_mutations"] == 1
    # 6->5 differs only by a transition to missing -> not counted by default
    assert g.edges["6", "5"]["n_mutations"] == 0


def test_count_edge_mutations_treat_missing():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    cas.tl.count_edge_mutations(tdata, tree_key="tree", treat_missing_as_mutation=True)
    g = tdata.obst["tree"]
    # 6->5 now counts the 5 -> -1 transition
    assert g.edges["6", "5"]["n_mutations"] == 1


def test_count_edge_mutations_requires_ancestral_states():
    tdata = make_tdata()
    with pytest.raises(CassiopeiaError):
        cas.tl.count_edge_mutations(tdata, tree_key="tree")


def test_count_edge_mutations_rejects_non_treedata():
    g = nx.DiGraph()
    g.add_edge("r", "a")
    with pytest.raises(TypeError):
        cas.tl.count_edge_mutations(g)


def test_collapse_edges_mutationless():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    cas.tl.collapse_edges(tdata, tree_key="tree")
    g = tdata.obst["tree"]
    # node 4 has identical states to its parent 6 and is collapsed out;
    # node 5 differs only by a transition to missing, so it is mutationless too
    assert "4" not in g.nodes
    assert "5" not in g.nodes
    assert set(leaves(g)) == {"0", "1", "2", "3"}
    # all leaves reattach directly to 6
    assert set(g.successors("6")) == {"0", "1", "2", "3"}


def test_collapse_edges_requires_ancestral_states():
    tdata = make_tdata()
    with pytest.raises(CassiopeiaError):
        cas.tl.collapse_edges(tdata, tree_key="tree")


def test_collapse_edges_rejects_non_treedata():
    g = nx.DiGraph()
    g.add_edge("r", "a")
    with pytest.raises(TypeError):
        cas.tl.collapse_edges(g)


def test_collapse_edges_unknown_criteria():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    with pytest.raises(ValueError):
        cas.tl.collapse_edges(tdata, tree_key="tree", criteria="not_a_criterion")


def test_collapse_edges_copy():
    tdata = make_tdata()
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    out = cas.tl.collapse_edges(tdata, tree_key="tree", copy=True)
    assert isinstance(out, td.TreeData)
    # original tree still has node 4
    assert "4" in tdata.obst["tree"].nodes
    assert "4" not in out.obst["tree"].nodes


def test_collapse_edges_mutationless_sums_branch_lengths():
    tdata = make_tdata()
    g = tdata.obst["tree"].copy()
    # node 4 is mutationless w.r.t. parent 6; its branch lengths should sum
    g["6"]["4"]["length"] = 1.5
    g["4"]["0"]["length"] = 2.0
    g["4"]["1"]["length"] = 0.5
    tdata.obst["tree"] = g
    cas.tl.ancestral_characters(tdata, tree_key="tree")
    cas.tl.collapse_edges(tdata, tree_key="tree")
    g = tdata.obst["tree"]
    assert "4" not in g.nodes
    assert g["6"]["0"]["length"] == pytest.approx(3.5)
    assert g["6"]["1"]["length"] == pytest.approx(2.0)


def make_unifurcation_tdata():
    g = nx.DiGraph()
    for u, v in [("0", "1"), ("0", "2"), ("1", "3"), ("2", "4"), ("2", "5")]:
        g.add_edge(u, v)
    g["0"]["1"]["length"] = 1.0
    g["0"]["2"]["length"] = 2.0
    g["1"]["3"]["length"] = 1.0
    g["2"]["4"]["length"] = 2.0
    g["2"]["5"]["length"] = 1.0
    return td.TreeData(obs=pd.DataFrame(index=["3", "4", "5"]), obst={"tree": g})


def test_collapse_edges_unifurcation():
    tdata = make_unifurcation_tdata()
    cas.tl.collapse_edges(tdata, tree_key="tree", criteria="unifurcation")
    g = tdata.obst["tree"]
    # node 1 (single child) is spliced out, branch lengths summed
    assert "1" not in g.nodes
    assert set(g.successors("0")) == {"2", "3"}
    assert g["0"]["3"]["length"] == pytest.approx(2.0)


def test_collapse_edges_unifurcation_needs_no_characters():
    # the structural criterion must not require character states on nodes
    tdata = make_unifurcation_tdata()
    out = cas.tl.collapse_edges(tdata, tree_key="tree", criteria="unifurcation", copy=True)
    assert "1" not in out.obst["tree"].nodes
    assert "1" in tdata.obst["tree"].nodes


def test_mean_depth():
    g = nx.DiGraph()
    for u, v in [("0", "1"), ("0", "2"), ("2", "3"), ("2", "4")]:
        g.add_edge(u, v)
    depths = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2}
    for n, d in depths.items():
        g.nodes[n]["depth"] = d
    tdata = td.TreeData(obs=pd.DataFrame(index=["1", "3", "4"]), obst={"tree": g})
    # leaves are 1 (depth 1), 3 and 4 (depth 2) -> mean 5/3
    assert cas.tl.mean_depth(tdata, "depth", tree_key="tree") == pytest.approx(5 / 3)
