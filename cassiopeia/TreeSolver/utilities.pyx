from collections import defaultdict, OrderedDict
import networkx as nx
import random
import numpy as np
import scipy as sp
from skbio.tree import TreeNode, majority_rule
from io import StringIO
from tqdm import tqdm
from numba import njit
import pandas as pd
import pickle as pic
from ete3 import Tree
import pylab

from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.lineage_solver.solver_utils import node_parent


def tree_collapse(tree):
    """
	Given a networkx graph in the form of a tree, collapse two nodes together if there are no mutations seperating the two nodes

	:param graph: Networkx Graph as a tree
	:return: Collapsed tree as a Networkx object
	"""
    if isinstance(tree, nx.DiGraph):
        graph = tree
    else:
        graph = tree.get_network()

    new = {}
    for node in graph.nodes():

        if isinstance(node, str):
            new[node] = node.split("_")[0]
        else:
            new[node] = node.char_string

    new_graph = nx.relabel_nodes(graph, new)

    dct = defaultdict(str)
    while len(dct) != len(new_graph.nodes()):
        for node in new_graph:
            if "|" in node:
                dct[node] = node
            else:
                succ = list(new_graph.successors(node))
                if len(succ) == 1:
                    if "|" in succ[0]:
                        dct[node] = succ[0]
                    elif "|" in dct[succ[0]]:
                        dct[node] = dct[succ[0]]
                else:
                    if "|" in succ[0] and "|" in succ[1]:
                        dct[node] = node_parent(succ[0], succ[1])
                    elif "|" in dct[succ[0]] and "|" in succ[1]:
                        dct[node] = node_parent(dct[succ[0]], succ[1])
                    elif "|" in succ[0] and "|" in dct[succ[1]]:
                        dct[node] = node_parent(succ[0], dct[succ[1]])
                    elif "|" in dct[succ[0]] and "|" in dct[succ[1]]:
                        dct[node] = node_parent(dct[succ[0]], dct[succ[1]])

    new_graph = nx.relabel_nodes(new_graph, dct)
    new_graph.remove_edges_from(new_graph.selfloop_edges())

    final_dct = {}
    for n in new_graph:

        final_dct[n] = Node("state-node", character_vec=n.split("|"))

    new_graph = nx.relabel_nodes(new_graph, final_dct)

    # new2 = {}
    # for node in new_graph.nodes():
    # 	new2[node] = node+'_x'
    # new_graph = nx.relabel_nodes(new_graph, new2)

    return new_graph


def tree_collapse2(tree):
    """
	Given a networkx graph in the form of a tree, collapse two nodes together if there are no mutations seperating the two nodes

	:param graph: Networkx Graph as a tree
	:return: Collapsed tree as a Networkx object
	"""
    if isinstance(tree, nx.DiGraph):
        graph = tree
    else:
        graph = tree.get_network()

    leaves = [n for n in graph if graph.out_degree(n) == 0]
    for l in leaves:

        # traverse up beginning at leaf
        parent = list(graph.predecessors(l))[0]
        pair = (parent, l)

        root = [n for n in graph if graph.in_degree(n) == 0][
            0
        ]  # need to reinstantiate root, in case we reassigned it

        is_root = False
        while not is_root:

            u, v = pair[0], pair[1]
            if u == root:
                is_root = True

            if u.get_character_string() == v.get_character_string():

                # replace u with v
                children_of_parent = graph.successors(u)

                if not is_root:
                    new_parent = list(graph.predecessors(u))[0]

                graph.remove_node(u)  # removes parent node
                for c in children_of_parent:
                    if c != v:
                        graph.add_edge(v, c)

                if not is_root:
                    graph.add_edge(new_parent, v)
                    pair = (new_parent, v)

            else:
                if not is_root:
                    pair = (list(graph.predecessors(u))[0], u)

    return graph


def find_parent(node_list):
    parent = []
    if isinstance(node_list[0], Node):
        node_list = [n.get_character_string() for n in node_list]

    num_char = len(node_list[0].split("|"))
    for x in range(num_char):
        inherited = True

        state = node_list[0].split("|")[x]
        for n in node_list:
            if n.split("|")[x] != state:
                inherited = False

        if not inherited:
            parent.append("0")

        else:
            parent.append(state)

    return Node("state-node", parent, is_target=False)


def fill_in_tree(tree, cm=None):

    # rename samples to character strings
    rndct = {}
    for n in tree.nodes:
        if cm is None:
            rndct[n] = Node(n.name, n.char_string.split("|"), is_target=n.is_target)
        else:
            if n.name in cm.index.values:
                rndct[n] = Node(
                    "state-node",
                    [str(k) for k in cm.loc[n.name].values],
                    is_target=True,
                )
            else:
                rndct[n] = Node("state-node", "", is_target=False)

    tree = nx.relabel_nodes(tree, rndct)

    root = [n for n in tree if tree.in_degree(n) == 0][0]

    # run dfs and reconstruct states
    anc_dct = {}
    for n in tqdm(
        nx.dfs_postorder_nodes(tree, root), total=len([n for n in tree.nodes])
    ):
        if "|" not in n.char_string or len(n.char_string) == 0:
            children = list(tree[n].keys())
            for c in range(len(children)):
                if children[c] in anc_dct:
                    children[c] = anc_dct[children[c]]
            anc_dct[n] = find_parent(children)

    tree = nx.relabel_nodes(tree, anc_dct)

    tree.remove_edges_from(tree.selfloop_edges())

    return tree


def score_parsimony(net, priors=None):

    if not isinstance(net, nx.DiGraph):
        net = net.get_network()

    root = [n for n in net if net.in_degree(n) == 0][0]

    score = 0
    for e in nx.dfs_edges(net, source=root):

        assert isinstance(e[0], Node)

        score += e[0].get_mut_length(e[1], priors=priors)

    return score


def find_consensus_tree(trees, character_matrix, cutoff=0.5):

    _trees = [
        TreeNode.read(StringIO(convert_network_to_newick_format(t.network)))
        for t in trees
    ]
    np.random.shuffle(_trees)
    consensus = majority_rule(_trees, cutoff=cutoff)[0]
    G = nx.DiGraph()

    # Create dict from scikit bio TreeNode to cassiopeia.Node
    e2cass = {}
    for n in consensus.postorder():
        if n.name is not None:
            nn = Node(
                n.name,
                character_matrix.loc[n.name].values,
                is_target=True,
                support=n.support,
            )
        else:
            nn = Node("state-node", [], support=n.support)

        e2cass[n] = nn
        G.add_node(nn)

    for p in consensus.traverse("postorder"):

        pn = e2cass[p]

        for c in p.children:
            cn = e2cass[c]

            G.add_edge(pn, cn)

    return G


def get_modified_hamming_dist(n1, n2):

    x_list, y_list = n1.split("_")[0].split("|"), n2.split("_")[0].split("|")

    count = 0
    for i in range(0, len(x_list)):

        if x_list[i] == y_list[i]:
            count += 0

        elif (
            x_list[i] == "-" or y_list[i] == "-"
        ):
            count += 0

        elif x_list[i] == "0" or y_list[i] == "0":
            count += 1

        else:
            count += 2

    return count


def compute_pairwise_edit_dists(nodes, verbose=True):

    edit_dist = []
    _leaves = nodes

    all_pairs = []
    for i1 in tqdm(range(len(_leaves)), desc="Creating pairs to compare"):
        l1 = _leaves[i1]
        for i2 in range(i1 + 1, len(_leaves)):
            l2 = _leaves[i2]

            all_pairs.append((l1, l2))

    for p in tqdm(all_pairs, desc="Computing modified hamming distances"):

        edit_dist.append(get_modified_hamming_dist(p[0], p[1]))

    return np.array(edit_dist), all_pairs


def find_neighbors(target_nodes, n_neighbors=10):

    edit_dists, all_pairs = compute_pairwise_edit_dists(target_nodes)
    ds = sp.spatial.distance.squareform(edit_dists)

    sample_range = np.arange(ds.shape[0])[:, None]
    indices = np.argpartition(ds, n_neighbors - 1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(ds[sample_range, indices])]
    distances = ds[sample_range, indices]

    # create neighbors dict
    neighbors = {}
    dists = {}
    for i, inds in zip(range(len(indices)), indices):
        n = target_nodes[i]
        neighbors[n] = [target_nodes[j] for j in inds]
        dists[n] = [d for d in distances[i]]

    return neighbors, dists


def read_and_process_data(filename, lineage_group=None, intBC_minimum_appearance=0.1):
    """
	Given an alleletable file, converts the corresponding cell samples to a string format
	to be passed into the solver


	:param filename:
		The name of the corresponding allele table file
	:param lineage_group:
		The lineage group as a string to draw samples from, None if all samples are wanted
	:param intBC_minimum_appearance:
		The minimum percentage appearance an integration barcode should have across the lineage group
	:return:
		A dictionary mapping samples to their corresponding string representation
		Please use .values() to get the corresponding

	"""

    # Reading in the samples
    samples = defaultdict(dict)
    int_bc_counts = defaultdict(int)
    for line in open(filename):

        if "cellBC" in line:
            pass
        else:
            line = line.split()
            if lineage_group is None or lineage_group in line[6]:
                samples[line[0]][line[1] + "_1"] = line[3]
                samples[line[0]][line[1] + "_2"] = line[4]
                samples[line[0]][line[1] + "_3"] = line[5]
                int_bc_counts[line[1]] += 1

    # Filtering samples for integration barcodes that appear in at least intBC_minimum_appearance of samples
    filtered_samples = defaultdict(dict)
    for sample in samples:
        for allele in samples[sample]:
            if (
                int_bc_counts[allele.split("_")[0]]
                >= len(samples) * intBC_minimum_appearance
            ):
                filtered_samples[sample][allele] = samples[sample][allele]

    # Finding all unique integration barcodes
    intbc_uniq = set()
    for s in filtered_samples:
        for key in filtered_samples[s]:
            intbc_uniq.add(key)

    # Converting the samples to string format
    samples_as_string = defaultdict(str)
    int_bc_counter = defaultdict(dict)
    for intbc in sorted(list(intbc_uniq)):
        for sample in filtered_samples:

            if intbc in filtered_samples[sample]:
                if filtered_samples[sample][intbc] == "None":
                    samples_as_string[sample] += "0|"
                else:
                    if filtered_samples[sample][intbc] in int_bc_counter[intbc]:
                        samples_as_string[sample] += (
                            str(
                                int_bc_counter[intbc][filtered_samples[sample][intbc]]
                                + 1
                            )
                            + "|"
                        )
                    else:
                        int_bc_counter[intbc][filtered_samples[sample][intbc]] = (
                            len(int_bc_counter[intbc]) + 1
                        )
                        samples_as_string[sample] += (
                            str(
                                int_bc_counter[intbc][filtered_samples[sample][intbc]]
                                + 1
                            )
                            + "|"
                        )

            else:
                samples_as_string[sample] += "-|"
    # Dropping final | in string
    for sample in samples_as_string:
        samples_as_string[sample] = samples_as_string[sample][:-1]

    return samples_as_string


def convert_network_to_newick_format(graph, use_intermediate_names=True):
    """
	Given a networkx network, converts to proper Newick format.

	:param graph:
		Networkx graph object
	:return: String in newick format representing the above graph
	"""

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        if node.name == "internal" or node.name == "state-node":
            _name = node.get_character_string()
        else:
            _name = node.name

        if use_intermediate_names:
            return (
                "%s" % (_name,)
                if is_leaf
                else (
                    "("
                    + ",".join(_to_newick_str(g, child) for child in g.successors(node))
                    + ")"
                    + _name
                )
            )

        return (
            "%s" % (_name,) + ":1"
            if is_leaf
            else (
                "("
                + ",".join(_to_newick_str(g, child) for child in g.successors(node))
                + ")"
            )
        )

    def to_newick_str(g, root=0):  # 0 assumed to be the root
        return _to_newick_str(g, root) + ";"

    return to_newick_str(
        graph, [node for node in graph if graph.in_degree(node) == 0][0]
    )


def newick_to_network(newick_filepath, cm, f=1):
    """
	Given a file path to a newick file, convert to a directed graph.

	:param newick_filepath:
		File path to a newick text file
	:param f:
		Parameter to be passed to Ete3 while reading in the newick file. (Default 1)
	:return: a networkx file of the tree
	"""

    G = nx.DiGraph()  # the new graph
    cm_lookup = cm.apply(lambda x: "|".join(x.values), axis=1)

    try:
        tree = Tree(newick_filepath, format=f)
    except:
        tree = Tree(newick_filepath)

    # Create dict from ete3 node to cassiopeia.Node
    e2cass = {}
    for n in tree.traverse("postorder"):

        if "|" in n.name:
            nn = Node("state-node", n.name.split("|"), support=n.support)
        elif n.name != "" or "Inner" in n.name:
            nn = Node(n.name, [], support=n.support)
        else:
            nn = Node("state-node", [], support=n.support)

        if n.is_leaf() and nn.char_string in cm_lookup:
            nn.is_target = True

        e2cass[n] = nn
        G.add_node(nn)

    for p in tree.traverse("postorder"):

        pn = e2cass[p]

        for c in p.children:
            cn = e2cass[c]

            G.add_edge(pn, cn)

    return G


def get_indel_props(at, group_var=["intBC"]):
    """
	Given an alleletable file, this function will split the alleletable into independent
	lineage groups and estimate the indel formation probabilities. This is done by
	treating each intBC as an independent observation and counting how many intBC contain
	a specific mutation, irrespective of the cell.

	:param at:
		The allele table pandas DataFrame
	:param group_var:
		Columns by which to group and count indels. This will effectively serve as the denominator when calculating the 
		frequenceis (i.e. N intBCs or M * N_m for M lineage groups and N_m intBCs per lineage groups if you group by intBC and LG)
	:return:
		An M x 2 pandas DataFrame mapping all M mutations to the frequency and raw counts
		of how many intBC this mutation appeared on.

	"""

    uniq_alleles = np.union1d(at["r1"], np.union1d(at["r2"], at["r3"]))

    groups = at.groupby(group_var).agg({"r1": "unique", "r2": "unique", "r3": "unique"})

    count = defaultdict(int)

    for i in tqdm(groups.index, desc="Counting unique alleles"):
        alleles = np.union1d(
            groups.loc[i, "r1"], np.union1d(groups.loc[i, "r2"], groups.loc[i, "r3"])
        )
        for a in alleles:
            if a != a:
                continue
            if "None" not in a:
                count[a] += 1

    tot = len(groups.index)
    freqs = dict(zip(list(count.keys()), [v / tot for v in count.values()]))

    return_df = pd.DataFrame([count, freqs]).T
    return_df.columns = ["count", "freq"]

    return_df.index.name = "indel"
    return return_df


def process_allele_table(
    cm, no_context=False, mutation_map=None, to_drop=None, allele_rep_thresh=1.0
):
    """
	Given an alleletable, create a character strings and lineage-specific mutation maps.
	A character string for a cell consists of a summary of all mutations observed at each
	cut site, delimited by a '|' character. We codify these mutations into integers, where
	each unique integer in a given column (character cut site) corresponds to a unique
	indel observed at that site.

	:param cm:
		The allele table pandas DataFrame.
	:param no_context:
		Do not use sequence context when identifying unique indels. (Default = False)
	:param mutation_map:
		A specification of the indel formation probabilities, in the form of a pandas
		DataFrame as returned from the `get_indel_props` function. (Default = None)
	:return:
		A list of three items - 1) a conversion of all cells into "character strings",
		2) A dictionary mapping each character to its mutation map. This mutation map
		is another dictionary storing the probabilities of mutating to any given
		state at that character. (e.g. {"character 1": {1: 0.5, 2: 0.5}})
		(C) dictionary mapping each indel to the number that represents it per
		character.
	"""

    filtered_samples = defaultdict(OrderedDict)
    for sample in cm.index:
        cell = cm.loc[sample, "cellBC"]
        intBC = cm.loc[sample, "intBC"]
        cut_sites = ["_r1", "_r2", "_r3"]

        to_add = []
        i = 1
        for c in cut_sites:
            if intBC + c not in to_drop:
                if no_context:
                    to_add.append(("intBC", c, "r" + str(i) + "_no_context"))
                else:
                    to_add.append(("intBC", c, "r" + str(i)))

            i += 1

        for ent in to_add:
            filtered_samples[cell][cm.loc[sample, ent[0]] + ent[1]] = cm.loc[
                sample, ent[2]
            ]

    samples_as_string = defaultdict(str)
    allele_counter = defaultdict(OrderedDict)

    _intbc_uniq = []
    allele_dist = defaultdict(list)
    for s in filtered_samples:
        for key in filtered_samples[s]:
            if key not in _intbc_uniq:
                _intbc_uniq.append(key)
            allele_dist[key].append(filtered_samples[s][key])

    # remove intBCs that are not diverse enough
    intbc_uniq = []
    dropped = []
    for key in allele_dist.keys():

        props = np.unique(allele_dist[key], return_counts=True)[1]
        props = props / len(allele_dist[key])
        if np.any(props > allele_rep_thresh):
            dropped.append(key)
        else:
            intbc_uniq.append(key)

    print(
        "Dropping the following intBCs due to lack of diversity with threshold "
        + str(allele_rep_thresh)
        + ": "
        + str(dropped)
    )

    prior_probs = defaultdict(dict)
    indel_to_charstate = defaultdict(dict)
    # for all characters
    for i in tqdm(range(len(list(intbc_uniq))), desc="Processing characters"):

        c = list(intbc_uniq)[i]

        # for all samples, construct a character string
        for sample in filtered_samples.keys():

            if c in filtered_samples[sample]:

                state = filtered_samples[sample][c]

                if type(state) != str and np.isnan(state):
                    samples_as_string[sample] += "-|"
                    continue

                if state == "NONE" or "None" in state:
                    samples_as_string[sample] += "0|"
                else:
                    if state in allele_counter[c]:
                        samples_as_string[sample] += (
                            str(allele_counter[c][state] + 1) + "|"
                        )
                    else:
                        # if this is the first time we're seeing the state for this character,
                        allele_counter[c][state] = len(allele_counter[c]) + 1
                        samples_as_string[sample] += (
                            str(allele_counter[c][state] + 1) + "|"
                        )

                        # add a new entry to the character's probability map
                        if mutation_map is not None:
                            prob = np.mean(mutation_map.loc[state]["freq"])
                            prior_probs[i][str(len(allele_counter[c]) + 1)] = float(
                                prob
                            )
                            indel_to_charstate[i][
                                str(len(allele_counter[c]) + 1)
                            ] = state
            else:
                samples_as_string[sample] += "-|"
    for sample in samples_as_string:
        samples_as_string[sample] = samples_as_string[sample][:-1]

    return samples_as_string, prior_probs, indel_to_charstate


def string_to_cm(string_sample_values):
    """
	Create a character matrix from the character strings, as created in `process_allele_table`.

	:param string_sample_values:
	   Input character strings, one for each cell.
	:return:
	   A Character Matrix, returned as a pandas DataFrame of size N x C, where we have
	   N cells and C characters.
	"""

    m = len(string_sample_values[list(string_sample_values.keys())[0]].split("|"))
    n = len(string_sample_values.keys())

    cols = ["r" + str(i) for i in range(m)]
    cm = pd.DataFrame(np.zeros((n, m)))
    indices = []
    for i, k in zip(range(n), string_sample_values.keys()):
        indices.append(k)
        alleles = np.array(string_sample_values[k].split("|"))
        cm.iloc[i, :] = alleles

    cm.index = indices
    cm.index.name = "cellBC"
    cm.columns = cols

    return cm


def write_to_charmat(string_sample_values, out_fp):
    """
	Write the character strings out to file.

	:param string_sample_values:
		Input character strings, one for each cell.
	:param out_fp:
		File path to be written to.
	:return:
		None.
	"""

    m = len(string_sample_values[list(string_sample_values.keys())[0]].split("|"))

    with open(out_fp, "w") as f:

        cols = ["cellBC"] + [("r" + str(i)) for i in range(m)]
        f.write("\t".join(cols) + "\n")

        for k in string_sample_values.keys():

            f.write(k)
            alleles = string_sample_values[k].split("|")

            for a in alleles:
                f.write("\t" + str(a))

            f.write("\n")


def alleletable_to_character_matrix(
    at,
    out_fp=None,
    mutation_map=None,
    no_context=False,
    write=True,
    to_drop=[],
    allele_rep_thresh=1.0,
):
    """
	Wrapper function for creating character matrices out of allele tables.

	:param at:
		Allele table as a pandas DataFrame.
	:param out_fp:
		Output file path, only necessary when write = True (Default = None)
	:param mutation_map:
		Mutation map as a pandas DataFrame. This can be created with the
		`get_indel_props` function. (Default = None)
	:param no_context:
		Do not use sequence context when calling character states (Default = False)
	:param write:
		Write out to file. This requires `out_fp` to be specified as well. (Default = True)
	:param to_drop:
		List of Target Sites to omit (Default = [])
	:return:
		None if write is specified. If not, this returns three items: return an N x C character matrix as a pandas DataFrame, the
		mutation map, and the indel to character state mapping. If writing out to file,
		the mutation and indel to character state mappings are also saved as pickle
		files.
	"""

    character_matrix_values, prior_probs, indel_to_charstate = process_allele_table(
        at,
        no_context=no_context,
        mutation_map=mutation_map,
        to_drop=to_drop,
        allele_rep_thresh=allele_rep_thresh,
    )

    if write:

        out_stem = "".join(out_fp.split(".")[:-1])
        if out_fp is None:
            raise Exception("Need to specify an output file if writing to file")

        write_to_charmat(character_matrix_values, out_fp)

        if mutation_map is not None:
            # write prior probability dictionary to pickle for convenience
            pic.dump(prior_probs, open(out_stem + "_priorprobs.pkl", "wb"))

            # write indel to character state mapping to pickle
            pic.dump(
                indel_to_charstate, open(out_stem + "_indel_character_map.pkl", "wb")
            )

    else:
        return string_to_cm(character_matrix_values), prior_probs, indel_to_charstate


def create_bootstrap_from_alleletable(at, no_context=False, mutation_map=None, B=100):

    lineage_profile = alleletable_to_lineage_profile(
        at, no_context=no_context, write=False
    )

    intbcs = at["intBC"].unique()
    M = len(intbcs)

    cms = []

    for b in tqdm(range(B), desc="drawing bootstrap samples"):

        samp = np.random.choice(intbcs, M, replace=True)
        intbc_b = sum([[i + "_r1", i + "_r2", i + "_r3"] for i in samp], [])
        b_sample = lineage_profile[intbc_b]

        cm_b, prior_probs, indel_to_charstate = lineage_profile_to_charmat(
            b_sample, mutation_map=mutation_map
        )

        cms.append((cm_b, prior_probs, indel_to_charstate, intbc_b))

    return cms


def lineage_profile_to_charmat(lp, mutation_map=None):

    prior_probs = defaultdict(dict)
    indel_to_charstate = defaultdict(dict)

    samples = []

    lp.columns = ["r" + str(i) for i in range(len(lp.columns))]
    cols_to_unique = dict(
        zip(lp.columns, [lp[x].factorize()[1].values for x in lp.columns])
    )
    cols_to_num = dict(zip(lp.columns, range(lp.shape[1])))

    mut_counter = dict(zip(lp.columns, [0] * lp.shape[1]))
    mut_to_state = defaultdict(dict)

    for col in cols_to_unique.keys():

        for _it in cols_to_unique[col]:
            if (type(_it) != str and np.isnan(_it)) or _it == "NC":
                mut_to_state[col][_it] = "-"

            elif "None" in _it or "NONE" in _it:
                mut_to_state[col][_it] = "0"

            else:
                mut_to_state[col][_it] = mut_counter[col] + 1
                mut_counter[col] += 1
                
                if mutation_map is not None:
                    c = cols_to_num[col]
                    prob = np.mean(mutation_map.loc[_it]["freq"])
                    prior_probs[c][str(mut_to_state[col][_it])] = float(prob)
                    indel_to_charstate[c][str(mut_to_state[col][_it])] = _it

    cm = lp.apply(
        lambda x: [
            mut_to_state[x.name][v] if type(v) == str else "-" for v in x.values
        ],
        axis=0,
    )

    cm.index = lp.index
    cm.columns = ["r" + str(i) for i in range(lp.shape[1])]

    return cm, prior_probs, indel_to_charstate


def alleletable_to_lineage_profile(lg, out_fp=None, no_context=False, write=True):
    """
	Wrapper function for creating lineage profiles out of allele tables. These are
	identical in concept to character matrices but retain their mutation identities
	as values in the matrix rather than integers.

	:param at:
		Allele table as a pandas DataFrame.
	:param out_fp:
		Output file path, only necessary when write = True (Default = None)
	:param no_context:
		Do not use sequence context when calling character states (Default = False)
	:param write:
		Write out to file. This requires `out_fp` to be specified as well. (Default = True)
	:return:
		None if write is specified. If not, return an N x C lineage profile as a pandas DataFrame.
	"""

    if no_context:
        g = lg.groupby(["cellBC", "intBC"]).agg(
            {
                "r1_no_context": "unique",
                "r2_no_context": "unique",
                "r3_no_context": "unique",
            }
        )
    else:
        g = lg.groupby(["cellBC", "intBC"]).agg(
            {"r1": "unique", "r2": "unique", "r3": "unique"}
        )

    intbcs = lg["intBC"].unique()

    # create mutltindex df by hand
    i1 = []
    for i in intbcs:
        i1 += [i] * 3

    if no_context:
        i2 = ["r1_no_context", "r2_no_context", "r3_no_context"] * len(intbcs)
    else:
        i2 = ["r1", "r2", "r3"] * len(intbcs)

    indices = [i1, i2]

    allele_piv = pd.DataFrame(index=g.index.levels[0], columns=indices)

    for j in tqdm(g.index, desc="filling in multiindex table"):
        vals = map(lambda x: x[0], g.loc[j])
        if no_context:
            (
                allele_piv.loc[j[0]][j[1], "r1_no_context"],
                allele_piv.loc[j[0]][j[1], "r2_no_context"],
                allele_piv.loc[j[0]][j[1], "r3_no_context"],
            ) = vals
        else:
            (
                allele_piv.loc[j[0]][j[1], "r1"],
                allele_piv.loc[j[0]][j[1], "r2"],
                allele_piv.loc[j[0]][j[1], "r3"],
            ) = vals

    allele_piv2 = pd.pivot_table(
        lg, index=["cellBC"], columns=["intBC"], values="UMI", aggfunc=pylab.size
    )
    col_order = (
        allele_piv2.dropna(axis=1, how="all")
        .sum()
        .sort_values(ascending=False, inplace=False)
        .index
    )

    lineage_profile = allele_piv[col_order]

    # collapse column names here
    lineage_profile.columns = [
        "_".join(tup).rstrip("_") for tup in lineage_profile.columns.values
    ]

    if write:
        if out_fp is None:
            raise Exception("Specify an output file")
        lineage_profile.to_csv(out_fp, sep="\t")
    else:
        return lineage_profile
