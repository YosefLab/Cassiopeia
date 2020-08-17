from collections import defaultdict
import networkx as nx
import numpy as np
import hashlib

from .solver_utils import root_finder, get_edge_length


def find_split(
    nodes,
    priors=None,
    considered=set(),
    fuzzy=False,
    probabilistic=False,
    minimum_allele_rep=1.0
):

    # Tracks frequency of states for each character in nodes
    character_mutation_mapping = defaultdict(int)

    # Tracks frequency of dropout for each character in nodes
    missing_value_prop = defaultdict(float)

    # Accounting for frequency of mutated states per character, in order to choose the best split
    for node in nodes:
        node_list = node.split("_")[0].split("|")
        for i in range(0, len(node_list)):
            char = node_list[i]

            if char == "-":
                missing_value_prop[str(i)] += 1.0 / len(nodes)

            if (str(i), char) not in considered:
                # you can't split on a 'None' state
                if char != "0":
                    if priors:
                        character_mutation_mapping[(str(i), char)] -= np.log(
                            priors[int(i)][char]
                        )
                    else:
                        character_mutation_mapping[(str(i), char)] += 1

    # Choosing the best mutation to split on (ie character and state)
    character, state = 0, 0
    max_cost = 0

    min_prior = 1
    if priors:
        for i in priors.keys():
            for j in priors[i].keys():
                min_prior = min(min_prior, priors[i][j])

    if probabilistic:

        entries, vals = (
            list(character_mutation_mapping.keys()),
            list(character_mutation_mapping.values()),
        )
        tot = np.sum([v for v in vals])
        probs = [v / tot for v in vals]
        entry = entries[np.random.choice(list(range(len(entries))), p=probs)]

        character, state = int(entry[0]), entry[1]

    else:
        epsilon = 0
        for i, j in character_mutation_mapping:

            if fuzzy:
                epsilon = np.random.normal()

            if (
                max_cost < (character_mutation_mapping[(i, j)] + epsilon)
                and missing_value_prop[str(i)] < minimum_allele_rep
            ):
                max_cost = character_mutation_mapping[(i, j)]
                character, state = i, j

        character = int(character)

    return character, state

def classify_missing_value(
    node,
    left_split,
    right_split,
    knn_neighbors,
    knn_distances,
    theta=0.1,
    kernel=True,
    mode="knn",
    lookahead_depth=3,
    left_states=[],
    right_states=[]
):
    """
	Classifies a cell with a missing value as belonging in the left split or the right split of a character split. This function will return a 
	boolean indicating whether or not the node belongs in the right split (i.e. has the charcter state).

	:param node:
		A node, represented as a character string: 'Ch1|Ch2|....|Chn'
	:param left_split:
		A list of nodes that are inferred not to have the character state (i.e. negatives)
	:param right_split:
		A list of nodes that are inferred to have the character state (i.e. positives)
	:param knn_neighbors:
		A dictionary storing for each node its closest neighbors
	:param knn_distances:
		A dictionary storing for each node the allele distances to its closest neighbors. These should be modified allele distances
	:param theta:
		Width of the Gaussian Kernel used to smooth the KNN distances. Only used if kernel = True and mode = 'knn' (default)
	:param kernel:
		Apply a Guassian kernel to smooth the KNN distances. Only used if mode = 'knn' (default)
	:param mode:
		Choose a mode to classify negative cells:
			- 'knn': assign based on a k-nearest-neighbor approach
			- 'avg': assign based on average similarity to either groups using a naive hamming distance
			- 'modified_avg': assign based on average similairty using a slightly more nuanced similarity function (A-A + 2, A-None + 1, None-None/Missing-A + 0)
	:return:
		Returns a boolean - True if the node belongs in the right split and False if it belongs in the left split.
	"""

    right_split_score = 0
    left_split_score = 0

    if mode == "knn":
        for n_i, neighbor in zip(range(len(knn_neighbors[node])), knn_neighbors[node]):

            if neighbor in right_split:
                if not kernel:
                    right_split_score += 1
                else:
                    right_split_score += np.exp(
                        -1 * knn_distances[node][n_i] / 0.1 ** 2
                    )
            if neighbor in left_split:
                # if the neighbor isn't in the right split, by default we prefer to put it
                # into the left split
                if not kernel:
                    left_split_score += 1
                else:
                    left_split_score += np.exp(-1 * knn_distances[node][n_i] / 0.1 ** 2)

        if not kernel:
            normfact = len(knn_neighbors[node])
        else:
            normfact = np.sum(
                [
                    np.exp(knn_distances[node][n_i])
                    for n_i in range(len(knn_neighbors[node]))
                ]
            )

        avg_right_split_score = right_split_score / normfact
        avg_left_split_score = left_split_score / normfact

    elif mode == "avg":

        node_list = node.split("|")
        num_not_missing = len([n for n in node_list if n != "-"])
        for i in range(0, len(node_list)):
            if node_list[i] != "0" and node_list[i] != "-":
                for node_2 in left_split:
                    node2_list = node_2.split("|")
                    if node_list[i] == node2_list[i]:
                        left_split_score += 1
                for node_2 in right_split:
                    node2_list = node_2.split("|")
                    if node_list[i] == node2_list[i]:
                        right_split_score += 1

        avg_left_split_score = left_split_score / float(
            len(left_split) * num_not_missing + 1
        )
        avg_right_split_score = right_split_score / float(
            len(right_split) * num_not_missing + 1
        )

    elif mode == "modified_avg":

        node_list = node.split("|")
        for i in range(0, len(node_list)):

            for node_2 in left_split:
                node2_list = node_2.split("|")
                if node_list[i] == node2_list:
                    left_split_score += 2
                if node_list[i] == "0" or node2_list[i] == "0":
                    left_split_score += 1

            for node_2 in right_split:
                node2_list = node_2.split("|")
                if node_list[i] == node2_list:
                    right_split_score += 2
                if node_list[i] == "0" or node2_list[i] == "0":
                    right_split_score += 1

        avg_left_split_score = left_split_score / float(len(left_split) + 1)
        avg_right_split_score = right_split_score / float(len(right_split) + 1)

    elif mode == "lookahead":

        node_list = node.split("|")
        left_score, right_score = 0, 0
        for char in left_states:
            if node_list[char] == left_states[char]:
                left_score = left_score + 1
        for char in right_states:
            if node_list[char] == right_states[char]:
                right_score = right_score + 1

        avg_right_split_score = right_score
        avg_left_split_score = left_score

    else:
        raise Exception(
            "Classification method not recognized. Please choose from: lookahead, knn, avg, modified_avg"
        )

    if avg_right_split_score >= avg_left_split_score:
        return True

    return False


def perform_split(
    nodes,
    character,
    state,
    knn_neighbors,
    knn_distances,
    considered,
    missing_data_mode="lookahead",
    lookahead_depth=3
):
    """
	Performs a split on a given character and state, separating the set of targets into two mutually exclusive groups based on the 
	presence or absence of the character state. This procedure also will classify cells with missing values in the selected character, 
	using the `classify_missing_value` function.

	:param targets:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param character:
		An integer indicating the position in the character array to consider.
	:param state:
		An integer indicating a particular state in the character on which to split.
	:return:
		Returns a set of two lists - right_split and left_split - segmenting the targets. Cells in the right split were inferred to have
		the character state and those in the left split did not. 
	"""

    # Splitting nodes based on whether they have the mutation, don't have the mutation, or are NA('-') in that character
    # Right split is where nodes with the mutation go, everyone else goes to left split or NA chars
    left_split, right_split, NA_chars = [], [], []
    
    if state == "-":
        for node in nodes:
            node_list = node.split("|")
            if node_list[character] == state:
                right_split.append(node)
            else:
                left_split.append(node)
        return left_split, right_split

    for node in nodes:
        node_list = node.split("|")
        if node_list[character] == state:
            right_split.append(node)
        elif node_list[character] == "-":
            NA_chars.append(node)
        else:
            left_split.append(node)

    # order NA_chars by "strongest" candidates for imputation
    if missing_data_mode == "knn":
        NA_scores = []
        for node in NA_chars:
            score = 0
            for neighbor in knn_neighbors[node]:
                if neighbor in right_split or neighbor in left_split:
                    score += 1
            NA_scores.append(score)

        NA_dict = dict(zip(NA_chars, NA_scores))

    else:
        NA_dict = dict(zip(NA_chars, [1] * len(NA_chars)))

    left_states, right_states = [], []
    if missing_data_mode == "lookahead":

        left_states = look_ahead_helper(
            left_split, lookahead_depth, dict(), considered.copy()
        )
        right_states = look_ahead_helper(
            right_split, lookahead_depth, dict(), considered.copy()
        )

    # Seperates all nodes with NA in the character chosen to be split upon
    # Puts in right split or left split based on which list shares more mutated characters with this string
    for node, score in sorted(NA_dict.items(), key=lambda kv: kv[1]):

        if classify_missing_value(
            node,
            left_split,
            right_split,
            knn_neighbors,
            knn_distances,
            theta=0.1,
            kernel=True,
            mode=missing_data_mode,
            lookahead_depth=lookahead_depth,
            left_states=left_states,
            right_states=right_states
        ):
            right_split.append(node)
        else:
            left_split.append(node)

    return left_split, right_split

def look_ahead_helper(targets, depth, splits, considered):

    if depth == 0 or len(targets) == 1 or len(targets) == 0:
        splits_temp = splits.copy()
        return splits_temp
    else:
        character, state = find_split(targets, considered=considered.copy())
        splits[character] = state
        considered.add((str(character), state))
        left_split, right_split, NA_chars = [], [], []
        
        for node in targets:
            node_list = node.split("|")
            if node_list[character] == state:
                right_split.append(node)
            elif node_list[character] == "-":
                NA_chars.append(node)
            else:
                left_split.append(node)

        left_states = look_ahead_helper(
            left_split, depth - 1, splits.copy(), considered.copy()
        )
        right_states = look_ahead_helper(
            right_split, depth - 1, splits.copy(), considered.copy()
        )

        right_states.update(left_states)
        return right_states

def greedy_build_heritable(
    nodes,
    knn_neighbors,
    knn_distances,
    priors=None,
    cell_cutoff=200,
    lca_cutoff=None,
    considered=set(),
    uniq="",
    targets=[],
    fuzzy=False,
    probabilistic=False,
    minimum_allele_rep=1.0,
    missing_data_mode="lookahead",
    lookahead_depth=3
):
    """
	Greedy algorithm which finds a probable mutation subgraph for given nodes.
	This algorithm chooses splits within the tree based on which mutation occurs most frequently,
	weighted by the prior probabilities of each mutation state for each character.
	Strings with NA ('-') as a state in the split character are segregated with the
	set of nodes which they most closely match to w.r.t. all other characters.

	:param nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param knn_neighbors:
		A dictionary storing for each node its closest neighbors
	:param knn_distances:
		A dictionary storing for each node the allele distances to its closest neighbors. These should be modified allele distances
	:param priors:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:param cutoff:
		A cutoff that tells the greedy algorithm to stop, and return a partial sub-tree
		Set to -1 to run through to the individual samples (ie return the full tree)

	:param considered:
		Internal parameter which keeps track of which mutations have been considered in a set
		DO NOT MODIFY
	:param uniq:
		Internal parameter which keeps track of the path of mutations (1 = mutation taken, 0 = mutation not taken)
		DO NOT MODIFY
	:return:
		Returns a graph which contains splits as nodes in the form "character state (uniq_identifier)", and leaves
		as either samples, or the roots of the subsets of samples that need to be considered by another algorithm.
		Edges are labeled with the corresponding mutation taken
		AND
		a list in the form [[sub_root, sub_samples],....] which is a list of subproblems still needed to be solved
	"""

    # G models the network that is returned recursively
    G = nx.DiGraph()

    root = root_finder(nodes)
    if lca_cutoff is not None:
        distances = [get_edge_length(root, t) for t in nodes]

    # Base case check for recursion, returns a graph with one node corresponding to the root of the remaining nodes
    if lca_cutoff is not None:
        if max(distances) <= lca_cutoff or len(nodes) == 1:
            root = root_finder(nodes)
            G.add_node(root)
            return G, [[root, nodes]]
    else:
        if len(nodes) <= cell_cutoff or len(nodes) == 1:
            root = root_finder(nodes)
            G.add_node(root)
            return G, [[root, nodes]]

    character, state = find_split(
        nodes,
        priors=priors,
        considered=considered.copy(),
        fuzzy=fuzzy,
        probabilistic=probabilistic,
        minimum_allele_rep=minimum_allele_rep
    )

    # If there is no good split left, stop the process and return a graph with the remainder of nodes
    if character == 0 and state == 0:
        if len(nodes) == 1:
            G.add_node(nodes[0])
        else:
            for i in range(0, len(nodes)):
                if nodes[i] != root:
                    G.add_edge(root, nodes[i])
        return G, []

    # Add character, state that split occurred to already considered mutations
    considered.add((str(character), state))

    left_split, right_split = perform_split(
        nodes,
        character,
        state,
        knn_neighbors,
        knn_distances,
        considered.copy(),
        missing_data_mode,
        lookahead_depth
    )

    # Create new graph for storing results
    G = nx.DiGraph()
    splitter = root

    # Recursively build left side of network (ie side that did not mutation at the character with the specific state)
    G.add_node(splitter)
    left_subproblems = []
    left_network = None
    if len(left_split) != 0:
        left_root = root_finder(left_split)

        left_network, left_subproblems = greedy_build_heritable(
            left_split,
            knn_neighbors,
            knn_distances,
            priors,
            cell_cutoff,
            lca_cutoff,
            considered.copy(),
            uniq + "0",
            targets,
            fuzzy,
            probabilistic,
            minimum_allele_rep,
            missing_data_mode,
            lookahead_depth
        )

        left_nodes = [
            node for node in left_network.nodes() if left_network.in_degree(node) == 0
        ]
        dup_dict = {}
        for n in left_network:
            if n in list(G.nodes()) and n != left_root:
                dup_dict[n] = (
                    n + "_" + str(hashlib.md5(left_root.encode("utf-8")).hexdigest())
                )
        left_network = nx.relabel_nodes(left_network, dup_dict)
        G = nx.compose(G, left_network)
        if root != left_root:
            G.add_edge(splitter, left_root, weight=0, label="None")

    # Recursively build right side of network
    right_network, right_subproblems = greedy_build_heritable(
        right_split,
        knn_neighbors,
        knn_distances,
        priors,
        cell_cutoff,
        lca_cutoff,
        considered.copy(),
        uniq + "1",
        targets,
        fuzzy,
        probabilistic,
        minimum_allele_rep,
        missing_data_mode,
        lookahead_depth
    )
    right_nodes = [
        node for node in right_network.nodes() if right_network.in_degree(node) == 0
    ]
    right_root = root_finder(right_split)

    dup_dict = {}
    for n in right_network:
        if n in list(G.nodes()) and n != right_root:
            dup_dict[n] = (
                n + "_" + str(hashlib.md5(right_root.encode("utf-8")).hexdigest())
            )
    for n in dup_dict:
        rename_dict = {n: dup_dict[n]}
        if right_network.out_degree(n) != 0:
            right_network = nx.relabel_nodes(right_network, rename_dict)
        else:
            rename_dict = {n: dup_dict[n]}
            G = nx.relabel_nodes(G, rename_dict)

    G = nx.compose(G, right_network)

    if root != right_root:
        if not priors:
            G.add_edge(
                splitter,
                right_root,
                weight=1,
                label=str(character) + ": 0 -> " + str(state),
            )
        else:
            G.add_edge(
                splitter,
                right_root,
                weight=-np.log(priors[int(character)][state]),
                label=str(character) + ": 0 -> " + str(state),
            )

    return G, left_subproblems + right_subproblems


def compute_entropy_of_split(cells):

    C = len(cells[0].split("|"))
    N = len(cells)

    entropies = []
    for c in range(C):
        counts_per_state = defaultdict(int)

        for cell in cells:
            state = cell.split("|")[c]
            counts_per_state[state] += 1

        # convert counts to frequencies
        counts_per_state = dict([(k, v / N) for k, v in counts_per_state.items()])

        ent = -1 * np.sum([p * np.log(p) for p in counts_per_state.values()])
        entropies.append(ent)

    return np.mean(entropies)
