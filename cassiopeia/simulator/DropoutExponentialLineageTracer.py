import numpy as np
import random

def overlay_mutation_continuous(
    network,
    num_chars,
    mutation_prob_map,
    basal_rate,
    cassette_size = 1,
    epsilon = None,
    silence_rates = None,
):
    for node in network.nodes:
        network.nodes[node]["characters"] = ["0"] * num_chars

    root = [n for n in network if network.in_degree(n) == 0][0]
    network.nodes[root]["parent_lifespan"] = 0

    for i in mutation_prob_map:  # edit the mutation map to only include the probs
        # of mutating to each state, given that character is chosen to mutate
        sum = 0
        mutation_prob_map[i].pop("0", None)
        for j in mutation_prob_map[i]:
            sum += mutation_prob_map[i][j]
        new_probs = {}
        for j in mutation_prob_map[i]:
            new_probs[j] = mutation_prob_map[i][j] / sum
        mutation_prob_map[i] = new_probs

    # Call the helper on the root, start propogating the heritable mutations
    # throughout the tree
    mutation_helper_continuous(
        network,
        root,
        basal_rate,
        mutation_prob_map,
        network.nodes[root]["characters"],
        set(),
        cassette_size,
        epsilon,
        silence_rates,
    )

    leaves = [n for n in network if network.out_degree(n) == 0 and network.in_degree(n) == 1]

    return leaves

def mutation_helper_continuous(
    network,
    node,
    basal_rate,
    mutation_prob_map,
    curr_mutations,
    dropout_indices,
    cassette_size,
    epsilon,
    silence_rates,
):
    # Copy the current mutations on the lineage and get the branch length
    # representing the lifespan of the current cell
    curr_sample = curr_mutations.copy()
    new_dropout_indices = dropout_indices.copy()
    t = network.nodes[node]["parent_lifespan"]

    # First acquire heritable dropout through epigenetic silencing from the
    # silencing rate from each character. If a heritable dropout occurs
    # on current branch, then it will mask any other mutations that occur,
    # so we don't need to consider the timing of the mutations (whether dropout
    # or mutation happens first)
    for i in range(len(silence_rates)):
        silence_prob = 1 - (np.exp(-t * silence_rates[i]))
        if random.uniform(0, 1) < silence_prob:
            for j in range(i * cassette_size, (i + 1) * cassette_size):
                new_dropout_indices.add(j)

    # Use exponential CDF to get probability of mutation over the branch
    p = 1 - (np.exp(-t * basal_rate))

    # Take only non-mutation/dropped out cells
    base_chars = []
    for i in range(0, len(curr_sample)):
        if curr_sample[i] == "0" and i not in new_dropout_indices:
            base_chars.append(i)

    # Use binomial to determine which characters mutate, identical to sampling
    # each characters independently with probability p
    draws = np.random.binomial(len(base_chars), p)
    chosen_ind = np.random.choice(base_chars, draws, replace=False)
    cassettes = {}

    new_sample = curr_sample[:]

    # For each chosen character, determine the state it mutates to, as well
    # as uniformly sample the time at which it mutates over the branch length
    # Annotate dictionary of the time frame when the mutation occured, 
    # determinedand by epsilon, and which cassette and target site it occurs
    for i in chosen_ind:
        values, probabilities = zip(*mutation_prob_map[i].items())
        new_character = np.random.choice(values, p=probabilities)
        new_sample[i] = new_character
        time = np.random.uniform(0.0, t)
        left = max(0, time - epsilon)
        right = min(t, time + epsilon)
        cass_num = i // cassette_size
        if cass_num in cassettes:
            cassettes[cass_num].append((left, right, i))
        else:
            cassettes[cass_num] = [(left, right, i)]

    # Create double resection events by looking at if there are mutations that
    # happen in the same time window on the same cassette. These events span
    # the indices where the mutations occur. Currently, we treat these as 
    # dropouts but perhaps we should treat them as mutations that occur across
    # multiple sites, and treat them as dropout only if they span the entire 
    # cassette
    for cass_num in cassettes.keys():
        if len(cassettes[cass_num]) > 1:
            time_ranges = []
            for cut in cassettes[cass_num]:
                time_ranges.append(cut)
            time_ranges.sort(key=lambda x: x[0])

            seen = set()
            for cut in time_ranges:
                if cut[2] in seen:
                    continue
                for cut2 in time_ranges:
                    if cut2[2] in seen:
                        continue
                    if cut[1] >= cut2[0]:
                        if cut[2] != cut2[2]:
                            for e in range(
                                min(cut[2], cut2[2]), max(cut[2], cut2[2]) + 1
                            ):
                                if e not in new_dropout_indices:
                                    new_dropout_indices.add(e)
                                    seen.add(e)
                                    seen.add(e)
                                    new_sample[e] = curr_sample[e]
                            break
                seen.add(cut[2])

    # Annotate the tree with the character matrix and dropouts
    network.nodes[node]["characters"] = new_sample
    network.nodes[node]["dropout"] = list(new_dropout_indices)

    # Call helper on each successor
    if network.out_degree(node) > 0:
        for i in network.successors(node):
            network.nodes[i]["parent_lifespan"] = network.get_edge_data(node, i)[
                "weight"
            ]
            mutation_helper_continuous(
                network,
                i,
                basal_rate,
                mutation_prob_map,
                new_sample,
                new_dropout_indices,
                cassette_size,
                epsilon,
                silence_rates,
            )