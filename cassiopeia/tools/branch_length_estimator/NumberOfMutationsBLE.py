from cassiopeia.data import CassiopeiaTree
from .BranchLengthEstimator import BranchLengthEstimator


class NumberOfMutationsBLE(BranchLengthEstimator):
    r"""
    A naive branch length estimator that estimates branch lengths
    as the number of mutations on that edge. This is thus a
    very naive baseline model.

    This estimator requires that the ancestral states are provided.

    Args:
        length_of_mutationless_edges: Useful to make them not collapse to 0.
        treat_missing_states_as_mutations: If True, missing states will be treated as
            their own CRISPR/Cas9 mutations.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        length_of_mutationless_edges: float = 0,
        treat_missing_states_as_mutations: bool = True,
        verbose: bool = False,
    ):
        self.length_of_mutationless_edges = length_of_mutationless_edges
        self.treat_missing_states_as_mutations = treat_missing_states_as_mutations
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        # Extract parameters
        length_of_mutationless_edges = self.length_of_mutationless_edges
        treat_missing_states_as_mutations = self.treat_missing_states_as_mutations
        verbose = self.verbose

        estimated_edge_lengths = {}

        # TODO: This is copy-pasta from IIDExponentialBLE.
        for (parent, child) in tree.edges:
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            num_uncuts = 0
            num_cuts = 0
            num_missing = 0
            for parent_state, child_state in zip(parent_states, child_states):
                # We only care about uncut states.
                if parent_state == 0:
                    if child_state == 0:
                        num_uncuts += 1
                    elif child_state == tree.missing_state_indicator:
                        num_missing += 1
                    else:
                        num_cuts += 1
            if treat_missing_states_as_mutations:
                # TODO: Test this functionality!
                num_cuts += num_missing
                num_missing = 0
            if num_cuts == 0:
                estimated_edge_length = length_of_mutationless_edges
            else:
                estimated_edge_length = num_cuts
            estimated_edge_lengths[(parent, child)] = estimated_edge_length


        times = {node: 0 for node in tree.nodes}
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = times[parent] + estimated_edge_lengths[(parent, child)]

        max_time = max(times.values())
        for leaf in tree.leaves:
            times[leaf] = max_time

        # We smooth out epsilons that might make a parent's time greater
        # than its child
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])
        tree.set_times(times)
