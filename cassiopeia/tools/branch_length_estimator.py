import abc
import cvxpy as cp

from .tree import Tree


class BranchLengthEstimator(abc.ABC):
    r"""
    Abstract base class for all branch length estimators.
    """
    @abc.abstractmethod
    def estimate_branch_lengths(self, tree: Tree) -> None:
        r"""
        Annotates the tree's nodes with their estimated age, and
        the tree's branches with their lengths.
        Operates on the tree in-place.

        Args:
            tree: The tree for which to estimate branch lengths.
        """


class PoissonConvexBLE(BranchLengthEstimator):
    r"""
    A simple branch length estimator that assumes that the characters evolve IID
    over the phylogeny with the same cutting rate.

    Maximum Parsinomy is used to impute the ancestral states first. Doing so
    leads to a convex optimization problem.
    """
    def __init__(
        self,
        minimum_edge_length: float = 0,  # TODO: minimum_branch_length?
        l2_regularization: float = 0,
        verbose: bool = False
    ):
        self.minimum_edge_length = minimum_edge_length
        self.l2_regularization = l2_regularization
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: Tree) -> float:
        r"""
        TODO: This shouldn't return the log-likelihood according to the API.
        What should we do about this? Maybe let's look at sklearn?

        Estimates branch lengths for the given tree.

        This is in fact an exponential cone program, which is a special kind of
        convex problem:
        https://docs.mosek.com/modeling-cookbook/expo.html

        Args:
            tree: The tree for which to estimate branch lengths.

        Returns:
            The log-likelihood under the model for the computed branch lengths.
        """
        # Extract parameters
        minimum_edge_length = self.minimum_edge_length
        l2_regularization = self.l2_regularization
        verbose = self.verbose

        # # Wrap the networkx DiGraph for goodies.
        # T = Tree(tree)
        T = tree

        # # # # # Create variables of the optimization problem # # # # #
        r_X_t_variables = dict([(node_id, cp.Variable(name=f'r_X_t_{node_id}'))
                                for node_id in T.nodes()])
        time_increases_constraints = [
            r_X_t_variables[parent]
            >= r_X_t_variables[child] + minimum_edge_length
            for (parent, child) in T.edges()
        ]
        leaves_have_age_0_constraints =\
            [r_X_t_variables[leaf] == 0 for leaf in T.leaves()]
        non_negative_r_X_t_constraints =\
            [r_X_t >= 0 for r_X_t in r_X_t_variables.values()]
        all_constraints =\
            time_increases_constraints + \
            leaves_have_age_0_constraints + \
            non_negative_r_X_t_constraints

        # # # # # Compute the log-likelihood # # # # #
        log_likelihood = 0

        # Because all rates are equal, the number of cuts in each node is a
        # sufficient statistic. This makes the solver WAY faster!
        for (parent, child) in T.edges():
            edge_length = r_X_t_variables[parent] - r_X_t_variables[child]
            zeros_parent = T.get_state(parent).count('0')  # TODO: '0'...
            zeros_child = T.get_state(child).count('0')  # TODO: '0'...
            new_cuts_child = zeros_parent - zeros_child
            assert(new_cuts_child >= 0)
            # Add log-lik for characters that didn't get cut
            log_likelihood += zeros_child * (-edge_length)
            # Add log-lik for characters that got cut
            log_likelihood += new_cuts_child * cp.log(1 - cp.exp(-edge_length))

        # # # # # Add regularization # # # # #

        l2_penalty = 0
        for (parent, child) in T.edges():
            for child_of_child in T.children(child):
                edge_length_above =\
                    r_X_t_variables[parent] - r_X_t_variables[child]
                edge_length_below =\
                    r_X_t_variables[child] - r_X_t_variables[child_of_child]
                l2_penalty += (edge_length_above - edge_length_below) ** 2
        l2_penalty *= l2_regularization

        # # # # # Solve the problem # # # # #

        obj = cp.Maximize(log_likelihood - l2_penalty)
        prob = cp.Problem(obj, all_constraints)

        f_star = prob.solve(solver='ECOS', verbose=verbose)

        # # # # # Populate the tree with the estimated branch lengths # # # # #

        for node in T.nodes():
            T.set_age(node, age=r_X_t_variables[node].value)

        for (parent, child) in T.edges():
            new_edge_length =\
                r_X_t_variables[parent].value - r_X_t_variables[child].value
            T.set_edge_length(
                parent,
                child,
                length=new_edge_length)

        return f_star

    def score(self, tree: Tree) -> float:
        r"""
        The log-likelihood of the given data under the model
        """
        raise NotImplementedError()
