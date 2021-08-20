# distutils: language = c++

from ._iid_exponential_bayesian cimport _InferPosteriorTimes
from libcpp.vector cimport vector
from libcpp.map cimport map

from typing import List, Tuple

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class _PyInferPosteriorTimes:
    """
    Infer posterior node times under the Bayesian model.

    The method 'run' takes in all the information needed to perform inference.
    """
    cdef _InferPosteriorTimes* c_infer_posterior_times

    def __cinit__(self):
        self.c_infer_posterior_times = new _InferPosteriorTimes();

    def run(
        self,
        int N,
        vector[vector[int]] children,
        int root,
        vector[int] is_internal_node,
        vector[int] get_number_of_mutated_characters_in_node,
        vector[int] non_root_internal_nodes,
        vector[int] leaves,
        vector[int] parent,
        int K,
        vector[int] Ks,
        int T,
        double r,
        double lam,
        double sampling_probability,
        vector[int] is_leaf,
    ):
        """
        Infer posterior node time distribution.

        Args:
            N: Number of nodes in tree.
            children: Adjacency list of graph.
            root: Root of graph.
            is_internal_node: Binary indicator for whether the node is internal
                or not.
            get_number_of_mutated_characters_in_node: Number of mutated
                characters in the node.
            non_root_internal_nodes: The non-root internal nodes.
            leaves: The leaves of the tree.
            parent: The parent of each node in the tree (or a negative number
                for the root)
            K: The number of characters
            Ks: The number of non-missing characters in each node.
            T: The number of timesteps of the discretization.
            r: The CRISRP/Cas9 mutation rate.
            lam: The birth rate.
            sampling_probability: The probability that a leaf is subsampled from
                the ground truth phylogeny.
            is_leaf: Binary indicator for whether a node is a leaf or not.

        Raises:
            ValueError if the discretization level T is too small.
        """
        self.c_infer_posterior_times.run(
            N,
            children,
            root,
            is_internal_node,
            get_number_of_mutated_characters_in_node,
            non_root_internal_nodes,
            leaves,
            parent,
            K,
            Ks,
            T,
            r,
            lam,
            sampling_probability,
            is_leaf,
        )

    def get_posterior_means_res(self) -> List[Tuple[int, float]]:
        """
        Posterior mean node times.

        Returns a list of tuples (node, posterior_time), containing the posterior mean
        time 'posterior_time' of node 'node'.
        """
        return self.c_infer_posterior_times.get_posterior_means_res()
    
    def get_posteriors_res(self) -> List[Tuple[int, List[float]]]:
        """
        Posterior node time distributions.

        Returns a list of tuples (node, posterior_time_distribution), containing
        the posterior time 'posterior_time_distribution' of node 'node'. Here
        'posterior_time_distribution' is a list of length T + 1, where
        posterior_time_distribution[t] is the posterior probability that node
        'node' has (discretized) time t.

        Note that this is the normalized version of get_log_joints_res.
        """
        return self.c_infer_posterior_times.get_posteriors_res()
    
    def get_log_joints_res(self) -> List[Tuple[int, List[float]]]:
        """
        Joint (node, time) log probabilities.

        Returns a list of tuples (node, log_joint), containing the log joint
        probability of node 'node' taking a given (discretized) time t (given
        the observed character matrix and tree topology); this is log_joint[t].

        Note that this is the unnormalized version of get_log_joints_res.
        """
        return self.c_infer_posterior_times.get_log_joints_res()
    
    def get_log_likelihood_res(self):
        """
        Log likelihood of the observed data.

        The log likelihood of the observed character matrix and tree topology
        under the Bayesian model.

        Note that this is just the log-sum-exp of get_log_joints_res for any
        node.
        """
        return self.c_infer_posterior_times.get_log_likelihood_res()

    def __dealloc__(self):
        del self.c_infer_posterior_times

