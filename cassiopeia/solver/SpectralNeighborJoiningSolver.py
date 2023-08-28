"""
This file stores a subclass of DistanceSolver, SpectralNeighborJoining. This 
algorithm is based on the one developed by Jaffe, et al. (2021) in their paper 
titled "Spectral neighbor joining for reconstruction of latent tree models",
published in the SIAM Journal for Math and Data Science. 
 
"""
from typing import Callable, Dict, List, Optional, Tuple

import itertools
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver import (
    dissimilarity_functions,
    DistanceSolver,
    solver_utilities,
)



class SpectralNeighborJoiningSolver(DistanceSolver.DistanceSolver):
    """
    Spectral Neighbor-Joining class for Cassiopeia.

    Implements a variation on the Spectral Neighbor-Joining (abbrev SNJ)
    algorithm described by Jaffe et al. in their 2020 paper. This class
    implements the abstract class DistanceSolver, and inherits the 'solve'
    method from DistanceSolver with the 'find_cherry' method implemented using a
    spectral method as in Jaffe et al. (2020). In particular, we iteratively
    join subsets of leaves into a cherry based on the second singular value of
    the RA matrix of the pair. We start with all the subsets being singleton
    sets with each leaf node, then compute the second singular values of the RA
    matrices where A = S U T, S and T are distinct subsets of leaves, and RA is
    the matrix with rows indexed by elements of A and columns indexed by
    elements of A 's complement, and entries are R(i, j).  These values are
    stored in a lambda matrix (rows and columns are subsets, entries are second
    singular values of RA) to avoid recalculating values.  Finally we find the
    pair of (S, T) that gives the smallest second singular value for RA and
    replace S and T with S U T in the set of subsets, thereby joining S and T
    into a cherry. Repeat this process until there are only two remaining
    subsets. The runtime is O(n^4), but in reality will perform better than
    this.

    Args:
        similarity_function: A function by which to compute the similarity map,
            corresponding to the affinity R(i, j) in Jaffe et al.
            (2020). Note that this similarity score should be multiplicative,
            i.e. R(i, j) = R(i, k) * R(k, j) for k on the path from i to j on
            the lineage tree. By default we will use exp(-d(i,j)) where d is the
            metric given by weighted_hamming_distance from
            dissimilarity_functions without weights given.
        add_root: Whether or not to add an implicit root to the tree, i.e. a
            root with unmutated characters.
        prior_transformation: Function to use when transforming
            priors into weights
    """

    def __init__(
        self,
        similarity_function: Optional[
            Callable[
                [np.ndarray, np.ndarray, int, Dict[int, Dict[int, float]]],
                float,
            ]
        ] = dissimilarity_functions.exponential_negative_hamming_distance,
        # type: ignore
        add_root: bool = False,
        prior_transformation: str = "negative_log",
    ):
        self._implementation = "generic_spectral_nj"
        
        super().__init__(
            dissimilarity_function=similarity_function,
            add_root=add_root,
            prior_transformation=prior_transformation,
        )  # type: ignore
        
        self._similarity_map = None
        self._lambda_indices = None


    def get_dissimilarity_map(
        self, cassiopeia_tree: CassiopeiaTree, layer: Optional[str] = None
    ) -> pd.DataFrame:
        """Outputs the first lambda matrix.

        The lambda matrix is a k x k matrix indexed by the k subsets
        being considered at each step of the SNJ algorithm. The i, j th element
        is given by taking the second singular value of the RA matrix with A
        given by the union of the two sets indexed by i and j. This function
        generates the initial lambda matrix, where every subset is a singleton
        leaf node. This function has a runtime of O(n^3), with n the
        number of leaf nodes.

        Args:
            cassiopeia_tree: the CassiopeiaTree object passed into solve()
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.

        Returns:
            DataFrame object of the lambda matrix, but similar in structure to
                DistanceSolver's dissimilarity_map.
        """

        # get dissimilarity map and save it as private instance variable
        self._similarity_map = super().get_dissimilarity_map(
            cassiopeia_tree, layer
        )

        # prepare for first (pairwise) lambda matrix
        N = self._similarity_map.shape[0]
        node_names: np.ndarray = self._similarity_map.index.values
        self._lambda_indices = [[i] for i in range(N)]

        # generate the lambda matrix
        lambda_matrix_arr = np.zeros([N, N])
        for (j_idx, i_idx) in itertools.combinations(range(N), 2):

            svd2_val = self._compute_svd2(
                pair=(i_idx, j_idx), lambda_indices=self._lambda_indices
            )

            lambda_matrix_arr[i_idx, j_idx] = lambda_matrix_arr[
                j_idx, i_idx
            ] = svd2_val

        np.fill_diagonal(lambda_matrix_arr, np.inf)

        # convert array to dataframe
        lambda_matrix_df = pd.DataFrame(
            lambda_matrix_arr, index=node_names, columns=node_names
        )

        return lambda_matrix_df

    def find_cherry(self, dissimilarity_map: np.ndarray) -> Tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        With dissimilarity_map being the lambda matrix, this method finds the
        argmin pair of subsets of the lambda matrix. The lambda matrix
        consists of rows and columns indexed by subsets of leaves and with
        entries a_ij given by the second singular value of RA where A = S U T, S
        and T being the subsets of leaves corresponding to i and j respectively.
        If i = j, a_ij is set to 'np.inf'. The pair of samples to be
        joined into a cherry is given by the pair (i, j) with the smallest entry
        in the lambda matrix.

        Args:
            dissimilarity_matrix: Lambda matrix

        Returns:
            A tuple of integers representing rows in the
                dissimilarity matrix to join.
        """

        return np.unravel_index(
            np.argmin(dissimilarity_map, axis=None), dissimilarity_map.shape
        )

    def _compute_svd2(
        self, pair: Tuple[int, int], lambda_indices: List[List[int]]
    ) -> float:
        """Computes the second largest singular value for an RA matrix.
        
        From Jaffe et al., the RA matrix is the matrix given by taking
        rows indexed by elements of A, and columns indexed by elements of the
        complement of A. Entries are given by R(i, j), the similarity score
        between i and j. Here, our subset A is given by unioning the
        subsets given in 'pair'. The procedure takes O(n^3) time where n is the
        number of leaves. On average it should be more efficient because we are
        not computing the singular value decomposition of a whole nxn matrix, RA
        is smaller than nxn. 

        Args:
            pair: pair of indices i and j where i > j.
            lambda_indices: the list of subsets for
                which 'pair' refers to.

        Returns:
            The second largest singular value of the pair's RA matrix.
        """
        i_idx, j_idx = pair
        i_sub, j_sub = lambda_indices[i_idx], lambda_indices[j_idx]

        # get combined pair of subsets
        a_subset = [*i_sub, *j_sub]

        # get complement
        a_comp = lambda_indices.copy()
        a_comp.pop(i_idx)
        a_comp.pop(j_idx)
        a_comp_flat = list(itertools.chain.from_iterable(a_comp))

        # reconstruct RA matrix
        RA_matrix = self._similarity_map.values[np.ix_(a_subset, a_comp_flat)]

        # get second largest SVD if available, first if not.
        s = scipy.linalg.svd(RA_matrix, compute_uv=False, check_finite=False)
        if len(s) >= 2:
            svd2_val = s[1]
        else:
            svd2_val = 0

        return svd2_val

    def update_dissimilarity_map(
        self,
        similarity_map: pd.DataFrame,
        cherry: Tuple[str, str],
        new_node: str,
    ) -> pd.DataFrame:
        """Updates the lambda matrix using the pair of subsets from find_cherry.

        Args:
            similarity_map: lambda matrix to update
            cherry1: One of the children to join.
            cherry2: One of the children to join.
            new_node: New node name to add to the dissimilarity map

        Returns:
            An updated lambda matrix.
        """
        # get cherry nodes in index of lambda matrix
        i, j = (
            np.where(similarity_map.index == cherry[0])[0][0],
            np.where(similarity_map.index == cherry[1])[0][0],
        )

        # modify names
        node_names = similarity_map.index.values
        node_names = np.concatenate((node_names, [new_node]))
        boolmask = np.ones((len(node_names),), bool)
        boolmask[[i, j]] = False
        node_names = node_names[boolmask]

        # modify indices
        self._lambda_indices.append(
            [*self._lambda_indices[i], *self._lambda_indices[j]]
        )
        self._lambda_indices.pop(max(i, j))
        self._lambda_indices.pop(min(i, j))

        # new lambda indices
        N = len(self._lambda_indices)

        if N <= 2:
            return pd.DataFrame(
                np.zeros([N, N]), index=node_names, columns=node_names
            )

        # get the old lambda matrix
        lambda_matrix_arr = similarity_map.drop(
            index=[cherry[0], cherry[1]], columns=[cherry[0], cherry[1]]
        ).values

        # add new col + row to lambda matrix
        new_row = np.array([0.0] * (N - 1))
        lambda_matrix_arr = np.vstack(
            (lambda_matrix_arr, np.atleast_2d(new_row))
        )
        new_col = np.array([0.0] * N)
        lambda_matrix_arr = np.array(
            np.hstack((lambda_matrix_arr, np.atleast_2d(new_col).T))
        )

        # compute new SVDs
        i_idx = N - 1
        for j_idx in range(i_idx):
            svd2_val = self._compute_svd2(
                pair=(i_idx, j_idx), lambda_indices=self._lambda_indices
            )

            lambda_matrix_arr[i_idx, j_idx] = lambda_matrix_arr[
                j_idx, i_idx
            ] = svd2_val

        np.fill_diagonal(lambda_matrix_arr, np.inf)

        # regenerate lambda matrix
        lambda_matrix_df = pd.DataFrame(
            lambda_matrix_arr, index=node_names, columns=node_names
        )

        return lambda_matrix_df

    def setup_root_finder(self, cassiopeia_tree: CassiopeiaTree) -> None:
        """Gives the implicit rooting strategy for the SNJ Solver.

        By default, the SpectralNeighborJoining algorithm returns an
        unrooted tree.  To root this tree, an implicit root of all zeros is
        added to the character matrix. Then, the dissimilarity map is
        recalculated using the updated character matrix. If the tree already
        has a computed dissimilarity map, only the new similarities are
        calculated. See 'setup_root_finder' in NeighborJoiningSolver.

        Args:
            cassiopeia_tree: Input CassiopeiaTree to `solve`
        """
        character_matrix = cassiopeia_tree.character_matrix.copy()
        rooted_character_matrix = character_matrix.copy()

        root = [0] * rooted_character_matrix.shape[1]
        rooted_character_matrix.loc["root"] = root
        cassiopeia_tree.root_sample_name = "root"
        cassiopeia_tree.character_matrix = rooted_character_matrix

        if self.dissimilarity_function is None:
            raise DistanceSolver.DistanceSolverError(
                "Please specify a dissimilarity function to add an implicit "
                "root, or specify an explicit root"
            )

        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()
        if dissimilarity_map is None:
            cassiopeia_tree.compute_dissimilarity_map(
                self.dissimilarity_function, self.prior_transformation
            )
        else:
            dissimilarity = {"root": 0}
            for leaf in character_matrix.index:
                weights = None
                if cassiopeia_tree.priors:
                    weights = solver_utilities.transform_priors(
                        cassiopeia_tree.priors, self.prior_transformation
                    )
                dissimilarity[leaf] = self.dissimilarity_function(
                    rooted_character_matrix.loc["root"].values,
                    rooted_character_matrix.loc[leaf].values,
                    cassiopeia_tree.missing_state_indicator,
                    weights,
                )
            cassiopeia_tree.set_dissimilarity("root", dissimilarity)

        cassiopeia_tree.character_matrix = character_matrix

    def root_tree(
        self, tree: nx.Graph, root_sample: str, remaining_samples: List[str]
    ) -> nx.DiGraph:
        """Assigns a node as the root of the solved tree.

        Finds a location on the tree to place a root and converts the general
        graph to a directed graph with respect to that root.

        Args:
            tree: Networkx object representing the tree topology
            root_sample: Sample to treat as the root
            remaining_samples: The last two unjoined nodes in the tree

        Returns:
            A rooted tree
        """
        tree.add_edge(remaining_samples[0], remaining_samples[1])

        rooted_tree = nx.DiGraph()
        for e in nx.dfs_edges(tree, source=root_sample):
            rooted_tree.add_edge(e[0], e[1])

        return rooted_tree
