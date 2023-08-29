"""
This file stores a subclass of CassiopeiaSolver, the FastTreeSolver. This is 
a wrapper around optimized maximum likelihood algorithms implemented in FastTree
2.1 (http://www.microbesonline.org/fasttree/#Matrix). Since character states
need to be converted to amino acids, this solver only works if there are less 
than 20 character states.
"""

import abc
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import os
import tempfile
import ete3
import subprocess
import pandas as pd

from cassiopeia.data import CassiopeiaTree, utilities
from cassiopeia.mixins import DistanceSolverError
from cassiopeia.solver import CassiopeiaSolver, solver_utilities

class FastTreeSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    Distance based solver class.

    This solver serves as a wrapper around FastTree 2.

    Args:
        mutation_rate: Estimated site mutation rate per cell division. This is used to
            generate a transition matrix if transition_matrix is None.
        number_of_states: Number of character states. This is used to generate a
            transition matrix if transition_matrix is None.
        add_root: Whether or not to add an implicit root the tree. Only
            pertinent in algorithms that return an unrooted tree, by default
            (e.g. Neighbor Joining). Will not override an explicitly defined
            root, specified by the 'root_sample_name' attribute in the
            CassiopeiaTree.
        initial_tree_solver: Optional algorithm for generating the initial tree.
            Must be a subclass of CassiopeiaSolver.
        transition_matrix: Optional transition matrix in Whelan And Goldman format
            where amino-acids correspond to the following integer edit states:
                0: A
                1: R
                2: N
                3: D
                4: C
                5: Q
                6: E
                7: G
                8: H
        maximum_likelihood: Whether or not to use maximum likelihood when
            inferring the tree.
        minimum_evolution: Whether or not to use minimum evolution NNIs
            and SPRs when inferring the tree.
    """

    def __init__(
        self,
        mutation_rate: float = .1,
        number_of_states: int = 9,
        add_root: bool = False,
        initial_tree_solver: Optional[CassiopeiaSolver.CassiopeiaSolver] = None,
        transition_matrix: Optional[pd.DataFrame] = None,
        maximum_likelihood: bool = False,
        minimum_evolution: bool = False,
    ):

        super().__init__()

        # Check that mutation rate is between 0 and 1
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        self.mutation_rate = mutation_rate
        # Check that number of states is less than 20
        if number_of_states > 20:
            raise ValueError("Number of character states must be less than 20")
        self.number_of_states = number_of_states
        # Define mapping from itegers to amino acids
        self.num_to_aa = {0: 'A',1: 'R',2: 'N',3: 'D',4: 'C',5: 'Q',6: 'E',7: 'G',8: 'H'
                          ,9: 'I',10: 'L',11: 'K',12: 'M',13: 'F',14: 'P',15: 'S',16: 'T',17: 'W',
                          18: 'Y',19: 'V',-1: '-'}
        self.aas = list(self.num_to_aa.values())[0:20]
        self.add_root = add_root
        self.initial_tree_solver = initial_tree_solver
        if not transition_matrix is None:
            # Check that transition matrix is in Whelan and Goldman format
            self._check_transition_matrix(transition_matrix)
            self.transition_matrix = transition_matrix
        else:
            self.transition_matrix = self._setup_transition_matrix(self.mutation_rate, self.number_of_states)
        self.edit_distance = self._setup_edit_distance(self.number_of_states)
        self.maximum_likelihood = maximum_likelihood
        self.minimum_evolution = minimum_evolution

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """
        This solver serves as a wrapper around FastTree2. If initial_tree_solver
        is not None, the initial tree is generated using the specified solver.

        Args:
            cassiopeia_tree: CassiopeiaTree object to be populated
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
            collapse_mutationless_edges: Indicates if the final reconstructed
                tree should collapse mutationless edges based on internal states
                inferred by Camin-Sokal parsimony. In scoring accuracy, this
                removes artifacts caused by arbitrarily resolving polytomies.
        """

        # check that number of states matches transition matrix
        detected_states = cassiopeia_tree.character_matrix.values[cassiopeia_tree.character_matrix.values != -1]
        if self.number_of_states != len(pd.unique(detected_states.ravel())):
            raise ValueError("Number of character states does not match transition matrix")

        with tempfile.TemporaryDirectory() as temp_dir:

            # write transition matrix to file
            transition_matrix_path = os.path.join(temp_dir, "trans.mat")
            self._save_transition_matrix(self.transition_matrix, transition_matrix_path)
            # write character matrix to file
            character_matrix_path = os.path.join(temp_dir, "character_matrix.fasta")
            self._save_character_matrix_fasta(cassiopeia_tree.character_matrix, character_matrix_path)
            # write edit distance to file
            edit_distance_path = os.path.join(temp_dir, "edit_distance")
            self._save_edit_distance(self.edit_distance, edit_distance_path)
            # get tree path
            tree_path = os.path.join(temp_dir, "tree.nwck")

            # set additional parameters
            additional_params = ""
            if not self.maximum_likelihood:
                additional_params += " -noml"
            if not self.minimum_evolution:
                additional_params += " -nome"

            # if initial tree solver is not None run FastTree on initial tree
            if not self.initial_tree_solver is None:
                initial_tree = cassiopeia_tree.copy()
                self.initial_tree_solver.solve(initial_tree, layer=layer,collapse_mutationless_edges = False)
                initial_tree_path = os.path.join(temp_dir, "tree.nwck")
                initial_tree_path = "/lab/solexa_weissman/PEtracing_shared/PETracer_Analysis/Simulation/output/initial_tree.nwck"
                with open(initial_tree_path, "w") as f:
                    f.write(utilities.to_newick(initial_tree.get_tree_topology()))
                command = (f". ~/.bashrc && FastTree -nosupport -trans {transition_matrix_path}"
                        + additional_params + 
                        f" -matrix {edit_distance_path}"
                        f" -intree {initial_tree_path}"
                        f" {character_matrix_path} > {tree_path}")
                
            # else run FastTree
            else:
                command = (f". ~/.bashrc && FastTree -nosupport -trans {transition_matrix_path}"
                        + additional_params +
                        f" -matrix {edit_distance_path}"
                        f" {character_matrix_path} > {tree_path}")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(stdout.decode("utf-8"))
            T = ete3.Tree(tree_path, format=1)

            # remove temporary files
            os.remove(transition_matrix_path)
            os.remove(character_matrix_path)
            os.remove(tree_path)

        # remove root from character matrix before populating tree
        if (cassiopeia_tree.root_sample_name
            in cassiopeia_tree.character_matrix.index):
            cassiopeia_tree.character_matrix = (
                cassiopeia_tree.character_matrix.drop(
                    index=cassiopeia_tree.root_sample_name
                )
            )

        # root tree
        if (self.add_root):
            T.set_outgroup(T.get_midpoint_outgroup())
            root = ete3.TreeNode(name="root")
            root.add_child(T)
    
        # populate tree
        T.ladderize(direction=1)
        cassiopeia_tree.populate_tree(T,layer=layer)
        print(stderr.decode("utf-8"))

        # collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(
                infer_ancestral_characters=True
            )

    def _setup_transition_matrix(
        self, mutation_rate: float, number_of_states: int
    ) -> None:
        """Sets up the solver.

        Creates the transition_matrix using the mutation rate and number 
        of character states. Assumes that all transitions to valid states 
        are equally likely.

        Args:
            mutation_rate: Estimated site mutation rate per cell division.
            number_of_states: Number of character states.
        """
        # Tolerance
        tol = 1e-3 
        # Initialize transition matrix
        Q = np.zeros((20,20))
        # Probability of staying in unedited state
        Q[0,0] = np.log(1-mutation_rate)
        # Low probability of transition from unedited to invalid state
        Q[number_of_states:20,0] = np.ones(20 - number_of_states)*np.log(1+tol)
        # Probability of transitioning from unedited to edited
        Q[1:number_of_states,0] = np.ones(number_of_states-1)*np.sum(Q[:,0])*-1/(number_of_states-1)
        # Low probability of transitioning between edited states
        Q[0:20,1:20] = np.ones((20,19))*np.log(1-tol)*-1/19
        # High probability of staying in edited state
        for i in range(1,20):
            Q[i,i] = np.log(1-tol)
        # Uniform stationary distribution
        pi = np.ones(20) * tol
        pi[0:number_of_states] = np.ones(number_of_states)
        pi = pi / pi.sum()
        # Create dataframe
        transition_matrix = pd.DataFrame(Q)
        transition_matrix[20] = pi
        transition_matrix.columns = pd.Index(self.aas + ["*"])
        transition_matrix.index = pd.Index(self.aas)
        return transition_matrix

    def _save_transition_matrix(
        self, transition_matrix: pd.DataFrame,output_file: str
    ) -> None:
        with open(output_file, 'w') as f:
            f.write("\t".join(self.aas + ["*"])+"\n")
            for i in range(20):
                f.write(f"{transition_matrix.index[i]}")
                for j in range(0,21):
                    f.write(f"\t{transition_matrix.iloc[i,j]}")
                f.write("\n")

    def _check_transition_matrix(
            self, transition_matrix: pd.DataFrame
    ) -> None:
        """Checks if the transition matrix is in Whelan and Goldman format.

        Args:
            transition_matrix: Transition matrix to check.
        """
        assert isinstance(transition_matrix, pd.DataFrame) # is dataframe
        assert transition_matrix.shape == (20, 21) # is of shape (20, 21)
        Q = transition_matrix.iloc[:,:-1].values
        pi = transition_matrix.iloc[:,-1].values
        assert np.all(pi > 0)  # each element of pi must be positive
        assert np.abs(np.sum(pi) - 1) < 1e-3  # sum of pi must be 1
        assert np.all(np.diagonal(Q) < 0)  # each element on the diagonal of Q is negative
        assert np.all(Q >= 0) or np.all(Q[Q != np.diagonal(Q)] >= 0)  # each element off the diagonal of Q is non-negative
        assert np.abs(np.dot(Q, pi).sum()) < 1e-3  # Q · pi = 0
        assert np.all(np.isclose(Q.sum(axis=0), 0, atol = 1e-3))  # the sum of each column in Q is zero

    def _setup_edit_distance(
        self, number_of_states: int
    ) -> Dict:
        distances = np.random.uniform(-.5,.5,(20,20))
        distances = distances @ distances.T
        used = np.ones((number_of_states,number_of_states)) * 2
        used[:,0] = np.ones(number_of_states) * 1
        used[0,:] = np.ones(number_of_states) * 1
        np.fill_diagonal(used, 0)
        distances[0:number_of_states,0:number_of_states] = used
        # Get eigenvectors and eigenvalues
        eigenval, eigenvectors = np.linalg.eig(distances)
        eigeniv = np.real(np.linalg.inv(eigenvectors))
        eigenval = np.real(eigenval)
        edit_distance = {"distances":distances,"inverses":eigeniv,"eigenvalues":eigenval}
        return edit_distance

    def _save_character_matrix_fasta(
            self, character_matrix: pd.DataFrame,output_file: str
        ) -> None:
        with open(output_file, 'w') as f:
            for i in range(character_matrix.shape[0]):
                aa_seq = ''.join(character_matrix.iloc[i].map(self.num_to_aa))
                f.write(f">{character_matrix.index[i]}\n{aa_seq}\n")

    def _save_edit_distance(
            self, edit_distance: Dict,output_file: str
        ) -> None:
        with open(output_file + ".distances", 'w') as f:
            f.write("\t".join(self.aas)+"\n")
            for i in range(20):
                f.write(f"{self.aas[i]}")
                mat = edit_distance["distances"]
                for j in range(20):
                    f.write(f"\t{mat[i,j]}")
                f.write("\n")
        with open(output_file + ".inverses", 'w') as f:
            f.write("\t".join(self.aas)+"\n")
            for i in range(20):
                f.write(f"{self.aas[i]}")
                mat = edit_distance["inverses"]
                for j in range(20):
                    f.write(f"\t{mat[i,j]}")
                f.write("\n")
        with open(output_file + ".eigenvalues", 'w') as f:
            vec = edit_distance["eigenvalues"]
            for i in range(20):
                f.write(f"{vec[i]}\n")
            f.write("\n")