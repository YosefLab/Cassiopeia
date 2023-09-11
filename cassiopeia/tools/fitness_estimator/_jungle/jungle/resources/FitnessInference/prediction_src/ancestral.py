#########################################################################################
#
# author: Richard Neher
# email: richard.neher@tuebingen.mpg.de
#
# Reference: Richard A. Neher, Colin A Russell, Boris I Shraiman.
#            "Predicting evolution from the shape of genealogical trees"
#
#########################################################################################
#
# ancestral.py
# provides a class called ancestral_sequences that does maximum likelihood (independently
# for each site) estimation of ancestral sequences given a tree. The algorithm is framed
# as a messaging passing procedure, which is closely related to dynamics programming.
#
#########################################################################################


import copy

import numpy as np
from Bio import Phylo, Seq


class ancestral_sequences:
    """
    class that generates a biopython tree dressed with ancestral sequences
    and the marginal probabilities of different states in the tree
    """

    def __init__(
        self,
        tree,
        aln,
        alphabet="ACGT",
        sub_matrix=None,
        eps_branch_length=1e-7,
        copy_tree=True,
    ):
        """
        arguments:
        tree  -- a biopython tree
        aln   -- a biopython alignment with the same names as the terminal nodes

        keyword arguments:
        alphabet   -- allows character states
        sub_matrix -- substitution matrix. defaults to flat matrix
        eps_branch_length -- minimal branch length to prevent division by zero exception
        copy_tree -- if true, a new tree object is constructed
        """
        if copy_tree:
            self.T = copy.deepcopy(tree)
        else:
            self.T = tree
        self.alphabet = np.array(list(alphabet))
        self.nstates = self.alphabet.shape[0]
        self.seq_len = aln.get_alignment_length()
        self.pseudo_branch_length = eps_branch_length

        # construct substitution matrix if not provided
        if sub_matrix:
            self.sub_matrix = np.array(sub_matrix)
        else:
            # every mutation equally likely. subtract diagonal, normalize to rate 1.
            # this matrix is symmetric
            self.sub_matrix = (
                (
                    np.ones((self.nstates, self.nstates))
                    - self.nstates * np.eye(self.nstates)
                )
                * 1.0
                / (self.nstates - 1)
            )
            self.calc_eigendecomp()

        seq_names = [seq.id for seq in aln]
        for leaf in self.T.get_terminals():
            if leaf.name in seq_names:
                leaf.seq = aln[seq_names.index(leaf.name)].seq
                leaf.prob = self.get_state_array()
                # convert to a numpy array for convenient slicing
                tmp_seq_array = np.array(leaf.seq)
                # code the sequences as a 0/1 probability matrix  (redundant but convenient)
                for ni in range(self.nstates):
                    leaf.prob[:, ni] = tmp_seq_array == self.alphabet[ni]
            else:
                print(
                    (
                        "ancestral sequences: Leaf "
                        + leaf.name
                        + " has no sequence"
                    )
                )
        self.biopython_alphabet = leaf.seq.alphabet
        # dress each internal node with a probability vector for the ancestral state
        for node in self.T.get_nonterminals():
            node.prob = self.get_state_array()
        # there is no parental information to the root (the root node is artificial)
        # hence init the message with ones
        self.T.root.down_message = self.get_state_array()

    def get_state_array(self):
        """
        provide a unified function that returns an all one array
        for messages and probabilities of the right size
        """
        return np.ones((self.seq_len, self.nstates))

    def calc_eigendecomp(self):
        """
        calculates eigenvalues and eigenvectors.
        note that this assumes that the substitution matrix is symmetric
        """
        self.evals, self.evecs = np.linalg.eigh(self.sub_matrix)

    def calc_state_probabilites(self, P, t):
        """
        input: initial state, time interval
        return the solution of the character evolution equaiton
        """
        transition_matrix = np.dot(
            self.evecs,
            np.dot(
                np.diag(np.exp((t + self.pseudo_branch_length) * self.evals)),
                self.evecs.T,
            ),
        )
        return np.dot(P, transition_matrix.T)

    def normalize(self, clade):
        """
        normalize the distribution of states at each position such that the sum equals one
        """
        clade.prob /= np.repeat(
            np.array([np.sum(clade.prob, axis=1)]).T, self.nstates, axis=1
        )

        if np.isnan(np.sum(clade.prob[:])):
            print(
                ("encountered nan in ancestral inference in clade ", clade.name)
            )
            print((np.isnan(clade.prob).nonzero()))

    def log_normalize(self, clade):
        """
        convert the unnormalized and logarithmic probabilites array to linear and then normaliz
        normalize the distribution of states at each position such that the sum equals one
        """
        # substract the maximum value in each column
        clade.prob -= np.repeat(
            np.array([np.max(clade.prob, axis=1)]).T, self.nstates, axis=1
        )
        # exponentiate
        clade.prob = np.exp(clade.prob)
        # normalize
        self.normalize(clade)

    def calc_up_messages(self, clade):
        """
        recursively calculates the messages passed on the parents of each node
        input: clade whose up_messsage is to be calculated
        """
        if clade.is_terminal():
            # if clade is terminal, the sequence is fix and we can emit the state probabilities
            clade.up_message = self.calc_state_probabilites(
                clade.prob, clade.branch_length
            )
            # print "down clade", clade.name, 'min:', np.min(clade.up_message)
            clade.up_message[clade.up_message < 1e-30] = 1e-30
        else:
            # otherwise, multiply all down messages from children, normalize and pass down
            clade.prob[:] = 0
            for child in clade.clades:
                self.calc_up_messages(child)
                clade.prob += np.log(child.up_message)

            self.log_normalize(clade)
            clade.up_message = self.calc_state_probabilites(
                clade.prob, clade.branch_length
            )
            # print "down clade", clade.name, 'min:', np.min(clade.up_message)
            clade.up_message[clade.up_message < 1e-30] = 1e-30

    def calc_down_messages(self, clade):
        """
        calculate the messages that are passed on to the children
        input calde for which these are to calculated
        """
        if clade.is_terminal():
            # nothing to be done for terminal nodes
            return
        else:
            # else, loop over children and calculate the message for each of the children
            for child in clade.clades:
                # initialize with the message comming from the parent
                clade.prob[:] = np.log(clade.down_message)
                for child2 in clade.clades:
                    if child2 != child:
                        # multiply with the down message from each of the children, but skip child 1
                        clade.prob += np.log(child2.up_message)

                # normalize, adjust for modifications along the branch, and save.
                self.log_normalize(clade)
                child.down_message = self.calc_state_probabilites(
                    clade.prob, child.branch_length
                )
                # print "up clade", clade.name, 'min:', np.min(child.down_message)
                child.down_message[child.down_message < 1e-30] = 1e-30
            # do recursively for all children
            for child in clade.clades:
                self.calc_down_messages(child)

    def calc_marginal_probabilities(self, clade):
        """
        calculate the marginal probabilities by multiplying all incoming messages
        """
        if clade.is_terminal():
            return
        else:
            clade.prob[:] = np.log(clade.down_message)
            for child in clade.clades:
                clade.prob += np.log(child.up_message)

            # normalize and continue for all children
            self.log_normalize(clade)
            # print clade.name, np.max(1.0-np.max(clade.prob, axis=1))
            for child in clade.clades:
                self.calc_marginal_probabilities(child)

    def calc_most_likely_sequences(self, clade):
        """
        recursively calculate the most likely sequences for each node
        """
        if clade.is_terminal():
            return
        else:
            clade.seq = Seq.Seq(
                "".join(self.alphabet[np.argmax(clade.prob, axis=1)]),
                alphabet=self.biopython_alphabet,
            )

            # repeat for all children
            for child in clade.clades:
                self.calc_most_likely_sequences(child)

    def calc_ancestral_sequences(self):
        """
        given the initialized instance, calculates the most likely ancestral sequences
        and the marginal probabilities for each position at each internal node.
        """
        self.calc_up_messages(self.T.root)
        self.calc_down_messages(self.T.root)
        self.calc_marginal_probabilities(self.T.root)
        self.calc_most_likely_sequences(self.T.root)
