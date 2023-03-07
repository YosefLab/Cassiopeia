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
# fitness_inference.py
# provides a class that runs the fitness inference on a given tree. It inherits the
# class survival_gen_fun from solve_survival. Given a tree, it uses the same message
# passing framework used in ancestral.py to propagate fitness up and down the tree.
# It ultimately calculates the posterior fitness distribution of each internal and
# external nodes, as well as the mean posterior fitness and its variance.
#
#########################################################################################

import numpy as np

from .solve_survival import *

verbose = False


class fitness_inference(survival_gen_func):
    """
    class that generates a biopython tree dressed with ancestral fitness distributions
    """

    def __init__(
        self,
        eps_branch_length=1e-7,
        D=0.2,
        fit_grid=None,
        samp_frac=0.01,
        mem=1.0,
    ):
        """
        keyword arguments
        pseudo_branch_length -- small value to be added to each branch
                                to avoid zero branch_length exceptions
        D                    -- diffusion constant parameterizing how fitness changes over time
        fit_grid             -- discrete fitness grid on which all fitness distribution are calculated
        """
        if fit_grid is None:
            self.nstates = 49  # length of the state vector (# of fitness bins)
            self.fitness_grid = np.linspace(
                -4, 8, self.nstates
            )  # fitness range in standard deviations
        else:
            self.nstates = len(fit_grid)
            self.fitness_grid = np.array(fit_grid)
        self.boundary_layer = 4
        self.dfit = self.fitness_grid[1] - self.fitness_grid[0]

        self.pseudo_branch_length = eps_branch_length
        self.D = D  # diffusion constant of individual lineages.
        self.v = 1  # velocity = sigma^2 of the wave
        self.sampling_fraction = samp_frac  # sampling fraction
        self.memory = mem
        # make set-up propagator and integrate the generating function on t = [0,20]
        survival_gen_func.__init__(self, self.fitness_grid)
        self.integrate_phi(
            self.D,
            self.sampling_fraction,
            np.array(
                np.linspace(0, 10, 201).tolist()
                + np.linspace(10, 200, 20).tolist()
            ),
        )

    def set_tree(self, tree, time_scale=None):
        """
        initiializes the tree and set the tree up for fitness inference
        required argument:
        tree -- a biopython tree. This tree will be modified in place.
        keyword argument:
        time_scale  -- the unit of time and sensitively affects the propagation of fitness
        """
        self.T = tree
        self.T.ladderize()

        # record the ladder rank of all terminal nodes and add as an attribute to the nodes
        self.terminals = self.T.get_terminals()
        for ci, c in enumerate(self.terminals):
            c.ladder_rank = ci

        # add the time to present to all nodes
        self.non_terminals = self.T.get_nonterminals()
        self.depths = self.T.root.depths()
        self.time_to_present = {}
        for node in self.T.get_terminals():
            self.time_to_present[node] = 0
        for node in self.T.get_nonterminals():
            self.time_to_present[node] = (
                np.mean([self.depths[tnode] for tnode in node.get_terminals()])
                - self.depths[node]
            )

        # guess_time_scale returns an estimate of the pair coalescence time.
        # the pair coalescent time should be roughly v/D
        if time_scale:
            self.time_scale = time_scale
        else:
            self.time_scale = self.guess_time_scale()

        # dress each internal node with a probability vector for the ancestral fitness state
        for leaf in self.terminals + self.non_terminals:
            leaf.prob = self.get_state_array()

        # there is no parental information to the root (the root node is artificial)
        # hence init the message with a gaussian
        self.T.root.down_message = np.exp(-0.5 * self.fitness_grid**2)
        self.T.root.down_polarizer = 0

    def guess_time_scale(self):
        """
        returns the mean pair coalescent time rescaled with D/v
        this should roughly yield the inverse scale of amplification
        """
        max_depth = max(self.depths.values())
        random_pairs = (
            np.random.randint(len(self.terminals), size=50),
            np.random.randint(len(self.terminals), size=50),
        )
        pair_times = []
        for ci1, ci2 in zip(random_pairs[0], random_pairs[1]):
            c1 = self.terminals[ci1]
            c2 = self.terminals[ci2]
            # get the path from the root to the leaves
            p1 = [self.T.root] + self.T.root.get_path(c1)
            p2 = [self.T.root] + self.T.root.get_path(c2)
            # go through the path and check when the path diverge
            for si, (s1, s2) in enumerate(zip(p1, p2)):
                if s1 != s2:
                    break
            # add the distances from the forking point to the leafs - this is exactly T_2
            pair_times.append(
                min(p1[si - 1].distance(c1), p2[si - 1].distance(c2))
            )

        # calculate the mean pairwise coalescent time and multiply
        # by D/v to convert to 1/\sigma
        return np.mean(pair_times) * self.D / self.v

    def get_state_array(self):
        """
        provides a unified function that returns an all one array
        for messages and probabilities of the right size
        """
        return np.ones(self.nstates)

    def propagator(self, t1, t2):
        """
        input: initial state, time interval
        returns the solution of the fitness evolution
           - note that the offspring fitness is in dimension 0
           - the ancestor fitness coordinate is in dimension 1
           - the first self.boundary_layer and last self.boundary_layer
             values for the offspring fitness are 0
        """
        b = self.boundary_layer
        sol = np.zeros((self.nstates - 2 * b, self.nstates))
        sol = self.integrate_prop(
            self.D, self.sampling_fraction, self.fitness_grid[b:-b], t1, t2
        )[-1]

        return sol

    def normalize(self, clade):
        """
        normalize the distribution of states at each position such that the sum equals one
        """
        clade.prob /= np.sum(clade.prob)

        if np.isnan(np.sum(clade.prob[:])):
            print(
                (
                    "encountered nan in ancestral fitness inference in clade ",
                    clade.name,
                )
            )
            print((np.isnan(clade.prob).nonzero()))

    def log_normalize(self, clade):
        """
        convert the unnormalized and logarithmic probabilites array to linear and then
        normalize the distribution of states such that the sum equals one
        """
        # substract the maximum value in each column
        clade.prob -= np.max(clade.prob, axis=0)
        # exponentiate
        clade.prob = np.exp(clade.prob)
        # normalize
        self.normalize(clade)

    def calc_down_polarizers(self, clade):
        """
        calculate the polarizers that are passed on to the children
        input: clade for which these are to calculated
        """
        # print "calc_down_msg", clade
        if clade.is_terminal():
            # nothing to be done for terminal nodes
            return
        else:
            # else, loop over children and calculate the message for each of the children
            for child in clade.clades:
                child.down_polarizer = clade.down_polarizer
                for child2 in clade.clades:
                    if child2 != child:
                        child.down_polarizer += child2.up_polarizer

                bl = (
                    child.branch_length + self.pseudo_branch_length
                ) / self.time_scale
                child.down_polarizer *= np.exp(-bl / self.memory)
                child.down_polarizer += self.memory * (
                    1 - np.exp(-bl / self.memory)
                )
                # do recursively for all children
            for child in clade.clades:
                self.calc_down_polarizers(child)

    def calc_down_messages(self, clade):
        """
        calculate the messages that are passed on to the children
        input: clade for which these are to calculated
        """
        # print "calc_down_msg", clade
        if clade.is_terminal():
            # nothing to be done for terminal nodes
            return
        else:
            b = self.boundary_layer
            # else, loop over children and calculate the message for each of the children
            for child in clade.clades:
                child.down_message = np.zeros_like(self.fitness_grid)
                # initialize with the message coming from the parent
                clade.prob[:] = np.log(clade.down_message)
                for child2 in clade.clades:
                    if child2 != child:
                        # multiply with the down message from each of the children, but skip child 1
                        clade.prob += np.log(child2.up_message)

                # normalize, adjust for modifications along the branch, and save.
                self.log_normalize(clade)
                # node that the propagator is only calculated for offpring fitness self.boundary
                # away from the fitness grid. the ancestor fitness (y) is repeated hence
                # self.nstates-2*b times
                PY = np.repeat(
                    [clade.prob], self.nstates - 2 * self.boundary_layer, axis=0
                )
                child.down_message[b:-b] = (child.propagator * PY).sum(axis=1)
                child.down_message[
                    child.down_message < non_negativity_cutoff
                ] = non_negativity_cutoff
            # do recursively for all children
            for child in clade.clades:
                self.calc_down_messages(child)

    def calc_up_polarizers(self, clade):
        """
        input: clade whose up_polarizer (to parents) is to be calculated
        """
        clade.up_polarizer = 0
        # add the trees of the children shifted by shift bins
        for child in clade.clades:
            self.calc_up_polarizers(child)
            clade.up_polarizer += child.up_polarizer
        bl = (clade.branch_length + self.pseudo_branch_length) / self.time_scale
        clade.up_polarizer *= np.exp(-bl / self.memory)
        clade.up_polarizer += self.memory * (1 - np.exp(-bl / self.memory))

    def calc_up_messages(self, clade):
        """
        input: clade whose up_message (to parents is to be calculated
        """
        # print "calc_up_msg", clade
        b = self.boundary_layer
        clade.up_message = np.zeros(self.nstates)
        clade.prob[:] = 0
        # add the trees of the children shifted by shift bins
        for child in clade.clades:
            self.calc_up_messages(child)
            clade.prob += np.log(child.up_message)
        self.log_normalize(clade)
        if clade.is_terminal:
            eps_branch = self.pseudo_branch_length / self.time_scale
        else:
            eps_branch = self.pseudo_branch_length / self.time_scale
        clade.propagator = self.propagator(
            self.time_to_present[clade] / self.time_scale,
            (self.time_to_present[clade] + clade.branch_length)
            / self.time_scale
            + eps_branch,
        )
        try:
            shift = int(np.floor(clade.fitness_shift / self.dfit))
            if shift > 0:
                print(shift)
                clade.propagator[shift:, :] = clade.propagator[:-shift, :]
                clade.propagator[:shift, :] = non_negativity_cutoff
        except:
            if verbose:
                print("fitness shift not found!")

        PX = np.repeat([clade.prob[b:-b]], self.nstates, axis=0).T
        clade.up_message = (clade.propagator * PX).sum(axis=0)
        clade.up_message[
            clade.up_message < non_negativity_cutoff
        ] = non_negativity_cutoff

    def calc_marginal_probabilities(self, clade):
        """
        calculate the marginal probabilities by multiplying all incoming messages
        """
        clade.prob[:] = np.log(clade.down_message)
        for child in clade.clades:
            clade.prob[:] += np.log(child.up_message)

        # normalize and continue for all children
        self.log_normalize(clade)
        # print clade.name, np.max(1.0-np.max(clade.prob, axis=1))
        for child in clade.clades:
            self.calc_marginal_probabilities(child)

    def calc_marginal_polarizers(self, clade):
        """
        calculate the marginal probabilities by multiplying all incoming messages
        """
        clade.polarizer = clade.down_polarizer
        for child in clade.clades:
            clade.polarizer += child.up_polarizer

        # repeat for all children
        for child in clade.clades:
            self.calc_marginal_polarizers(child)

    def calc_mean_and_variance(self, clade):
        """
        recursively calculate the mean and variance of the fitness distribution at the node
        """
        clade.mean_fitness = np.sum(self.fitness_grid * clade.prob)
        clade.var_fitness = (
            np.sum(self.fitness_grid**2 * clade.prob)
            - clade.mean_fitness**2
        )

        # repeat for all children
        if clade.is_terminal() == False:
            for child in clade.clades:
                self.calc_mean_and_variance(child)

    def infer_ancestral_fitness(self):
        """
        given the initialized instance, calculates the ancestral fitness distribution
        and the marginal probabilities for each position at each internal node.
        """
        self.calc_up_messages(self.T.root)
        self.calc_down_messages(self.T.root)
        self.calc_marginal_probabilities(self.T.root)
        self.calc_mean_and_variance(self.T.root)

    def calculate_polarizers(self, mem=None):
        """
        given the initialized instance, calculates the ancestral fitness distribution
        and the marginal probabilities for each position at each internal node.
        """
        if mem is not None:
            self.memory = mem
        self.calc_up_polarizers(self.T.root)
        self.calc_down_polarizers(self.T.root)
        self.calc_marginal_polarizers(self.T.root)
