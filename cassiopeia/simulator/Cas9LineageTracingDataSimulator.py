"""
A Cas9-based lineage tracing data simulator. This is a sublcass of the
LineageTracingDataSimulator that simulates the data produced from Cas9-based
technologies (e.g, as described in Chan et al, Nature 2019 or McKenna et al,
Science 2016). This simulator implements the method `overlay_data` which takes
in a CassiopeiaTree with edge lengths and overlays states onto cut-sites.
"""
import copy
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator.LineageTracingDataSimulator import (
    LineageTracingDataSimulator,
)


class Cas9LineageTracingDataSimulator(LineageTracingDataSimulator):
    """Simulates Cas9-based lineage tracing data.

    This subclass of `LineageTracingDataSimulator` implements the `overlay_data`
    function to simulate Cas9-based mutations onto a defined set of "cassette".
    These cassettes emulate the TargetSites described in Chan et al, Nature 2019
    and the GESTALT arrays described in McKenna et al, Science 2020 in which
    several Cas9 cut-sites are arrayed together. In the Chan et al technology,
    these cassettes are of length 3; in McKenna et al, these cassettes are of
    length 10.

    The class accepts several parameters that govern the Cas9-based tracer.
    First and foremost is the cutting rate, describing how fast Cas9 is able to
    cut. We model Cas9 cutting as an exponential process, parameterized by the
    specified mutation rate - specifically, for a lifetime `t` and parameter
    `lambda`, the expected probability of Cas9 mutation, per site, is
    `exp(-lambda * t)`.

    Second, the class accepts the architecture of the recorder - described by
    the size of the cassette (by default 3) and the number of cassettes. The
    resulting lineage will have (size_of_cassette * number_of_cassettes)
    characters. The architecture is important in the sense that it encodes
    the correlation between characters. This is critical in two ways: the
    first is with respect to silencing -- entire cassettes are lost from
    silencing, either transcriptional or stochastic events. The second is with
    respect to Cas9 resections, in which more than one Cas9 molecule introduces
    a cut to the same cassette. In this event, all cut sites intermediate will
    be observed as missing. In this simulation, we allow this event to occur
    if Cas9 molecules perform cuts on the same cassette at any point in a cell's
    lifetime. Importantly, setting the cassette length to 1 will remove any
    resection events due to Cas9 cutting, and will reduce the amount of
    transcriptionally silencing observed. This behavior can be manually turned
    off by setting `collapse_sites_on_cassette=False`, which will keep
    cuts that occur simultaneously on the same cassette as separate events,
    instead of causing a resection event.

    Third, the class accepts a state distribution describing the relative
    likelihoods of various indels. This is very useful, as it is typical that a
    handful of mutations are far likelier than the bulk of the possible
    mutations.

    Finally, the class accepts two types of silencing rates. The first is the
    heritable silencing rate which is a rare event in which an entire cassette
    is transcriptionally silenced and therefore not observed. The second type
    of silencing is a stochastic dropout rate which simulates the loss of
    cassettes due to the low sensitivity of the RNA-sequencing assay.

    The function `overlay_data` will operate on the tree in place and will
    specifically modify the data stored in the character attributes.

    Args:
        number_of_cassettes: Number of cassettes (i.e., arrays of target sites)
        size_of_cassette: Number of editable target sites per cassette
        mutation_rate: Exponential parameter for the Cas9 cutting rate. The user
            can either pass in a float, in which every site will mutate at
            the same rate, or a list with an entry for each cut site indicating
            potentially a different mutation rate for every site.
        state_generating_distribution: Distribution from which to simulate state
            likelihoods. This is only used if mutation priors are not
            specified to the simulator.
        number_of_states: Number of states to simulate
        mutation_priors: A mapping from state to probability that a user
            can specify. If this argument is not None, states will not be
            pulled from the state distribution.
        heritable_silencing_rate: Silencing rate for the cassettes, per node,
            simulating heritable missing data events.
        stochastic_silencing_rate: Rate at which to randomly drop out cassettes,
            to simulate dropout due to low sensitivity of assays.
        heritable_missing_data_state: Integer representing data that has gone
            missing due to a heritable event (i.e. Cas9 resection or heritable
            silencing).
        stochastic_missing_data_state: Integer representing data that has
            gone missing due to the stochastic dropout from single-cell assay
            sensitivity.
        random_seed: Numpy random seed to use for deterministic simulations.
            Note that the numpy random seed gets set during every call to
            `overlay_data`, thereby producing deterministic simulations every
            time this function is called.
        collapse_sites_on_cassette: Whether or not to collapse cuts that occur
            in the same cassette in a single iteration. This option only takes
            effect when `size_of_cassette` is greater than 1. Defaults to True.

    Raises:
        DataSimulatorError if assumptions about the system are broken.
    """

    def __init__(
        self,
        number_of_cassettes: int = 10,
        size_of_cassette: int = 3,
        mutation_rate: Union[float, List[float]] = 0.01,
        state_generating_distribution: Callable[
            [], float
        ] = lambda: np.random.exponential(1e-5),
        number_of_states: int = 100,
        state_priors: Optional[Dict[int, float]] = None,
        heritable_silencing_rate: float = 1e-4,
        stochastic_silencing_rate: float = 1e-2,
        heritable_missing_data_state: int = -1,
        stochastic_missing_data_state: int = -1,
        random_seed: Optional[int] = None,
        collapse_sites_on_cassette: bool = True,
    ):

        if number_of_cassettes <= 0 or not isinstance(number_of_cassettes, int):
            raise DataSimulatorError("Specify a positive number of cassettes.")
        if size_of_cassette <= 0 or not isinstance(size_of_cassette, int):
            raise DataSimulatorError(
                "Specify a positive number of cut-sites" " per cassette."
            )

        self.size_of_cassette = size_of_cassette
        self.number_of_cassettes = number_of_cassettes
        self.collapse_sites_on_cassette = collapse_sites_on_cassette

        if isinstance(mutation_rate, float):
            if mutation_rate < 0:
                raise DataSimulatorError(
                    "Mutation rate needs to be" " non-negative."
                )
            number_of_characters = size_of_cassette * number_of_cassettes
            self.mutation_rate_per_character = [
                mutation_rate
            ] * number_of_characters
        else:
            if len(mutation_rate) != (
                self.number_of_cassettes * self.size_of_cassette
            ):
                raise DataSimulatorError(
                    "Length of mutation rate array is not"
                    " the same as the number of characters."
                )

            if np.any(np.array(mutation_rate) < 0):
                raise DataSimulatorError(
                    "Mutation rate needs to be" " non-negative."
                )
            self.mutation_rate_per_character = mutation_rate[:]

        if state_priors is None:
            self.number_of_states = number_of_states
            self.state_generating_distribution = state_generating_distribution
        else:
            Z = np.sum([v for v in state_priors.values()])
            if not math.isclose(Z, 1.0):
                raise DataSimulatorError("Mutation priors do not sum to 1.")
            self.number_of_states = len(state_priors)
            self.state_generating_distribution = None

        self.mutation_priors = state_priors

        self.heritable_silencing_rate = heritable_silencing_rate
        self.stochastic_silencing_rate = stochastic_silencing_rate

        self.heritable_missing_data_state = heritable_missing_data_state
        self.stochastic_missing_data_state = stochastic_missing_data_state

        self.random_seed = random_seed

    def overlay_data(self, tree: CassiopeiaTree):
        """Overlays Cas9-based lineage tracing data onto the CassiopeiaTree.

        Args:
            tree: Input CassiopeiaTree
        """

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # create state priors if they don't exist.
        # This will set the instance's variable for mutation priors and will
        # use this for all future simulations.
        if self.mutation_priors is None:
            self.mutation_priors = {}
            probabilites = [
                self.state_generating_distribution()
                for _ in range(self.number_of_states)
            ]
            Z = np.sum(probabilites)
            for i in range(self.number_of_states):
                self.mutation_priors[i + 1] = probabilites[i] / Z

        number_of_characters = self.number_of_cassettes * self.size_of_cassette

        # initialize character states
        character_matrix = {}
        for node in tree.nodes:
            character_matrix[node] = [-1] * number_of_characters

        for node in tree.depth_first_traverse_nodes(tree.root, postorder=False):

            if tree.is_root(node):
                character_matrix[node] = [0] * number_of_characters
                continue

            parent = tree.parent(node)
            life_time = tree.get_time(node) - tree.get_time(parent)

            character_array = character_matrix[parent]
            open_sites = [
                c
                for c in range(len(character_array))
                if character_array[c] == 0
            ]

            new_cuts = []
            for site in open_sites:
                mutation_rate = self.mutation_rate_per_character[site]
                mutation_probability = 1 - (np.exp(-life_time * mutation_rate))

                if np.random.uniform() < mutation_probability:
                    new_cuts.append(site)

            # collapse cuts that are on the same cassette
            cuts_remaining = new_cuts
            if self.collapse_sites_on_cassette and self.size_of_cassette > 1:
                character_array, cuts_remaining = self.collapse_sites(
                    character_array, new_cuts
                )

            # introduce new states at cut sites
            character_array = self.introduce_states(
                character_array, cuts_remaining
            )

            # silence cassettes
            silencing_probability = 1 - (
                np.exp(-life_time * self.heritable_silencing_rate)
            )
            character_array = self.silence_cassettes(
                character_array,
                silencing_probability,
                self.heritable_missing_data_state,
                self.heritable_missing_data_state,
            )

            character_matrix[node] = character_array

        # apply stochastic silencing
        for leaf in tree.leaves:
            character_matrix[leaf] = self.silence_cassettes(
                character_matrix[leaf],
                self.stochastic_silencing_rate,
                self.stochastic_missing_data_state,
                self.heritable_missing_data_state,
            )

        tree.set_all_character_states(character_matrix)

    def collapse_sites(
        self, character_array: List[int], cuts: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Collapses cassettes.

        Given a character array and a new set of cuts that Cas9 is inducing,
        this function will infer which cuts occur within a given cassette and
        collapse the sites between the two cuts.

        Args:
            character_array: Character array in progress
            cuts: Sites in the character array that are being cut.

        Returns:
            The updated character array and sites that are not part of a
                cassette collapse.
        """

        updated_character_array = character_array.copy()

        cassettes = self.get_cassettes()
        cut_to_cassette = np.digitize(cuts, cassettes)
        cuts_remaining = []
        for cassette in np.unique(cut_to_cassette):
            cut_indices = np.where(cut_to_cassette == cassette)[0]
            if len(cut_indices) > 1:
                sites_to_collapse = np.array(cuts)[cut_indices]
                left, right = (
                    np.min(sites_to_collapse),
                    np.max(sites_to_collapse),
                )
                for site in range(left, right + 1):
                    updated_character_array[
                        site
                    ] = self.heritable_missing_data_state
            else:
                cuts_remaining.append(np.array(cuts)[cut_indices[0]])

        return updated_character_array, cuts_remaining

    def introduce_states(
        self, character_array: List[int], cuts: List[int]
    ) -> List[int]:
        """Adds states to character array.

        New states are added to the character array at the predefined cut
        locations.

        Args:
            character_array: Character array
            cuts: Loci being cut

        Returns:
            An updated character array.
        """

        updated_character_array = character_array.copy()

        states = list(self.mutation_priors.keys())
        probabilities = list(self.mutation_priors.values())

        for i in cuts:
            state = np.random.choice(states, 1, p=probabilities)[0]
            updated_character_array[i] = state

        return updated_character_array

    def silence_cassettes(
        self,
        character_array: List[int],
        silencing_rate: float,
        missing_state: int = -1,
        heritable_missing_data_state: int = -1,
    ) -> List[int]:
        """Silences cassettes.

        Using the specified silencing rate, this function will randomly select
        cassettes to silence.

        Args:
            character_array: Character array
            silencing_rate: Silencing rate.
            missing_state: State to use for encoding missing data.
            heritable_missing_data_state: Heritable states, which should not be
                overwritten.

        Returns:
            An updated character array.

        """

        updated_character_array = character_array.copy()

        cassettes = self.get_cassettes()
        cut_site_by_cassette = np.digitize(
            range(len(character_array)), cassettes
        )

        for cassette in range(1, self.number_of_cassettes + 1):
            if np.random.uniform() < silencing_rate:
                indices = np.where(cut_site_by_cassette == cassette)
                left, right = np.min(indices), np.max(indices)
                for site in range(left, right + 1):
                    if updated_character_array[site] != heritable_missing_data_state:
                        updated_character_array[site] = missing_state

        return updated_character_array

    def get_cassettes(self) -> List[int]:
        """Obtain indices of individual cassettes.

        A helper function that returns the indices that correpspond to the
        independent cassettes in the experiment.

        Returns:
            An array of indices corresponding to the start positions of the
                cassettes.
        """

        cassettes = [
            (self.size_of_cassette * j)
            for j in range(0, self.number_of_cassettes)
        ]
        return cassettes
