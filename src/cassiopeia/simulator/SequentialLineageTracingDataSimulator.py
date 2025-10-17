"""Sequential lineage tracing data simulator built on LineageTracingDataSimulator."""

import math

import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator.LineageTracingDataSimulator import (
    LineageTracingDataSimulator,
)


class SequentialLineageTracingDataSimulator(LineageTracingDataSimulator):
    """Simulates Cas9-based lineage tracing data.

    This subclass of `LineageTracingDataSimulator` implements the `overlay_data`
    function to simulate Cas9-based edits onto a defined sequential recording
    site, aka "DNA tape" or "cassette". These cassettes are designed such that
    only one site can be edited at a time and the position of the active site
    shifts as edits are made. One example of this type of lineage recorder is
    the DNA Typewriter system described by Choi et al, Nature 2022.

    First, the class accepts several parameters that govern the Cas9-based
    tracer. These are 1. the initiation rate which describes the rate at which
    particular cassette starts recording and 2. the continuation rate which
    describes the rate at which sequential events occur once recording has
    started. We model Cas9 recording as an exponential process,
    parameterized by the specified initiation and continuation rates.

    Second, the class accepts the architecture of the recorder - described by
    the size of the cassette (by default 3) and the number of cassettes. The
    resulting lineage will have (size_of_cassette * number_of_cassettes)
    characters.

    Third, the class accepts a state distribution describing the relative
    likelihoods of various states. This is very useful, as the probability of
    different editing outcomes can vary greatly.

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
        initiation_rate: Exponential parameter for the Cas9 initiation rate
        continuation_rate: Exponential parameter for the Cas9 continuation rate
        state_priors: An optional dictionary mapping states to their prior
            probabilities.
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

    Raises:
            DataSimulatorError if assumptions about the system are broken.
    """

    def __init__(
        self,
        number_of_cassettes: int = 10,
        size_of_cassette: int = 3,
        initiation_rate: float = 0.1,
        continuation_rate: float = 0.1,
        state_priors: dict[int, float] = None,
        heritable_silencing_rate: float = 0,
        stochastic_silencing_rate: float = 0,
        heritable_missing_data_state: int = -1,
        stochastic_missing_data_state: int = -1,
        random_seed: int | None = None,
    ):
        if number_of_cassettes <= 0 or not isinstance(number_of_cassettes, int):
            raise DataSimulatorError("Specify a positive number of cassettes.")
        if size_of_cassette <= 0 or not isinstance(size_of_cassette, int):
            raise DataSimulatorError("Specify a positive number of cut-sites per cassette.")
        if not isinstance(initiation_rate, float):
            raise DataSimulatorError("Initiation rate must be a float.")
        if initiation_rate <= 0:
            raise DataSimulatorError("Initiation rate must be positive.")
        if not isinstance(continuation_rate, float):
            raise DataSimulatorError("Continuation rate must be a float.")
        if continuation_rate <= 0:
            raise DataSimulatorError("Continuation rate must be positive.")
        if not isinstance(state_priors, dict):
            raise DataSimulatorError("State priors dictionary is required")
        if np.any(np.array(list(state_priors.values())) < 0):
            raise DataSimulatorError("State prior to be non-negative.")
        Z = np.sum(list(state_priors.values()))
        if not math.isclose(Z, 1.0):
            raise DataSimulatorError("State priors do not sum to 1.")

        self.size_of_cassette = size_of_cassette
        self.number_of_cassettes = number_of_cassettes

        self.initiation_rate = initiation_rate
        self.continuation_rate = continuation_rate
        self.state_priors = state_priors

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

        # initialize character matrix
        number_of_characters = self.number_of_cassettes * self.size_of_cassette
        initiation_sites = range(0, number_of_characters, self.size_of_cassette)
        character_matrix = {}
        for node in tree.nodes:
            character_matrix[node] = [-1] * number_of_characters

        for node in tree.depth_first_traverse_nodes(tree.root, postorder=False):
            if tree.is_root(node):
                character_matrix[node] = [0] * number_of_characters
                continue

            parent = tree.parent(node)
            life_time = tree.get_time(node) - tree.get_time(parent)

            character_array = character_matrix[parent].copy()

            # sequentially edit each cassette
            for c in initiation_sites:
                # missing
                if character_array[c] == -1:
                    continue
                # initiated
                elif character_array[c] != 0:
                    time = 0
                    i = c + 1
                    while time < life_time and i < c + self.size_of_cassette:
                        if character_array[i] == 0:
                            time = time + np.random.exponential(1 / self.continuation_rate)
                            if time < life_time:
                                self.edit_site(character_array, i, self.state_priors)
                        i += 1
                # uninitiated
                elif character_array[c] == 0:
                    i = c
                    time = np.random.exponential(1 / self.initiation_rate)
                    if time < life_time:
                        self.edit_site(character_array, i, self.state_priors)
                    i += 1
                    while time < life_time and i < c + self.size_of_cassette:
                        if character_array[i] == 0:
                            time = time + np.random.exponential(1 / self.continuation_rate)
                            if time < life_time:
                                self.edit_site(character_array, i, self.state_priors)
                        i += 1

            # silence cassettes
            silencing_probability = 1 - (np.exp(-life_time * self.heritable_silencing_rate))
            character_array = self.silence_cassettes(
                character_array,
                silencing_probability,
                self.heritable_missing_data_state,
            )

            character_matrix[node] = character_array

        # apply stochastic silencing
        for leaf in tree.leaves:
            character_matrix[leaf] = self.silence_cassettes(
                character_matrix[leaf],
                self.stochastic_silencing_rate,
                self.stochastic_missing_data_state,
            )

        tree.set_all_character_states(character_matrix)

    def edit_site(
        self, character_array: list[int], site: int, state_priors: dict[int, float]
    ) -> list[int]:
        """Edits a site.

        Args:
            character_array: Character array
            site: Site to edit
            state_priors: Dictionary mapping states to their prior
                probabilities.
        """
        states = list(state_priors.keys())
        probabilities = list(state_priors.values())
        state = np.random.choice(states, 1, p=probabilities)[0]
        character_array[site] = state

    def silence_cassettes(
        self,
        character_array: list[int],
        silencing_rate: float,
        missing_state: int = -1,
    ) -> list[int]:
        """Silences cassettes.

        Using the specified silencing rate, this function will randomly select
        cassettes to silence.

        Args:
            character_array: Character array
            silencing_rate: Silencing rate.
            missing_state: State to use for encoding missing data.

        Returns:
                    An updated character array.

        """
        updated_character_array = character_array.copy()

        number_of_characters = self.number_of_cassettes * self.size_of_cassette
        cassettes = range(0, number_of_characters, self.size_of_cassette)
        cut_site_by_cassette = np.digitize(range(len(character_array)), cassettes)

        for cassette in range(1, self.number_of_cassettes + 1):
            if np.random.uniform() < silencing_rate:
                indices = np.where(cut_site_by_cassette == cassette)
                left, right = np.min(indices), np.max(indices)
                for site in range(left, right + 1):
                    updated_character_array[site] = missing_state

        return updated_character_array
