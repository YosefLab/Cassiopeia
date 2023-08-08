"""
A prime editing-based lineage tracing data simulator. This simulator implements 
the method `overlay_data` which takes in a CassiopeiaTree with edge lengths and 
overlays states onto target-sites.
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


class PELineageTracingDataSimulator(LineageTracingDataSimulator):
    """Simulates prime-editing based lineage tracing data.

    This subclass of `LineageTracingDataSimulator` implements the `overlay_data`
    function to simulate prime edits to a defined number of cassettes.

    Args:
        number_of_cassettes: Number of cassettes (i.e., arrays of target sites)
        size_of_cassette: Number of editable target sites per cassette
        mutation_rate: Per site editing rate. Can be a float or a list of floats 
            of length `size_of_cassette` or `number_of_cassettes * size_of_cassette`.
        mutation_priors: Dictionary mapping states to their prior probabilities.
            Can also be a list of dictionaries of length `size_of_cassette` or
            `number_of_cassettes * size_of_cassette`.
        heritable_missing_rate: Missing rate for the cassettes, per node,
            simulating heritable missing data events.
        stochastic_missing_rate: Rate at which to randomly drop out cassettes,
            to simulate dropout due to low sensitivity of assays.
        heritable_missing_data_state: Integer representing data that has gone
            missing due to a heritable event.
        stochastic_missing_data_state: Integer representing data that has
            gone missing due to the stochastic dropout from assay
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
        mutation_rate: Union[float, List[float]] = 0.01,
        state_priors: Dict[int, float] = [{i:.125 for i in range(0,8)}] * 3,
        heritable_missing_rate: float = 0,
        stochastic_missing_rate: float = 0,
        heritable_missing_data_state: int = -1,
        stochastic_missing_data_state: int = -1,
        random_seed: Optional[int] = None,
        collapse_sites_on_cassette: bool = True,
    ):

        if number_of_cassettes <= 0 or not isinstance(number_of_cassettes, int):
            raise DataSimulatorError("Specify a positive number of cassettes.")
        if size_of_cassette <= 0 or not isinstance(size_of_cassette, int):
            raise DataSimulatorError(
                "Specify a positive number of cut-sites per cassette."
            )

        self.size_of_cassette = size_of_cassette
        self.number_of_cassettes = number_of_cassettes
        self.number_of_characters = size_of_cassette * number_of_cassettes

        if isinstance(mutation_rate, float):
            if mutation_rate < 0:
                raise DataSimulatorError(
                    "Mutation rate needs to be non-negative."
                )
            self.mutation_rate_per_character = [
                mutation_rate
            ] * self.number_of_characters
        else:
            if len(mutation_rate) == self.number_of_characters:
                self.mutation_rate_per_character = mutation_rate
            elif len(mutation_rate) == self.size_of_cassette:
                self.mutation_rate_per_character = (
                    mutation_rate * self.number_of_cassettes
                )
            else:
                raise DataSimulatorError(
                    "Length of mutation rate array is not"
                    " the same as the number of characters"
                    " or the number of cassettes."
                )
            if np.any(np.array(mutation_rate) < 0):
                raise DataSimulatorError(
                    "Mutation rate needs to be non-negative."
                )
            
        if isinstance(state_priors, dict):
            self.mutation_priors = [
                state_priors
            ] * self.number_of_characters
        else:
            if len(state_priors) == self.number_of_characters:
                self.mutation_priors = state_priors
            elif len(state_priors) == self.size_of_cassette:
                self.mutation_priors = (
                    state_priors * self.number_of_cassettes
                )
            else:
                raise DataSimulatorError(
                    "Length of mutation prior array is not"
                    " the same as the number of characters"
                    " or the number of cassettes."
                )

        self.heritable_missing_rate = heritable_missing_rate
        self.stochastic_missing_rate = stochastic_missing_rate

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


        # initialize character states
        character_matrix = {}
        for node in tree.nodes:
            character_matrix[node] = [-1] * self.number_of_characters

        for node in tree.depth_first_traverse_nodes(tree.root, postorder=False):

            if tree.is_root(node):
                character_matrix[node] = [0] * self.number_of_characters
                continue

            parent = tree.parent(node)
            life_time = tree.get_time(node) - tree.get_time(parent)

            character_array = character_matrix[parent]
            open_sites = [
                c
                for c in range(len(character_array))
                if character_array[c] == 0
            ]

            new_edits = []
            for site in open_sites:
                mutation_rate = self.mutation_rate_per_character[site]
                mutation_probability = 1 - (np.exp(-life_time * mutation_rate))

                if np.random.uniform() < mutation_probability:
                    new_edits.append(site)

            # introduce new states at cut sites
            character_array = self.introduce_states(
                character_array, new_edits
            )

            # silence cassettes
            silencing_probability = 1 - (
                np.exp(-life_time * self.heritable_missing_rate)
            )
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
                self.stochastic_missing_rate,
                self.stochastic_missing_data_state,
            )

        tree.set_all_character_states(character_matrix)

    def introduce_states(
        self, character_array: List[int], edits: List[int]
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

        for i in edits:
            state = np.random.choice(list(self.mutation_priors[i].keys()), 
                                    1, p=list(self.mutation_priors[i].values()))[0]
            updated_character_array[i] = state

        return updated_character_array

    def silence_cassettes(
        self,
        character_array: List[int],
        silencing_rate: float,
        missing_state: int = -1,
    ) -> List[int]:
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

        cassettes = self.get_cassettes()
        cut_site_by_cassette = np.digitize(
            range(len(character_array)), cassettes
        )

        for cassette in range(1, self.number_of_cassettes + 1):
            if np.random.uniform() < silencing_rate:
                indices = np.where(cut_site_by_cassette == cassette)
                left, right = np.min(indices), np.max(indices)
                for site in range(left, right + 1):
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