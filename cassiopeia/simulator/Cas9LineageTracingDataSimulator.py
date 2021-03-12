"""
A Cas9-based lineage tracing data simulator. This is a sublcass of the 
LineageTracingDataSimulator that simulates the data produced from Cas9-based
technologies (e.g, as described in Chan et al, Nature 2019 or McKenna et al,
Science 2016). This simulator implements the method `overlay_data` which takes
in a CassiopeiaTree with edge lengths and overlays states onto cut-sites.
"""
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import LineageTracingDataSimulator


class Cas9LineageTracingDataSimulator(LineageTracingDataSimulator):
    """Simulates Cas9-based lineage tracing data.

    This subclass of `LineageTracingDataSimulator` extends the `overlay_data`
    function to simulate Cas9-based mutations onto a defined "cassette". These
    cassettes emulate the TargetSites described in Chan et al, Nature 2019 and
    the GESTALT arrays described in McKenna et al, Science 2020 in which several
    Cas9 cut-sites are arrayed together. In the Chan et al technology, these
    cassettes are of length 3; in McKenna et al, these cassettes are of length
    10.

    The class accepts several parameters that govern the Cas9-based tracer.
    First and foremost is the cutting rate, describing how fast Cas9 is able to
    cut. We model Cas9 cutting as an exponential process, parameterized by the
    specified mutation rate - specifically, for a lifetime `t` and parameter
    `lambda`, the expected probability of Cas9 mutation, per site, is
    `exp(-lambda * t)`.
    
    Second, the class accepts architecture of the recorder - described by the
    size of the cassette (by default 3) and the number of cassettes. The
    resulting lineage will have |cassette|*(# of cassettes) characters.

    Third, the class accepts a state distribution describing the relative
    likelihoods of various indels. This is very useful, as it is typical that a
    handful of mutations are far likelier than the bulk of the possible
    mutations.

    Finally, the class accepts a silencing rate. This is normally a rare event
    in which an entire cassette is transcriptionally silenced and therefore
    not observed.

    The function `overlay_data` will operate on the tree in place and will
    specifically modify the data stored in the character attributes.

    Args:
        number_of_cassettes: Number of cassettes (i.e., arrays of target sites)
        size_of_cassette: Number of editable target sites per cassette
        mutation_rate: Exponential parameter for the Cas9 cutting rate.
        state_distribution: Distribution from which to simulate state
            likelihoods
        number_of_states: Number of states to simulate
        silencing_rate: Silencing rate for the cassettes, per node
    """

    def __init__(
        self,
        number_of_cassettes: int = 10,
        size_of_cassette: int = 3,
        mutation_rate: float = 0.01,
        state_distribution: Callable[None, float] = lambda: np.random.exponential(1e-5),
        number_of_states: int = 100,
        silencing_rate: float = 1e4,
    ):

        self.number_of_cut_sites = number_of_cut_sites
        self.number_of_cassettes = number_of_cassettes
        self.mutation_rate = mutation_rate

        self.mutation_priors = {}
        probabilites = [state_distribution() for _ in range(number_of_states)]
        Z = sum(probabilites)
        for i in range(number_of_states):
            self.mutation_priors[i + 1] = probabilites[i] / Z

        self.silencing_rate = silencing_rate
    
    def overlay_data(self, tree: CassiopeiaTree):
        """Overlays Cas9-based lineage tracing data onto the CassiopeiaTree.

        Args:
            tree: Input CassiopeiaTree
        """
        
        number_of_characters = self.number_of_cassettes * self.number_of_cut_sites
        for node in tree.depth_first_traverse_nodes(tree.root, postorder=False):

            if tree.is_root(node):
                tree.set_character_states = [0]*number_of_characters
                continue

            parent = tree.get_parent(node) 
            t = tree.get_time(node) - tree.get_time(parent)

            p = 1 - (np.exp(-t * self.mutation_rate))
    
            character_array = tree.get_character_states(parent)
            open_sites = [c for c in range(len(character_array)) if character_array[c] == 0]

            number_of_mutations = np.random.binomial(len(open_sites), p)
            new_cuts = np.random.choice(open_sites, number_of_mutations)
            
            # collapse cuts that are on the same cassette
            character_array, cuts_remaining = self.collapse_sites(character_array, new_cuts)
            
            # introduce new states at cut sites
            character_array = self.introduce_states(character_array, cuts_remaining)
            
            # silence cassettes
            character_array = self.silence_cassettes(character_array)

            tree.set_character_states(node, character_array)


    def collapse_sites(self, character_array: List[int], cuts: List[int]) -> Tuple[List[int], List[int]]:
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

        cassettes = self.get_cassettes()
        cut_to_cassette = np.digitize(cuts, cassettes)
        cuts_remaining =[]
        for cassette in np.unique(cut_to_cassette):
            cut_indices = np.where(cut_to_cassette == cassette)[0]
            if len(cut_indices) > 1:
                cuts = new_cuts[cut_indices]
                left, right = np.min(cuts), np.max(cuts)
                for site in range(left, right+1):
                    character_array[site] = -1
            else:
                cuts_remaining.append(cut_indices[0])
        
        return character_array, cuts_remaining

    def introduce_states(self, character_array: List[int], cuts: List[int]) -> List[int]:
        """Adds states to character array.

        New states are added to the character array at the predefined cut
        locations.

        Args:
            character_array: Character array
            cuts: Loci being cut

        Returns:
            An updated character array.
        """

        for i in cuts:
            state = self.random.choice(list(self.mutation_priors.keys()), 1, p=list(self.mutation_priors.values()))
            character_array[i] = state

        return character_array

    def silence_cassettes(self, character_array: List[int]):
        """Silences cassettes.

        Using the predefined silencing rate of this simulator, this function
        simulates the rare event of transcriptional silencing.

        Args:
            character_array: Character array

        Returns:
            An updated character array.

        """
        cassettes = self.get_cassettes()
        cut_site_by_cassette = np.digitize(range(len(character_array)), cassettes)
        for cassette in cassettes:
            if np.random.uniform() < self.silencing_rate:
                indices = np.where(cut_site_by_cassette)
                left, right = np.min(indices), np.max(indices)
                for site in range(left, right+1):
                    character_array[site] = -1

    def get_cassettes(self) -> List[int]:
        """Obtain indices of individual cassettes.

        A helper function that returns the indices that correpspond to the 
        independent cassettes in the experiment.

        Returns:
            An array of indices corresponding to the start positions of the
                cassettes.
        """

        cassettes = [(self.number_of_cut_sites * j) for j in range(1, self.number_of_cassettes)]
        return cassettes
