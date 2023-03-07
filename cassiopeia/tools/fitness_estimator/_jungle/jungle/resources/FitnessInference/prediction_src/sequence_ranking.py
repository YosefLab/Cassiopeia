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
# class sequence_ranking
# a subclass of node_ranking that operates on alignments rather than trees.
# given an alignment, it will generate a tree using the tree_building function
# in tree_utils.
#
# class alignment
# a simple class holding an aligment, calculates allele frequencies, etc and
# provides methods to measure the average distance to a particular sequence
# or the distance between two populations.
#
#########################################################################################
#


import time

import numpy as np
from Bio import Seq, SeqRecord
from Bio.Align import AlignInfo, MultipleSeqAlignment
from Bio.Alphabet import generic_dna, generic_protein

from . import tree_utils
from .node_ranking import node_ranking

verbose = True

aa_codes = {
    "A": ("Ala", "Alanine"),
    "R": ("Arg", "Arginine"),
    "N": ("Asn", "Asparagine"),
    "D": ("Asp", "Aspartic acid"),
    "C": ("Cys", "Cysteine"),
    "Q": ("Gln", "Glutamine"),
    "E": ("Glu", "Glutamic acid"),
    "G": ("Gly", "Glycine"),
    "H": ("His", "Histidine"),
    "I": ("Ile", "Isoleucine"),
    "L": ("Leu", "Leucine"),
    "K": ("Lys", "Lysine"),
    "M": ("Met", "Methionine"),
    "F": ("Phe", "Phenylalanine"),
    "P": ("Pro", "Proline"),
    "S": ("Ser", "Serine"),
    "T": ("Thr", "Threonine"),
    "W": ("Trp", "Tryptophan"),
    "Y": ("Tyr", "Tyrosine"),
    "V": ("Val", "Valine"),
    "B": ("Asx", "Aspartic acid or Asparagine"),
    "Z": ("Glx", "Glutamine or Glutamic acid"),
}


##############################################################################
## class holding alignment + outgroup + allele frequencies etx
##############################################################################


class alignment(object):
    """
    class holding an aligment, an outgroup, allele frequencies and utility functions
    """

    def __init__(
        self, aln, outgroup, cds=None, collapse=False, build_tree=True
    ):
        """
        parameters:
        aln         --  biopython alignment
        outgroup    --  outgroup sequence
        annotation  --  dictionary or panda DataFrame that holds spatio/temporal info
        cds         --  coding region
        """
        self.aln = aln
        self.outgroup = outgroup
        self.collapse = collapse
        self.alphabet = "ACTG-"
        self.aa_alphabet = "".join(list(aa_codes.keys()))
        self.make_tree = build_tree
        if cds is not None:
            self.cds = cds
            self.protein = True
        else:
            # specifies the beginning and end of the coding region, allows to add
            # a padding of XX to the beginning of the seq, in case not the complete
            # cds is present (i.e., to preserve aa numbering in a protein)
            self.cds = {"begin": 0, "end": 0, "pad": 0}
            self.protein = False
        self.process_alignment()

    def process_alignment(self):
        """
        calculate different properties of the alignment that are needed for later
        distance calculations
        """
        if verbose:
            t1 = time.time()
            print("processing alignment of", len(self.aln), "sequences")
        self.summary_info = AlignInfo.SummaryInfo(self.aln)
        self.consensus = self.summary_info.dumb_consensus()
        if verbose:
            print("calculating allele frequencies...", end=" ")
        self.calculate_allele_frequencies()
        if self.protein:
            self.translate_alignment()
        if verbose:
            print("done in ", np.round(time.time() - t1, 2), "seconds")
            t1 = time.time()
            print("calculating tree...")
        if self.make_tree:
            self.build_tree(collapse_nodes=self.collapse)
            if verbose:
                print("done in ", np.round(time.time() - t1, 2), "seconds")

    def calculate_allele_frequencies(self):
        """
        calculates the allele frequencies of the stored alignment
        """
        # allocate an array for the allele frequencies and cast the alignment to an array
        self.allele_frequencies = np.zeros(
            (len(self.alphabet), self.aln.get_alignment_length())
        )
        self.aln_array = np.array(self.aln)
        # loop over all nucleotides, calculate the frequency in each column
        for ni, nuc in enumerate(self.alphabet):
            self.allele_frequencies[ni, :] = np.mean(
                self.aln_array == nuc, axis=0
            )

    def calculate_aa_allele_frequencies(self):
        """
        calculates the allele frequencies of the stored amino acid alignment
        """
        if self.protein:
            self.aa_allele_frequencies = np.zeros(
                (len(self.aa_alphabet), self.aa_aln.get_alignment_length())
            )
            tmp_aln = np.array(self.aa_aln)
            # loop over the entire amino acid alphabet, calculate the frequency in each column
            for ai, aa in enumerate(self.aa_alphabet):
                self.aa_allele_frequencies[ai, :] = np.mean(
                    tmp_aln == aa, axis=0
                )
        else:
            print("Not a protein sequence")

    def translate_alignment(self):
        """
        translate the alignment and calculate amino acid consensus
        """
        self.aa_aln = MultipleSeqAlignment([])
        if self.cds["end"] >= 0:
            last_base = self.cds["end"]
        else:
            last_base = self.aln.get_alignment_length() + self.cds["end"] + 1

        # translate, add cds['pad'] Xs at the beginning
        # TODO: make translation gap-tolerant
        for seq in self.aln:
            try:
                tmp_seq = (
                    "X" * self.cds["pad"]
                    + seq.seq[self.cds["begin"] : last_base].translate()
                )
            except:
                tmp_seq = Seq.Seq(
                    "X"
                    * (
                        self.cds["pad"]
                        + (self.cds["end"] - self.cds["begin"]) / 3
                    ),
                    generic_protein,
                )
                print(self.cds)
            if self.cds["end"] - self.cds["begin"] == 0:
                tmp_seq = Seq.Seq("X", generic_protein)

            self.aa_aln.append(
                SeqRecord.SeqRecord(seq=tmp_seq, name=seq.name, id=seq.id)
            )
        # process amino acid alignment
        self.aa_summary_info = AlignInfo.SummaryInfo(self.aa_aln)
        self.aa_consensus = self.aa_summary_info.dumb_consensus()
        self.calculate_aa_allele_frequencies()

    def mean_distance_to_sequence(self, query):
        """
        calculate the average hamming distance between the query sequence and
        the stored alignment based on the allele frequencies
        """
        distance = 0
        # calculate the average distance at each alignment column via the allele frequencies
        # for each nucleotide state. average over columns, sum over nucleotides
        for ni, nuc in enumerate(self.alphabet):
            distance += np.mean(
                (np.array(query) == nuc) * (1 - self.allele_frequencies[ni, :])
            )
        return distance

    def mean_distance_to_set(self, other_af):
        """
        calculate the average hamming distance between the another alignment
        and the stored alignment based on the allele frequencies
        """
        # calculate the average distance at each alignment column via the allele frequencies
        # for each nucleotide state. average over columns, sum over nucleotides
        return np.mean(
            np.sum(self.allele_frequencies * (1.0 - other_af), axis=0)
        )

    def aa_distance_to_sequence(self, query, positions=None):
        """
        calculate the average hamming distance between the query sequence and
        the stored alignment based on the allele frequencies
        """
        if self.protein:
            distance = 0
            if positions is None:
                positions = np.arange(len(query))
            relevant_positions = np.zeros(len(query))
            relevant_positions[positions] = 1
            # calculate the average distance at each alignment column via the allele frequencies
            # for each amino acid. average over columns, sum over amino acids
            for ai, aa in enumerate(self.aa_alphabet):
                distance += np.mean(
                    (np.array(query) == aa)
                    * (1 - self.aa_allele_frequencies[ai, :])
                    * relevant_positions
                )
            return distance
        else:
            print("Not a protein sequence")
            return np.nan

    def aa_distance_to_set(self, other, positions=None):
        """
        calculate the average hamming distance between the another alignment
        and the stored alignment based on the allele frequencies
        """
        if self.protein:
            if positions is None:
                positions = np.arange(self.aa_allele_frequencies.shape[1])
            relevant_positions = np.zeros(self.aa_allele_frequencies.shape[1])
            relevant_positions[positions] = 1
            # calculate the average distance at each alignment column via the allele frequencies
            # for each amino acid. average over columns, sum over amino acids
            return np.mean(
                np.sum(
                    self.aa_allele_frequencies
                    * (1.0 - other.aa_allele_frequencies),
                    axis=0,
                )
                * relevant_positions
            )
        else:
            print("Not a protein sequence")
            return np.nan

    def build_tree(self, collapse_nodes=False):
        """
        given the alignment and the outgroup, infer a phylogenetic tree, root with outgroup
        and infer ancestral states for each internal node
        """
        if verbose:
            print("calculating tree and infering ancestral sequences")
            tmp_t = time.time()
        self.T = tree_utils.calculate_tree(
            self.aln, self.outgroup, ancestral=True
        )

        # put mutations on branches
        if self.protein:
            tree_utils.translate_sequences_on_tree(self.T, self.cds)
        tree_utils.mutations_on_branches(self.T, aa=self.protein)
        if verbose:
            print("done in ", np.round(time.time() - tmp_t, 2), "s")

        # collapse branches with 0 length (that is identical sequences on both ends of the branch) if requested
        if collapse_nodes:
            tree_utils.collapse_zero_branches(self.T.root)


##############################################################################
## sub class of node_ranking that operates on an alignment rather than a tree
##############################################################################


class sequence_ranking(node_ranking):
    """
    subclass of node_ranking that handles sequence data, builds a tree,
    and uses the prediction and ranking function of the base
    """

    def __init__(self, sequence_data, distance_scale=1.0, *args, **kwargs):
        node_ranking.__init__(self, *args, **kwargs)
        self.distance_scale = distance_scale
        self.data = sequence_data

        # calcute the coalescence time scale associated with tree
        pairwise_distance = np.sum(
            (self.data.allele_frequencies * (1 - self.data.allele_frequencies)),
            axis=0,
        ).mean()
        # distance_scale * D * T_2 = 1, T2 = pi/2
        self.time_scale = distance_scale * self.D * pairwise_distance * 0.5
        self.set_tree(tree=self.data.T, time_scale=self.time_scale)

    def predict(self):
        """
        initialize the tree of the underlying fitness inference.
        returns:
            the highest ranked external node
        """
        self.compute_rankings()
        return self.best_node(self.methods[0])
