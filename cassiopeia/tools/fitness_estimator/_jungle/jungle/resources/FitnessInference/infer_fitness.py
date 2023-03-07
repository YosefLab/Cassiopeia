#!/usr/bin/env python
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
# rank_sequences.py
# Run the fitness inference on sequences in an alignment. Output is going to be written
# to a folder named by current date and time.
# INPUT:
# --aln             name of the alingment file, fasta format. can be gzipped
# --outgroup        name of the outgroup sequence. has to in the alignment file
# other parameters specify parameters of the inference algorithm and are optional
#
#########################################################################################


import argparse

#########################################################################################
###parse the command line arguments
#########################################################################################
parser = argparse.ArgumentParser(
    description="rank sequences in a multiple sequence aligment"
)
parser.add_argument(
    "--aln", type=str, required=True, help="alignment of sequences to by ranked"
)
parser.add_argument(
    "--outgroup", type=str, required=True, help="name of outgroup sequence"
)
parser.add_argument(
    "--eps_branch",
    default=1e-5,
    type=float,
    help="minimal branch length for inference",
)
parser.add_argument(
    "--diffusion", default=0.5, type=float, help="fitness diffusion coefficient"
)
parser.add_argument(
    "--gamma",
    default=1.0,
    type=float,
    help="scale factor for time scale, choose high (>2) for prediction, 1 for fitness inference",
)
parser.add_argument(
    "--omega",
    default=0.1,
    type=float,
    help="approximate sampling fraction diveded by the fitness standard deviation",
)
parser.add_argument(
    "--collapse",
    const=True,
    default=False,
    nargs="?",
    help="collapse internal branches with identical sequences",
)
parser.add_argument(
    "--plot", const=True, default=False, nargs="?", help="plot trees"
)
params = parser.parse_args()
#########################################################################################

import matplotlib

matplotlib.use("pdf")
import argparse
import os
import sys
import time

sys.path.append("./prediction_src")
import numpy as np
import tree_utils
from Bio import Align, AlignIO, Phylo, SeqIO
from matplotlib import pyplot as plt
from sequence_ranking import *

## matplotlib set up
mpl_params = {
    "backend": "pdf",
    "axes.labelsize": 20,
    "text.fontsize": 20,
    "font.sans-serif": "Helvetica",
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": False,
}
plt.rcParams.update(mpl_params)


##########################################################################################
def ofunc(fname, mode):
    """
    custom file open that chooses between gzip and regular open
    """
    if fname[-3:] == ".gz":
        import gzip

        return gzip.open(fname, mode)
    else:
        return open(fname, mode)


##########################################################################################


##########################################################################################
## read the alignment, identify the outgroup
##########################################################################################
aln = Align.MultipleSeqAlignment([])
outgroup = None
with ofunc(params.aln, "r") as alnfile:
    for sec_rec in SeqIO.parse(alnfile, "fasta"):
        if sec_rec.name != params.outgroup:
            aln.append(sec_rec)
        else:
            outgroup = sec_rec

if outgroup is None:
    print("outgroup not in alignment -- FATAL")
    exit()

#######################################################################################
## set up the sequence data set and run the prediction algorithm
#######################################################################################
seq_data = alignment(aln, outgroup)

prediction = sequence_ranking(
    seq_data,
    eps_branch_length=params.eps_branch,
    pseudo_count=5,
    methods=["mean_fitness"],
    D=params.diffusion,
    distance_scale=params.gamma,
    samp_frac=params.omega,
)

best_node = prediction.predict()

#######################################################################################
## output
#######################################################################################

# make directory to write files to
dirname = "./" + time.strftime("%Y%m%d_%H-%M-%S")
if not os.path.isdir(dirname):
    os.mkdir(dirname)

# name internal nodes
for ni, node in enumerate(prediction.non_terminals):
    node.name = str(ni + 1)

# write tree to file
Phylo.write(prediction.T, dirname + "/reconstructed_tree.nwk", "newick")

# write inferred ancestral sequences to file
with open(dirname + "/ancestral_sequences.fasta", "w") as outfile:
    for node in prediction.non_terminals:
        outfile.write(">" + node.name + "\n" + str(node.seq) + "\n")

## write sequence ranking to file
# terminal nodes
prediction.rank_by_method(nodes=prediction.terminals, method="mean_fitness")
with open(dirname + "/sequence_ranking_terminals.txt", "w") as outfile:
    outfile.write(
        "#" + "\t".join(["name", "rank", "mean", "standard dev"]) + "\n"
    )
    for node in prediction.terminals:
        outfile.write(
            "\t".join(
                map(
                    str,
                    [
                        node.name,
                        node.rank,
                        node.mean_fitness,
                        np.sqrt(node.var_fitness),
                    ],
                )
            )
            + "\n"
        )

# terminal nodes
prediction.rank_by_method(nodes=prediction.non_terminals, method="mean_fitness")
with open(dirname + "/sequence_ranking_nonterminals.txt", "w") as outfile:
    outfile.write("#" + "\t".join(["name", "rank", "mean", "variance"]) + "\n")
    for node in prediction.non_terminals:
        outfile.write(
            "\t".join(
                map(
                    str,
                    [
                        node.name,
                        node.rank,
                        node.mean_fitness,
                        np.sqrt(node.var_fitness),
                    ],
                )
            )
            + "\n"
        )


if params.plot:
    tree_utils.plot_prediction_tree(prediction)
    plt.savefig(dirname + "/marked_up_tree.pdf")
