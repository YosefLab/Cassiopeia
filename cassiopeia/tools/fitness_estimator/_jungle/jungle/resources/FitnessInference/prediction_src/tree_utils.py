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
# tree_utils.py
# a collection of functions to build and manipulate trees.
#
#########################################################################################


import os
import time

import numpy as np
from matplotlib import cm

# define fasttree binary used for tree building,
# if no fasttree binary is found, no trees can be built

fasttree_bin = "/local10G/rfhorns/resources/FastTree/fasttree"

# if not os.path.exists(fasttree_bin):
#    if os.system('fasttree')==0:
#        fasttree_bin = 'fasttree'
#        # print 'using default fasttree binary'
#    else:
#        fasttree_bin = None
#        # print 'no fasttree binary found'
# else:
#    # print 'using fasttree binary:', fasttree_bin
#    pass

verbose = 1
my_favorite_cmap = cm.jet

##################################################
# UTILITY FUNCTION FOR TREE BUILDING, TREE LABELING AND TREE MANIPULATION
##################################################


def calculate_tree(aln, outgroup, ancestral=True):
    """
    takes a biopython alignment and an outgroup sequences, writes a temporary alignment
    and builds a tree using fasttree
    aln         --biopython alignment
    outgroup    -- outgroup sequence (biopython)
    """
    import gzip
    import subprocess
    from io import StringIO

    from Bio import AlignIO, Phylo, SeqIO

    from . import ancestral

    # write sequences into a temporary alignment file
    tmp_file_name = "tmp_" + str(np.random.randint(1000000000)) + ".fasta.gz"
    with gzip.open(tmp_file_name, "w") as outfile:
        SeqIO.write(outgroup, outfile, "fasta")
        for seq in aln:
            SeqIO.write(seq, outfile, "fasta")

    print("building tree of file", tmp_file_name)
    fasttree_cmd = [fasttree_bin, "-nt"]
    ps = subprocess.Popen(
        ("gunzip", "-c", tmp_file_name), stdout=subprocess.PIPE
    )
    fasttree_output = subprocess.check_output(fasttree_cmd, stdin=ps.stdout)
    ps.wait()

    # all fasttree output is returned as byte string, parse and write tree to file
    biopython_tree = Phylo.read(StringIO(fasttree_output), "newick")
    biopython_tree.root.branch_length = 0.000001
    with gzip.open(tmp_file_name, "r") as infile:
        tmp_aln = AlignIO.read(infile, "fasta")
    # determine the root clade, root, and ladderize
    outgroup_clade = [
        z
        for z in biopython_tree.get_terminals()
        if z.name.startswith(outgroup.name)
    ][0]
    biopython_tree.root_with_outgroup(outgroup_clade)
    biopython_tree.ladderize()

    # infer ancestral sequences in place
    print("set_alignment: inferring ancestral sequences...")
    anc_rec = ancestral.ancestral_sequences(
        biopython_tree, tmp_aln, copy_tree=False
    )
    anc_rec.calc_ancestral_sequences()

    biopython_tree.prune(outgroup_clade)
    biopython_tree.root.branch_length = 0.001
    os.remove(tmp_file_name)
    return biopython_tree


def branch_label(node, aa=True, display_positions=None):
    """
    add mutations as labels to branches
    arguments:
    aa                  -- if true, use amino acid sequences instead of nucleotide sequences
    display_positions   -- specify position at whcih mutations are displayed, ignore all others
    """
    if aa:
        mut_list = node.aa_mutations
    else:
        mut_list = node.mutations

    if display_positions:
        return "_".join(
            [
                mut[1] + str(mut[0] + 1) + mut[2]
                for mut in mut_list
                if mut[0] in display_positions
            ]
        )
    else:
        return "_".join([mut[1] + str(mut[0] + 1) + mut[2] for mut in mut_list])


def collapse_zero_branches(clade):
    """
    reduces branchings with identical ML sequences into multiple mergers
    """
    to_collapse = []
    for child in clade.clades:
        if child.is_terminal() == False:
            collapse_zero_branches(child)
            if str(child.seq) == str(clade.seq):
                to_collapse.append(child)
    for child in to_collapse:
        print("collapsing", child)
        clade.collapse(child)


def annotate_leaf(leaf, annotation):
    """
    adds all known attribute to the leaf
    parameters:
    leaf        --  node of a tree
    annotation  --  the annotation dictionary that corresponds to this node
    """
    for attr, val in annotation.items():
        leaf.__setattr__(attr, val)


def translate_sequences_on_tree(T, cds={"begin": 0, "pad": 0, "end": 0}):
    """
    loops over all nodes and translates nucleotide sequences
    keyword argument:
    cds  -- dictionary specifying begin, end, and possible missing part of a coding region
            the missing part of length cds['pad'] is filled with X
    """
    if verbose:
        tmp_time = time.time()
        print("translating_sequences_on_tree: ...", end=" ")
    for node in T.get_nonterminals() + T.get_terminals():
        if cds["end"] > 0:
            last_base = cds["end"]
        else:
            last_base = len(node.seq) + cds["end"]
        tmp_seq = node.seq[cds["begin"] : last_base]
        try:
            try:
                node.aa_seq = "X" * cds["pad"] + tmp_seq.translate()
            except:
                from Bio import Seq

                node.aa_seq = Seq.Seq(
                    "X" * (cds["pad"] + (last_base - cds["begin"]) / 3)
                )
                print(
                    "Error translating:",
                    node.name,
                    "replaced sequences by XXX...XXX",
                )
        except AttributeError as e:
            print("translate_sequences_on_tree: ", node.name, e)
    if verbose:
        print("done in", np.round(time.time() - tmp_time), "s")


def mutations_on_branches(T, aa=False):
    """
    compares sequences at the beginning and end of each branch and lists mutations
    recursively calls itself. creates an attribute mutations
    argument:
    T  --  Tree
    keyword arguments:
    aa_seq  --  if true, will look for aa_seq and also create an attribute aa_mutations
    """
    if verbose:
        tmp_time = time.time()
        print("mutations_on_branches: find all mutations ...", end=" ")
    T.root.mutations = []
    if aa:
        T.root.aa_mutations = []
    mutations_on_branches_subtree(T.root, aa)
    if verbose:
        print("done in", np.round(time.time() - tmp_time), "s")


def mutations_on_branches_subtree(subtree, aa=False):
    """
    compares sequences at the beginning and end of each branch and lists mutations
    recursively calls itself. creates an attribute mutations
    argument:
    subtree   --  clade of Biopython tree
    keyword arguments:
    aa_seq  --  if true, will look for aa_seq and also create an attribute aa_mutations
    """
    for child in subtree.clades:
        try:
            # compare to sequences nucleotide wise and make a list of differences
            mutation_pos = np.where(
                np.array(subtree.seq) != np.array(child.seq)
            )[0]
            child.mutations = [
                (pos, subtree.seq[pos], child.seq[pos]) for pos in mutation_pos
            ]
            # repeat for amino acids
            if aa:
                aa_mutation_pos = np.where(
                    np.array(subtree.aa_seq) != np.array(child.aa_seq)
                )[0]
                child.aa_mutations = [
                    (pos, subtree.aa_seq[pos], child.aa_seq[pos])
                    for pos in aa_mutation_pos
                    if subtree.aa_seq[pos] != "X" and child.aa_seq[pos] != "X"
                ]

            # go through the entire tree recursively
            if child.is_terminal() == False:
                mutations_on_branches_subtree(child, aa)
        except AttributeError as e:
            print("mutations_on_branches_subtree:", child.name, e)


def find_internal_nodes(sourceT, destT):
    """
    takes two trees and identifies internal nodes. finds the most recent internal node
    in destT that is ancestral to all terminal nodes of each internal node in sourceT
    and links the two nodes
    parameters:
    sourceT --  Biopython tree
    destT   --  Biopython tree
    """
    destT_leaf_lookup = {leaf.name: leaf for leaf in destT.get_terminals()}
    path_in_destT = {
        leaf.name: [destT.root]
        + destT.root.get_path(destT_leaf_lookup[leaf.name])
        for leaf in sourceT.get_terminals()
    }

    # set all mirror nodes to None, those that have actual mirror nodes will be set later
    for intnode in destT.get_nonterminals():
        intnode.mirror_node = None

    # go over all internal nodes in the source tree and find MRCA of its leafs
    # in the destination tree.
    for intnode in sourceT.get_nonterminals():
        tmp_path = [
            path_in_destT[leaf.name] for leaf in intnode.get_terminals()
        ]
        min_length = min([len(a) for a in tmp_path])
        pos = 0
        while pos < min_length and all(
            [tmp_path[0][pos] == p[pos] for p in tmp_path[1:]]
        ):
            pos += 1
        dest_intnode = tmp_path[0][pos - 1]
        dest_intnode.mirror_node = intnode
        try:
            dest_intnode.color = intnode.color
            dest_intnode.name = intnode.name
        except:
            pass
        if dest_intnode.is_terminal():
            print(pos)
            print(dest_intnode)
            for p in tmp_path:
                print(p)
        intnode.mirror_node = dest_intnode


##################################################
# UTILITY FUNCTION FOR TREE ANNOTAION, COLORING AND PLOTTING
##################################################


def label_nodes(T, seqs_to_label):
    """
    erases all prior node labels and labels only those that are provided
    in the dictionary seqs_to_label.
    """
    for node in T.get_terminals() + T.get_nonterminals():
        node.label = ""
    for node in T.get_terminals() + T.get_nonterminals():
        if node.name in seqs_to_label:
            node.label += seqs_to_label[node.name]


def node_label_func(node):
    """
    function that is passed to the tree drawing function
    """
    try:
        return node.name
    except:
        return None


def erase_color(tree):
    """
    set color fields of all nodes to None
    """
    for node in tree.get_terminals() + tree.get_nonterminals():
        node.color = None


def plot_prediction_tree(
    prediction,
    method="mean_fitness",
    internal=False,
    fig=None,
    axes=None,
    cb=True,
    scalebar=True,
    offset=0.0001,
    node_label_func=lambda x: None,
):
    """
    plot the tree used to predict. Color according to prediction
    """
    from matplotlib import cm

    if internal:
        nodes = prediction.non_terminals
    else:
        nodes = prediction.terminals
    prediction.color_tree(
        method=method,
        offset=offset,
        n_labels=1,
        cmap=my_favorite_cmap,
        nodes=nodes,
    )
    if not internal:
        prediction.interpolate_color()
    ax = draw_tree(
        prediction.T,
        fig=fig,
        axes=axes,
        node_label=node_label_func,
        branch_label=node_label_func,
        cb=cb,
        scalebar=scalebar,
    )


def draw_tree(
    tree,
    node_label=node_label_func,
    branch_label=node_label_func,
    cmap=cm.jet,
    fig=None,
    axes=None,
    cb=True,
    scalebar=True,
):
    """
    plots a tree on an empty canvas including a scalebar of length 0.05
    """
    import matplotlib.pyplot as plt
    from Bio import Phylo

    if axes is None:
        fig = plt.figure(figsize=(8, 6))
        axes = plt.subplot(111)

    Phylo.draw(
        tree,
        label_func=node_label,
        show_confidence=False,
        branch_labels=lambda x: None,
        axes=axes,
        do_show=False,
    )
    #               show_confidence = False,branch_labels = branch_label, axes=axes, do_show=False)
    axes.axis("off")

    if scalebar:
        xlimits = axes.get_xlim()
        ylimits = axes.get_ylim()
        x0 = xlimits[0] + (xlimits[1] - xlimits[0]) * 0.05
        x1 = x0 + 0.05
        y0 = ylimits[0] + (ylimits[1] - ylimits[0]) * 0.05
        plt.plot([x0, x1], [y0, y0], lw=2, c="k")
        plt.text(x0 + 0.025, y0 + (ylimits[1] - y0) * 0.01, "0.05", ha="center")

    # fake a colorbar
    if cb:
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100)
        )
        sm._A = []
        cbar = fig.colorbar(
            sm, ticks=[0, 100], shrink=0.5, aspect=10, pad=-0.05
        )
        cbar.set_ticklabels(["worst", "best"])
    #    plt.draw()
    plt.show()
    return axes


def plot_combined_tree(
    prediction,
    combined_data,
    method="mean_fitness",
    internal=False,
    axes=None,
    cb=True,
    offset=0.0001,
):
    """
    plot the tree of the combined prediction and test data.
    """
    if internal:
        nodes = combined_data.T.get_nonterminals()
    else:
        nodes = combined_data.T.get_terminals()

    prediction.color_other_tree(
        nodes, method, offset=offset, n_labels=1, cmap=my_favorite_cmap
    )
    if not internal:
        prediction.interpolate_color(combined_data.T)
    ax = draw_tree(combined_data.T, ax=axes, cb=cb)
