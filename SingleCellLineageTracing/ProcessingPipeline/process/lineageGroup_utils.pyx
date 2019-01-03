# Separate function file for some heavy lifting during lineage group calling. Wrapped in Cython, this should
# make things a lot faster. 

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import pylab
from matplotlib import colors, colorbar
from scipy import cluster
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from tqdm import tqdm
from rpy2.robjects import r, numpy2ri
import time
import yaml

sys.setrecursionlimit(10000)

def maxOverlap(mat):
    """
    Function to compute overlap of barcode occurences between column entries of the input matrix.

    :param mat: intBC x Cell binary pivot table (each entry is 1 iff the intBC occurred in the cell)
    :return: Maximum overlap matrix -- i.e. maximum percentage of integration barcodes shared between each pair of cells
    """

    cdef int m
    cdef double[:] dm
    cdef int k
    cdef double[:] mSum
    cdef int i
    cdef int j
    # Get number of columns of matrix
    m = mat.shape[1]

    # Instantiate matrix for storing max overlap between all intBCs
    dm = np.zeros(m * (m-1) // 2, dtype=float)

    # Count total number of non-NAs (i.e. total number of intBC's across cells)
    mat[np.isnan(mat)] = 0
    mat = mat.values
    mSum = np.sum(mat, axis=0)

    dmat = mat.T.dot(mat)

    # For each intBC, let's compute the
    k = 0
    for i in tqdm(range(m - 1), desc="Computing max overlap for every cell"):
        for j in range(i + 1, m):

            # count up total number of barcodes that are shared between each cell

            # Number of barcodes in the cell with fewer barcodes between cells i and j respectively
            mSize = min(mSum[i], mSum[j])
            if mSize > 0:

                # calculate the fraction of barcodes shared between cells
                dm[k] = dmat[i, j] / float(mSize)

            k += 1
    return dm

def mapIntBCs(moleculetable, outputfile, outputdir, verbose=True):
    """
    Performs a procedure to cleanly assign one allele to each intBC/cellBC pairing

    :param moleculetable: Allele table to be analyzed.
    :param outputdir: output directory file pmth
    :return: cleaned moleculetable.
    """

    iBC_assign = {}
    r1_assign = {}
    r2_assign = {}
    r3_assign = {}

    # Have to drop out all intBCs that are NaN
    moleculetable = moleculetable.dropna(subset=["intBC"])

    # create mappings from intBC/cellBC pairs to alleles
    moleculetable["status"] = "good"
    moleculetable["filter_column"] = moleculetable[["intBC", "cellBC"]].apply(lambda x: '_'.join(x), axis=1)
    moleculetable["filter_column2"] = moleculetable[["intBC", "cellBC", "allele"]].apply(lambda x: "_".join(x), axis=1)
    moleculetable["allele_counter"] = moleculetable["allele"]

    filter_dict = {}

    # For each intBC/cellBC pair, we want only one allele (select majority allele for now)
    corrected = 0
    numUMI_corrected = 0
    for n, group in tqdm(moleculetable.groupby(["filter_column"])):

        x1 = group.groupby(["filter_column2", "allele"]).agg({"readCount": "sum", "allele_counter": "count", "UMI": "count"}).sort_values("readCount", ascending=False).reset_index()

        # If we've found an intBC that corresponds to more than one allele in the same cell, then let's error correct towards
        # the more frequently occuring allele

        # But, this will ALWAYS be the first allele because we sorted above, so we can generalize and always assign the intBC to the
        # first element in x1.

        a = x1.iloc[0]["allele"]

        # Let's still keep count of how many times we had to re-assign for logging purposes
        filter_dict[x1.iloc[0]["filter_column2"]] = "good"
        if x1.shape[0] > 1:

            for i in range(1, x1.shape[0]):
                filter_dict[x1.iloc[i]["filter_column2"]] = "bad"
                corrected += 1
                numUMI_corrected += x1.loc[i,"UMI"]


            if verbose:
                for i in range(1, x1.shape[0]):
                    with open(outputdir + "/log_pickalleles.txt", "a") as f:
                        f.write(n + "\t" + x1.loc[i, "allele"] + "\t" + a + "\t")
                        f.write(str(x1.loc[i, "UMI"]) + "\t" + str(x1.loc[0, "UMI"]) + "\n")



    moleculetable["status"] = moleculetable["filter_column2"].map(filter_dict)
    moleculetable = moleculetable[(moleculetable["status"] == "good")]
    moleculetable.index = [i for i in range(moleculetable.shape[0])]
    moleculetable = moleculetable.drop(columns=["filter_column", "filter_column2", "allele_counter", "status"])


    # log results
    if verbose:
        with open(outputdir + "/" + outputfile, "a") as f:

            f.write("PICK ALLELES:\n")
            f.write("# alleles removed: " + str(corrected) + "\n")
            f.write("# UMIs affected (through removing alleles): " + str(numUMI_corrected) + "\n\n")

    return moleculetable

def filter_intra_doublets(mt, outputfile, outputdir, prop=0.35):
    """
    Filter doublets from the allele table AT.

    """
    doublet_list = []

    filter_dict = {}

    for n, g in tqdm(mt.groupby(["cellBC"]), desc="Filtering Intra-doublets"):

        x1 = g.groupby(["intBC", "allele"]).agg({"UMI": 'count', 'readCount': 'sum'}).sort_values("UMI", ascending=False).reset_index()

        doublet = False
        if x1.shape[0] > 0:

            for r1 in range(x1.shape[0]):

                iBC1, allele1 = x1.loc[r1, "intBC"], x1.loc[r1, "allele"]

                for r2 in range(r1 + 1, x1.shape[0]):

                    iBC2, allele2 = x1.loc[r2, "intBC"], x1.loc[r2, "allele"]

                    if iBC1 == iBC2 and allele1 != allele2:

                        totalCount = x1.loc[[r1, r2], "UMI"].sum()
                        props = x1.loc[[r1, r2], "UMI"] / totalCount
                        if props.iloc[1] >= prop:
                            filter_dict[n] = 'bad'
                            doublet = True
                            break
                if doublet:
                    break

        if not doublet:
            filter_dict[n] = "good"


    mt["status"] = mt["cellBC"].map(filter_dict)
    doublet_list = mt[(mt["status"] == "bad")]["cellBC"].unique()

    with open(outputdir + "/" + outputfile, "a") as f:
           f.write("Filtered " + str(len(doublet_list)) + " Intra-Lineage Group Doublets of " + str(len(mt["cellBC"].unique())) + "\n")

    mt = mt[(mt["status"] == "good")]
    mt = mt.drop(columns = ["status"])

    return mt

def get_intbc_sets(lgs, lg_names, thresh=None):

    intbc_sets = {}
    dropouts = {}

    for n, g in zip(lg_names, lgs):
        piv = pd.pivot_table(g, index="cellBC", columns="intBC", values="UMI", aggfunc=pylab.size)
        do = piv.apply(lambda x: x.isnull().sum() / float(piv.shape[0]), axis=0)

        if thresh is None:

            intbc_sets[n] = do.index
            dropouts[n] = do

        else:
            intbcs = do[do < thresh].index
            intbc_sets[n] = intbcs
            dropouts[n] = do

    return intbc_sets, dropouts

def compute_lg_membership(cell, intbc_sets, lg_dropouts, scale_by_dropout=True):
    
    lg_mem = {}
    ibcs = np.unique(cell["intBC"])

    for i in intbc_sets.keys():

        lg_do = lg_dropouts[i]
        intersect = np.intersect1d(ibcs, intbc_sets[i])
        if len(intersect) > 0:

            lg_mem[i] = np.sum(1 - lg_do[intersect]) / np.sum(1 - lg_do)

        else:
            lg_mem[i] = 0

    factor = 1.0 / np.sum(list(lg_mem.values()))
    for l in lg_mem:
        lg_mem[l] = lg_mem[l] * factor

    return lg_mem


def filter_inter_doublets(at, outputfile, outputdir, rule=0.35):

    def filter_cell(cell, rule):
        true_lg = cell.loc["LG"]
        return float(cell.loc[true_lg]) < rule

    collapsed_lgs = []
    lg_names = list(at["lineageGrp"].unique())

    for n in lg_names:
        collapsed_lgs.append(at[at["lineageGrp"] == n])

    ibc_sets, dropouts = get_intbc_sets(collapsed_lgs, lg_names)

    mems = {}
    for n, g in tqdm(at.groupby("cellBC"), desc="Creating Membership DF"):
        lg = int(g["lineageGrp"].iloc[0])
        mem = compute_lg_membership(g, ibc_sets, dropouts, scale_by_dropout = True)
        mem["LG"] = lg
        mems[n] = mem

    mem_df = pd.DataFrame.from_dict(mems).T

    filter_dict = {}

    for cell in tqdm(mem_df.index, desc="Identifying Inter doublets"):
        if filter_cell(mem_df.loc[cell], rule):
            filter_dict[cell] = "bad"
        else:
            filter_dict[cell] = "good"

    at["status"] = at["cellBC"].map(filter_dict)
    at_n = at[at["status"] == "good"]

    dl = len(at[at["status"] == "bad"]["cellBC"].unique())
    tot = len(at["cellBC"].unique())

    with open(outputdir + "/" + outputfile, "a") as f:
        f.write("Filtered " + str(dl) + " inter-doublets of " + str(tot) + " cells") 

    return at_n.drop(columns=["status"])

