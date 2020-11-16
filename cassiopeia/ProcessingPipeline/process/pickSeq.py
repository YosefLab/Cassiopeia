import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import spatial
import pylab
import itertools
from scipy import cluster
from tqdm import tqdm
import yaml
import time

def create_output_dir(outputdir = None):
    """
    A simple  function to e an output directory to store important logging informmtion,
    as well as important figures for qc
    """

    if outputdir is None:
        i = 1
        outputdir = "output" + str(i)

        while os.path.exists(outputdir):

            i += 1
            outputdir = "output" + str(i)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    with open(outputdir + "/pickseq_log.txt", "w") as f:
        f.write("PICK SEQ OUTPUT LOG:\n")

    return outputdir


def pickSeq(moleculetable, outputdir, verbose=True):
    """
    Pick most abundant sequence from a set of equivalent reads (i.e. same cellBC and UMI)
    Output a single sequence per unique cellBC-UMI pair
    """

    mt_filter = {}
    total_numReads = {}
    top_reads = {}
    second_reads = {}
    first_reads = {}

    for n, group in tqdm(moleculetable.groupby(["cellBC", "UMI"])):
        if group.shape[0] == 1:
            good_readName = group["readName"].iloc[0]
            mt_filter[good_readName] = "good"
            total_numReads[good_readName] = group["readCount"]
            top_reads[good_readName] = group["readCount"]
            # second_reads[good_readName] = group["readCount"]
        else:
            group_sort = group.sort_values("readCount", ascending=False).reset_index()
            good_readName = group_sort["readName"].iloc[0]
            mt_filter[good_readName] = "good" # add the first entry (highest readCount) to "good"
        
            total_numReads[good_readName] = group_sort["readCount"].sum()
            top_reads[good_readName] = group_sort["readCount"].iloc[0]
            second_reads[good_readName] = group_sort["readCount"].iloc[1]
            first_reads[good_readName] = group_sort["readCount"].iloc[0]

            for i in range(1,group.shape[0]):
                bad_readName = group_sort["readName"].iloc[i]
                mt_filter[bad_readName] = "bad" # add the rest of the entry(ies) (lowest readCount(s)) to "bad"

    # apply the filter using the hash table created above
    moleculetable["status"] = moleculetable["readName"].map(mt_filter)

    # filter based on status & reindex
    n_moleculetable = moleculetable[(moleculetable["status"] == "good")]

    h = plt.figure(figsize=(14, 10))
    plt.plot(top_reads.values(), total_numReads.values(), "r.")
    plt.ylabel("Total Reads")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Total vs. Top Reads for Picked Sequence")
    plt.savefig(outputdir + "/total_vs_top_reads_pickSeq.png")
    plt.close()
    
    h = plt.figure(figsize=(14, 10))
    plt.plot(first_reads.values(), second_reads.values(), "r.")
    plt.ylabel("Number Reads for Second Best Sequence")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Second Best vs. Top Reads for Picked Sequence")
    plt.savefig(outputdir + "/second_vs_top_reads_pickSeq.png")
    plt.close()

    return n_moleculetable


def filterCellBCs(moleculetable, outputdir, umi_per_cellBC_thresh, avg_reads_per_UMI_thresh, verbose=True):
    # NEED TO MAKE THE THRESHOLD FOR UMI/cellBC and reads/cellBC more dynamic!
    """
    Filter out cell barcodes that have too few UMIs or too few reads/UMI
    """

    tooFewUMI_UMI = []
    cellBC2nM = {}

    # Create a cell-filter dictionary for hash lookup later on when filling
    # in the table
    cell_filter = {}

    for n, group in tqdm(moleculetable.groupby(["cellBC"])):
        umi_per_cellBC_n = group.shape[0]
        reads_per_cellBC_n = group.agg({"readCount":'sum'}).readCount
        avg_reads_per_UMI_n = float(reads_per_cellBC_n)/float(umi_per_cellBC_n)
        if (umi_per_cellBC_n <= umi_per_cellBC_thresh) or (avg_reads_per_UMI_n <= avg_reads_per_UMI_thresh):
            cell_filter[n] = "bad"
            tooFewUMI_UMI.append(group.shape[0])
        else:
            cell_filter[n] = "good"
            cellBC2nM[n] = group.shape[0]

    # apply the filter using the hash table created above
    moleculetable["status"] = moleculetable["cellBC"].map(cell_filter)

    # count how many cells/umi's passed the filter for logging purposes
    status = cell_filter.values()
    tooFewUMI_cellBC = len(status) - len(np.where(status == "good")[0])
    tooFewUMI_UMI = np.sum(tooFewUMI_UMI)
    goodumis = moleculetable[(moleculetable["status"] == "good")].shape[0]

    # return filtered allele table
    n_moleculetable = moleculetable[(moleculetable["status"] == "good")]
    if verbose:
        with open(outputdir + "/pickseq_log.txt", "a") as f:
            f.write("Kept " + str(len(n_moleculetable.groupby(["cellBC"]))) + " cellBCs, with " + str(goodumis) + " UMIs\n")
            f.write("Removed " + str(tooFewUMI_cellBC) + " cellBCs, with " + str(tooFewUMI_UMI) + " UMIs\n")

    return n_moleculetable

def change_id(mt, sampleName):

    mt["status"] = "na"

    return mt

if __name__ == "__main__":

    # Read in parameters
    inp = sys.argv[1]
    with open(inp, "r") as stream:
        try:
            param = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception(exc)

    moleculetable_fp = param["sample_file"][0]
    sample_name = str(param["sample_name"][0])
    moleculetableFiltered_fp = param["output_file"][0]
    outputdir = param["output_dir"][0]
    cell_umi_thresh = param["cell_umi_thresh"][0]
    avg_reads_per_UMI_thresh = param["umi_read_thresh"][0]
    verbose = param["verbose"][0]

    outputdir = create_output_dir(outputdir)

    # Log time
    t0 = time.time()

    print(">>> READING DATA IN...")
    # read in allele table and apply desired transformmtions
    mt = pd.read_csv(moleculetable_fp, sep='\t')

    mt = change_id(mt, sample_name)

    print(">>> PLOTTING EQUIVALENCE CLASS READS...")
    equivClass_group = mt.groupby(["cellBC", "UMI"]).agg({"grpFlag": 'count'}).sort_values('grpFlag', ascending=False).reset_index()

    h = plt.figure(figsize=(8,5))
    ax = plt.hist(equivClass_group["grpFlag"], bins=range(1,equivClass_group["grpFlag"].max()))
    t = plt.title("Unique Seqs per cellBC+UMI")
    yax = plt.yscale('log', basey=10)
    plt.xlabel("Number of Unique Seqs")
    plt.ylabel("Count (Log)")
    plt.savefig(outputdir + "/" + "seqs_per_equivClass.png")

    print(">>> PERFORMING PICK SEQ...")
    mt = pickSeq(mt, outputdir)

    print(">>> FILTERING CELL BARCODES...")
    mt = filterCellBCs(mt, outputdir, cell_umi_thresh, avg_reads_per_UMI_thresh)

    print(">>> WRITING OUTPUT...")
    mt.to_csv(outputdir + "/" + moleculetableFiltered_fp, sep='\t')
