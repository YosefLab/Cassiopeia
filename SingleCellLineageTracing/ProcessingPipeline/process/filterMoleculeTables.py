import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import Levenshtein
import time
from tqdm import tqdm
import argparse
import yaml

# suppress warnings for mapping thmt takes place in some filter functions
pd.options.mode.chained_assignment = None

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

    with open(outputdir + "/filterlog.txt", "w") as f:
        f.write("FILTER MOLECULE TABLE OUTPUT LOG:\n")

    with open(outputdir + "/eclog_umi.txt", "w") as f:
        f.write("CellBC\tUR\tUC\tN_READS(R)\tN_READS(C)\n")

    with open(outputdir + "/log_pickalleles.txt", "w") as f:
        f.write("IntBC\tAR\tAC\tN_UMI(R)\tN_UMI(C)\n")


    return outputdir

def record_stats(moleculetable, outputdir, stage="Init"):
    """
    Simple function to record the number of UMIs and create QC plots for UMIs

    :param moleculetable: allele table
    :param outputdir: file pmth to the output directory
    :return: Num of UMIs
    """

    # Log number of UMIs
    numis = moleculetable.shape[0]

    # Count number of intBCs per cellBC
    iBCs = moleculetable.groupby(["cellBC"]).agg({"intBC": "nunique"})["intBC"]

    # Count UMI per intBC
    umi_per_ibc = np.array([])
    for n, g in tqdm(moleculetable.groupby(["cellBC"])):
        x = g.groupby(["intBC"]).agg({"UMI": "nunique"})["UMI"]
        if x.shape[0] > 0:
            umi_per_ibc = np.concatenate([umi_per_ibc, np.array(x)])


    # Count UMI per cellBC
    umi_per_cbc = moleculetable.groupby(['cellBC']).agg({"UMI": 'count'}).sort_values("UMI", ascending=False)["UMI"]

    return np.array(moleculetable["readCount"]), umi_per_ibc, np.array(umi_per_cbc)


def filterCellBCs(moleculetable, outputdir, umiCountThresh = 10, verbose=True):
    """
    Filter out cell barcodes thmt have too few UMIs

    :param moleculetable: allele table
    :param outputdir: file pmth to output directory
    :return: filtered allele table, cellBC to number umis mapping
    """

    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("FILTER CELL BARCODES:\n")

            f.write("Initial:\n")
            f.write("# Reads: " + str(sum(moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(moleculetable["cellBC"]))) + "\n")

    tooFewUMI_UMI = []
    cellBC2nM = {}

    # Create a cell-filter dictionary for hash lookup lmter on when filling
    # in the table
    cell_filter = {}

    for n, group in tqdm(moleculetable.groupby(["cellBC"])):
        if group.shape[0] <= umiCountThresh:
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

    # filter based on status & reindex
    n_moleculetable = moleculetable[(moleculetable["status"] == "good")]
    n_moleculetable.index = [i for i in range(n_moleculetable.shape[0])]

    # log results
    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("Post:\n")
            f.write("# Reads: " + str(sum(n_moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(n_moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(n_moleculetable["cellBC"]))) + "\n\n")


    return n_moleculetable, cellBC2nM

def errorCorrectUMI(moleculetable, outputdir, bcDistThresh = 1, allelePropThresh = 0.2, verbose=True):
    """
    Error correct UMIs based on allele & cellBC informmtion

    :param moleculetable: moleculetable
    :return: allele table with corrected UMIs
    """

    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("ERROR CORRECT UMIs:\n")

            f.write("Initial:\n")
            f.write("# Reads: " + str(sum(moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(moleculetable["cellBC"]))) + "\n")

    num_UMI_corrected = 0
    num_reads_corrected = 0

    to_drop = np.array([])
    for n, g in tqdm(moleculetable.groupby(["cellBC"])):

        # Let's correct UMIs by alleles -- if the same cellBC/UMI pair map to the same allele, let's try to assign the correct UMI barcode.
        #x1 = group.groupby(["UMI", "allele", "intBC"]).agg({'readCount': 'sum'}).sort_values("readCount", ascending=False).reset_index()

        # If we have more than one UMI in a cellBC (this should definitely be true)
        g = g.sort_values("readCount", ascending=False).reset_index()
        if g.shape[0] > 0:
            corrected_r = []
            for r1 in range(g.shape[0]):

                uBC1, allele1, iBC1 = g.loc[r1, "UMI"], g.loc[r1, "allele"], g.loc[r1, "intBC"]

                for r2 in range(r1 + 1, g.shape[0]):

                    uBC2, allele2, iBC2 = g.loc[r2, "UMI"],  g.loc[r2, "allele"], g.loc[r2, "intBC"]

                    # Compute the levenshtein distance between both umis
                    bcl = Levenshtein.distance(uBC1, uBC2)

                    # If we've found two UMIs thmt are reasonably similar with the same allele and iBC, let's try to error correct.
                    if bcl <= bcDistThresh and allele1 == allele2 and iBC1 == iBC2:
                        totalCount = g.loc[[r1, r2], "readCount"].sum()

                        props = g.loc[[r1, r2], "readCount"] / totalCount

                        # Let's just error correct towards the more highly represented UMI iff the allele proportion of the lowly
                        # represented UMI is below some threshold
                        if props[r2] <= allelePropThresh and r1 not in corrected_r:

                            badlocs = moleculetable[(moleculetable["cellBC"] == n) & (moleculetable["UMI"] == uBC2)]
                            corrlocs = moleculetable[(moleculetable["cellBC"] == n) & (moleculetable["UMI"] == uBC1)]

                            corrected_r.append(r2)


                            if len(badlocs.index.values) > 0 and badlocs.index.values[0] in moleculetable.index:
                                moleculetable.loc[corrlocs.index.values, "readCount"] += badlocs["readCount"].iloc[0]

                            moleculetable = moleculetable.drop(badlocs.index.values)


                            #to_drop = np.concatenate((to_drop, badlocs.index.values))

                            num_UMI_corrected += 1
                            num_reads_corrected += g.loc[r2, "readCount"]

                            if verbose:
                                with open(outputdir + "/eclog_umi.txt", "a") as f:
                                    f.write(n + "\t" + uBC2 + "\t" + uBC1 + "\t")
                                    f.write(str(g.loc[r2, "readCount"]) + "\t" + str(g.loc[r1, "readCount"]) + "\n")


    # log results
    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("Post:\n")
            f.write("# Reads: " + str(sum(moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(moleculetable["cellBC"]))) + "\n\n")

    moleculetable.index = [i for i in range(moleculetable.shape[0])]
    return moleculetable

def filterUMIs(moleculetable, outputdir, readCountThresh=100, verbose=True):
    """
    Filter out low-read UMIs

    :param alleltable: allele table to be filtered
    :param readCountThresh: read count theshold on which to filter UMIs
    :return: filtered allele table
    """
    t0 = time.time()

    # log results
    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("FILTER UMIS:\n")
            f.write("Initial:\n")
            f.write("# Reads: " + str(sum(moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(moleculetable["cellBC"]))) + "\n")

    filteredReads = []


    # filter based on status & reindex
    n_moleculetable = moleculetable.loc[moleculetable["readCount"] > readCountThresh]
    n_moleculetable.index = [i for i in range(n_moleculetable.shape[0])]

    # log results
    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("Post:\n")
            f.write("# Reads: " + str(sum(n_moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(n_moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(n_moleculetable["cellBC"]))) + "\n\n")


    print("FILTER MOLECULE TIME: " + str(time.time() - t0))
    return n_moleculetable

def errorCorrectIntBC(moleculetable, outputdir, prop = 0.5, umiCountThresh = 10,
                        bcDistThresh = 1, verbose=True):
    """
    Filter integration barcodes by their alleles and UMI proportion.

    :param moleculetable: allele table
    :param outputdir: file path to output directory
    :param prop: proportion by which to filter integration barcodes
    :param umiCountThresh: maximum umi count for which to correct barcodes
    :param bcDistThresh: barcode distance threshold, to decide what's similar enough to error correct
    :param verbose: boolean, indicating whether or not to write to output
    :return: filtered allele table with bad UMIs thrown out/error corrected
    """

    # log results
    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("ERROR CORRECT INTBCs:\n")

            f.write("Initial:\n")
            f.write("# Reads: " + str(sum(moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(moleculetable["cellBC"]))) + "\n")

    # create index filter hash map
    index_filter = {}
    for n in moleculetable.index.values:
        index_filter[n] = "good"

    recovered = 0
    numUMI_corrected = 0
    for name, grp in tqdm(moleculetable.groupby(["cellBC"])):

        # name = cellBC
        # grp = moleculetable[cellBC = name]

        x1 = grp.groupby(["intBC", "allele"]).agg({"UMI": 'count', "readCount": 'sum'}).sort_values("UMI", ascending=False).reset_index()

        if x1.shape[0] > 1:

            badList = []
            for r1 in range(x1.shape[0]):

                iBC1, allele1 = x1.loc[r1, "intBC"], x1.loc[r1, "allele"]

                for r2 in range(r1 + 1, x1.shape[0]):

                    iBC2, allele2 = x1.loc[r2, "intBC"], x1.loc[r2, "allele"]

                    bclDist = Levenshtein.distance(iBC1, iBC2)

                    if bclDist <= bcDistThresh and allele1 == allele2:

                        totalCount = x1.loc[[r1, r2], "UMI"].sum()

                        props = x1.loc[[r1, r2], "UMI"] / totalCount
                        umiCounts = x1.loc[[r1, r2], "UMI"]

                        # if the alleles are the same and the proportions are good, then let's error correct
                        if props[r2] < prop and umiCounts[r2] <= umiCountThresh:
                            bad_locs = moleculetable[(moleculetable["cellBC"] == name) & (moleculetable["intBC"] == iBC2) &
                                                   (moleculetable["allele"] == allele2)]

                            recovered += 1
                            numUMI_corrected += len(bad_locs.index.values)
                            moleculetable.loc[bad_locs.index.values, "intBC"] = iBC1


                            if verbose:
                                with open(outputdir + "/eclog_intbc.txt", "a") as f:
                                    f.write(name + "\t" + iBC2 + "\t" + iBC1 + "\t")
                                    f.write(str(x1.loc[r2, "UMI"]) + "\t" + str(x1.loc[r1, "UMI"]) + "\n")


    # log data
    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:
            f.write("Post:\n")
            f.write("# Reads: " + str(sum(moleculetable["readCount"])) + "\n")
            f.write("# UMIs: " + str(moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(moleculetable["cellBC"]))) + "\n")
            f.write("# Corrected intBCs: " + str(recovered) + "\n")
            f.write("# Corrected UMIs (associated with intBCs): " + str(numUMI_corrected) + '\n\n')


    # return filtered allele table
    #n_moleculetable = moleculetable[(moleculetable["status"] == "good")]
    moleculetable.index = [i for i in range(moleculetable.shape[0])]
    return moleculetable

def pickAlleles(moleculetable, outputdir, verbose=True):
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

    # Have to drop out all intBCs thmt are NaN
    moleculetable = moleculetable.dropna(subset=["intBC"])

    # create mappings from intBC/cellBC pairs to alleles
    moleculetable["status"] = "good"
    moleculetable["filter_column"] = moleculetable[["intBC", "cellBC"]].apply(lambda x: '_'.join(x), axis=1)
    moleculetable["allele_counter"] = moleculetable["allele"]

    # For each intBC/cellBC pair, we want only one allele (select majority allele for now)
    corrected = 0
    numUMI_corrected = 0
    for n, group in tqdm(moleculetable.groupby(["filter_column"])):

        x1 = group.groupby(["filter_column", "allele"]).agg({"r1": "unique", "r2": "unique",
                                "readCount": "count", "r3": "unique", "allele_counter": "count", "UMI": "count"}).sort_values("allele_counter", ascending=False).reset_index()

        # If we've found an intBC that corresponds to more than one allele in the same cell, then let's error correct towards
        # the more frequently occuring allele

        # But, this will ALWAYS be the first allele because we sorted above, so we can generalize and always assign the intBC to the
        # first element in x1.

        a = x1.loc[0, "allele"]

        # Let's still keep count of how many times we had to re-assign for logging purposes
        if x1.shape[0] > 1:
            badlocs = moleculetable[(moleculetable["filter_column"] == n) & (moleculetable["allele"] != a)]
            moleculetable.loc[badlocs.index.values, "status"] = "bad"

            corrected += (x1.shape[0]-1)
            numUMI_corrected += sum(x1["UMI"]) - x1.loc[0, "UMI"]

            if verbose:
                for i in range(1, x1.shape[0]):
                    with open(outputdir + "/log_pickalleles.txt", "a") as f:
                        f.write(n + "\t" + x1.loc[i, "allele"] + "\t" + a + "\t")
                        f.write(str(x1.loc[i, "UMI"]) + "\t" + str(x1.loc[0, "UMI"]) + "\n")


    moleculetable = moleculetable[(moleculetable["status"] == "good")]
    moleculetable.index = [i for i in range(moleculetable.shape[0])]
    moleculetable = moleculetable.drop(columns=["filter_column", "allele_counter", "status"])


    # log results
    if verbose:
        with open(outputdir + "/filterlog.txt", "a") as f:

            f.write("PICK ALLELES:\n")
            f.write("# alleles removed: " + str(corrected) + "\n")
            f.write("# UMIs affected (through removing alleles): " + str(numUMI_corrected) + "\n\n")

    return moleculetable

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("molecule_table", type=str, help="Molecule Table to filter")
    parser.add_argument("out_fp", type=str, help="Output file name to be written in the output directory")
    parser.add_argument("outputdir", type=str, help="File path to output directory location")
    parser.add_argument("--cell_umi_thresh", default=10, help="Minimum number of UMIs per cell")
    parser.add_argument("--umi_read_thresh", default=None, help="Minimum number of reads per UMIs")
    parser.add_argument("--intbc_prop_thresh", default=0.5, help="Minimum Proportion of reads allowed for error correcting of intBCs")
    parser.add_argument("--intbc_umi_thresh", default=10, help='Maximum Number of UMIs allowed for a intBC to be error-corrected')
    parser.add_argument("--intbc_dist_thresh", default = 1, help="Maximum distance between intBC allowed for error-correction")
    parser.add_argument("--verbose", default=True, help="Enable verbose output")
    parser.add_argument("--ec_intbc", default=False, help="Error Correct Integration Barcodes")

    args = parser.parse_args()
    moleculetable_fp = args.molecule_table
    moleculetableFiltered_fp = args.out_fp
    outputdir = args.outputdir
    cell_umi_thresh = args.cell_umi_thresh
    umi_read_thresh = args.umi_read_thresh
    intbc_prop_thresh = args.intbc_prop_thresh
    intbc_umi_thresh = args.intbc_umi_thresh
    intbc_dist_thresh = args.intbc_dist_thresh
    error_correct_intbc = args.ec_intbc
    verbose = args.verbose

    outputdir = create_output_dir(outputdir)

    rc_profile, upi_profile, upc_profile = {}, {}, {}

    # Log time
    t0 = time.time()

    print(">>> READING DATA IN...")
    # read in allele table and apply desired transformmtions
    mt = pd.read_csv(moleculetable_fp, sep='\t')
    mt.sort_values("readCount", ascending=False)
    mt["allele"] = mt.apply(lambda row: row.r1 + row.r2 + row.r3, axis=1)
    mt["status"] = "good"

    print(">>> LOGGING INITIAL STATS...")
    # record umis for downstream analysis
    rc_profile["Init"], upi_profile["Init"], upc_profile["Init"] = record_stats(mt, outputdir)

    print(">>> FILTERING CELL BARCODES...")
    # filter cells by number of UMIs
    filtered_mt, cellBC2nM = filterCellBCs(mt, outputdir, umiCountThresh = cell_umi_thresh, verbose=verbose)
    rc_profile["CellFilter"], upi_profile["CellFilter"], upc_profile["CellFilter"] = record_stats(mt, outputdir)

    # Determine read threshold
    if umi_read_thresh == 'None':
        R = filtered_mt["readCount"]
        umi_read_thresh = np.percentile(R, 99) / 10

    # Filter UMIs based on read count
    print(">>> FILTERING UMIS WITH THRESH " + str(umi_read_thresh) + "...")
    filtered_mt = filterUMIs(filtered_mt, outputdir, readCountThresh = umi_read_thresh, verbose = verbose)
    rc_profile["Filtered_UMI"], upi_profile["Filtered_UMI"], upc_profile["Filtered_UMI"] = record_stats(filtered_mt, outputdir, stage="Filtered_UMI")

    print(">>> PROCESSING INTBCs...")
    # filter and error correct integrmtion barcodes by allele
    if error_correct_intbc:

        filtered_mt = errorCorrectIntBC(filtered_mt, outputdir, prop = intbc_prop_thresh, umiCountThresh = intbc_umi_thresh,
                            bcDistThresh = intbc_dist_thresh, verbose=verbose)

    rc_profile["Process_intBC"], upi_profile["Process_intBC"], upc_profile["Process_intBC"] = record_stats(filtered_mt, outputdir, stage="Process_intBC")

    # filter molecule table cells one more time
    filtered_mt, cellBC2nM = filterCellBCs(filtered_mt, outputdir, umiCountThresh = cell_umi_thresh, verbose=verbose)
    rc_profile["Final"], upi_profile["Final"], upc_profile["Final"] = record_stats(filtered_mt, outputdir)


    # Count total filtered cellBCs
    cellBC_count = 0
    for name, grp in filtered_mt.groupby(["cellBC"]):
        cellBC_count += 1

    print(">>> LOGGING FINAL STATS...")
    record_stats(filtered_mt, outputdir, stage="Filtered")

    stages = ["Init", "CellFilter", "Filtered_UMI", "Process_intBC", "Final"]

    # Plot Read Per UMI Histogram
    h = plt.figure(figsize=(14, 10))
    for n in stages:
        ax = plt.hist(rc_profile[n], label=n, histtype="step", log=True, bins=200)
    plt.legend()
    plt.ylabel("Frequency")
    plt.xlabel("Number of Reads")
    plt.title("Reads Per UMI")
    plt.savefig(outputdir + "/reads_per_umi.png")
    plt.close()

    h = plt.figure(figsize=(14, 10))
    for n in stages:
        ax = plt.plot(upc_profile[n], label=n)
    plt.legend()
    plt.ylabel("Number of UMIs")
    plt.xlabel("Rank Order")
    plt.xscale("log", basex=10)
    plt.yscale("log", basey=10)
    plt.title("UMIs per CellBC")
    plt.savefig(outputdir + "/umis_per_cellbc.png")
    plt.close()

    h = plt.figure(figsize=(14, 10))
    for n in stages:
        ax = plt.hist(upi_profile[n], label=n, histtype="step", log=True, bins=200)
    plt.legend()
    plt.ylabel("Frequency")
    plt.xlabel("Number of UMIs")
    plt.title("UMIs per intBC")
    plt.savefig(outputdir + "/umis_per_intbc.png")
    plt.close()


    with open(outputdir + "/filterlog.txt", "a") as f:
        f.write("Overall: " + str(cellBC_count) + " cells, with " + str(filtered_mt.shape[0]) + " UMIs\n")

    filtered_mt.to_csv(outputdir + "/" + moleculetableFiltered_fp, sep='\t', index=False)

    with open(outputdir + "/filterlog.txt", "a") as f:
        f.write("Saved file: " + outputdir + "/" + moleculetableFiltered_fp + "\n")
        f.write("Final Time: " + str(round(time.time() - t0, 2)) + " seconds\n")
