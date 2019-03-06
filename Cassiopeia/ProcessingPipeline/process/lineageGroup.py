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
import networkx as nx

import argparse 

sys.setrecursionlimit(10000)
from . import lineageGroup_utils as lg_utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# NOTE: NEED PANDAS >= 0.22.0

def create_output_dir(outputdir = None):
    """
    A simple  function to create an output directory to store important logging information,
    as well as important figures for qc
    """

    if outputdir is None:
        i = 1
        outputdir = "output" + str(i)

        while os.path.exists(os.path.dirname(outputdir)):

            i += 1
            outputdir = "output" + str(i)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    with open(outputdir + "/lglog.txt", "w") as f:
        f.write("LINEAGE GROUP OUTPUT LOG:\n")

    return outputdir

def findTopLG(PIVOT_in, iteration, outputdir, min_intbc_prop = 0.2, kinship_thresh=0.2):

    # calculate sum of observed intBCs, identify top intBC
    intBC_sums = PIVOT_in.sum(0).sort_values(ascending=False)
    ordered_intBCs = intBC_sums.index.tolist()
    intBC_top = intBC_sums.index[0]


    # take subset of PIVOT table that contain cells that have the top intBC
    subPIVOT_in = PIVOT_in[PIVOT_in[intBC_top]>0]
    subPIVOT_in_sums = subPIVOT_in.sum(0)
    ordered_intBCs2 = subPIVOT_in_sums.sort_values(ascending=False).index.tolist()
    subPIVOT_in = subPIVOT_in[ordered_intBCs2]
    
    # binarize
    subPIVOT_in[subPIVOT_in>0]=1


    # Define intBC set
    subPIVOT_in_sums2 = subPIVOT_in.sum(0)
    total = subPIVOT_in_sums2[intBC_top]
    intBC_sums_filt = subPIVOT_in_sums2[subPIVOT_in_sums2>=min_intbc_prop*total]
    
    # Reduce PIV to only intBCs considered in set
    intBC_set = intBC_sums_filt.index.tolist()
    PIV_set = PIVOT_in.iloc[:,PIVOT_in.columns.isin(intBC_set)]

    # Calculate fraction of UMIs within intBC_set ("kinship") for each cell in PIV_set
    f_inset = PIV_set.sum(axis=1)

    # define set of cells with good kinship
    f_inset_filt = f_inset[f_inset>=kinship_thresh]
    LG_cells = f_inset_filt.index.tolist()

    # Return updated PIV with LG_cells removed
    PIV_noLG = PIVOT_in.iloc[~PIVOT_in.index.isin(LG_cells),:]

    # Return PIV with LG_cells assigned
    PIV_LG = PIVOT_in.iloc[PIVOT_in.index.isin(LG_cells),:]
    PIV_LG["lineageGrp"]= iteration+1
    
    with open(outputdir + "/lglog.txt", "a") as f:
        # print statements
        f.write("LG"+str(iteration+1)+" Assignment: " + str(PIV_LG.shape[0]) +  " cells assigned\n")
    
    # Plot distribution of kinship scores
    h4 = plt.figure(figsize=(15,10))
    ax4 = plt.hist(f_inset, bins=49, alpha=0.5, histtype='step')
    yax4 = plt.yscale('log', basey=10)
    plt.savefig(outputdir + "/kinship_scores.png")
 
    return PIV_LG, PIV_noLG, intBC_set

def iterative_lg_assign(pivot_in, min_clust_size, outputdir, min_intbc_thresh=0.2, kinship_thresh=0.2):
    ## Run LG Assign function

    # initiate output variables
    PIV_assigned = pd.DataFrame()
    master_intBC_list = []

    # Loop for iteratively assigning LGs
    prev_clust_size = np.inf

    i = 0
    while prev_clust_size > min_clust_size:
        # run function
        PIV_outs = findTopLG(pivot_in, i, outputdir, min_intbc_prop=min_intbc_thresh, kinship_thresh=kinship_thresh)
        
        # parse returned objects
        PIV_LG = PIV_outs[0]
        PIV_noLG = PIV_outs[1]
        intBC_set_i = PIV_outs[2]
        
        # append returned objects to output variables
        PIV_assigned = PIV_assigned.append(PIV_LG)
        master_intBC_list.append(intBC_set_i)
        
        # update PIVOT-in
        pivot_in = PIV_noLG

        prev_clust_size = PIV_LG.shape[0]

        i += 1

    return PIV_assigned, master_intBC_list

def get_lg_group(df, piv, curr_LG):

    lg_group = df[df["lineageGrp"] == curr_LG]
    cells = np.unique(lg_group["cellBC"])

    lg_pivot = piv.loc[cells]

    props = lg_pivot.apply(lambda x: pylab.sum(x) / len(x)).to_frame().reset_index()
    props.columns = ["iBC", "prop"]

    props = props.sort_values(by="prop", ascending=False)
    props.index = props["iBC"]

    return lg_group, props

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def assign_lineage_groups(dfMT, max_kinship_LG, master_intBCs):
    """
    Assign cells in the allele table to a lineage group

    :param alleletable: allele table
    :param ind1: clusterings
    :param df_pivot_I: binary pivot table relating cell BC to integration BC
    :return: allele table with lineage group assignments
    """

    dfMT["lineageGrp"]=0

    cellBC2LG = {}
    for n in max_kinship_LG.index:
        cellBC2LG[n] = max_kinship_LG.loc[n, "lineageGrp"]


    dfMT["lineageGrp"] = dfMT["cellBC"].map(cellBC2LG)

    dfMT["lineageGrp"] = dfMT["lineageGrp"].fillna(value=0)

    lg_sizes = {}
    rename_lg = {}

    for n, g in dfMT.groupby(["lineageGrp"]):

        if n != 0:
            lg_sizes[n] = len(g["cellBC"].unique())

    sorted_by_value = sorted(lg_sizes.items(), key = lambda kv: kv[1])[::-1]

    for i, tup in zip(range(1, len(sorted_by_value)+1), sorted_by_value):
        print(i, tup[0], float(i))
        rename_lg[tup[0]] = float(i)

    rename_lg[0] = 0.0

    dfMT["lineageGrp"] = dfMT.apply(lambda x: rename_lg[x.lineageGrp], axis=1)

    return dfMT



def plot_overlap_heatmap(at_pivot_I, at, outputdir):

    # remove old plots
    plt.close()

    flat_master = []
    for n, lg in at.groupby("lineageGrp"):

        for item in lg["intBC"].unique():
            flat_master.append(item)

    at_pivot_I = at_pivot_I[flat_master]

    h2 = plt.figure(figsize=(20,20))
    axmat2 = h2.add_axes([0.3,0.1,0.6,0.8])
    im2 = axmat2.matshow(at_pivot_I, aspect='auto', origin='upper')

    plt.savefig(outputdir + "/clustered_intbc.png")
    plt.close()

def add_cutsite_encoding(lg_group):

    lg_group["s1"] = 0
    lg_group["s2"] = 0
    lg_group["s3"] = 0


    for i in lg_group.index:
        if lg_group.loc[i, "r1"] == "['None']":
            lg_group.loc[i, "s1"] = .9
        elif "D" in lg_group.loc[i, "r1"]:
            lg_group.loc[i, "s1"] = 1.9
        elif 'I' in lg_group.loc[i, "r1"]:
            lg_group.loc[i, 's1'] = 2.9

        if lg_group.loc[i, "r2"] == "['None']":
            lg_group.loc[i, "s2"] = .9
        elif "D" in lg_group.loc[i, "r2"]:
            lg_group.loc[i, "s2"] = 1.9
        elif 'I' in lg_group.loc[i, "r2"]:
            lg_group.loc[i, 's2'] = 2.9

        if lg_group.loc[i, "r3"] == "['None']":
            lg_group.loc[i, "s3"] = .9
        elif "D" in lg_group.loc[i, "r3"]:
            lg_group.loc[i, "s3"] = 1.9
        elif 'I' in lg_group.loc[i, "r3"]:
            lg_group.loc[i, 's3'] = 2.9

    return lg_group

def plot_overlap_heatmap_lg(at, at_pivot_I, outputdir):

    if not os.path.exists(outputdir + "/lineageGrp_piv_heatmaps"):
        os.makedirs(outputdir + "/lineageGrp_piv_heatmaps")

    for n, lg_group in  tqdm(at.groupby("lineageGrp")):

        plt.close()

        lg_group = add_cutsite_encoding(lg_group)

        s_cmap = colors.ListedColormap(['grey', 'red', 'blue'], N=3)

        lg_group_pivot = pd.pivot_table(lg_group, index=["cellBC"], columns=["intBC"], values=['s1', 's2', 's3'], aggfunc=pylab.mean).T
        lg_group_pivot2 = pd.pivot_table(lg_group,index=['cellBC'],columns=['intBC'],values='UMI',aggfunc=pylab.size)

        cell_umi_count = lg_group.groupby(["cellBC"]).agg({"UMI": "count"}).sort_values(by="UMI")
        n_unique_alleles = lg_group.groupby(["intBC"]).agg({"r1": "nunique", "r2": "nunique", "r3": "nunique"})

        cellBCList = lg_group["cellBC"].unique()

        col_order = lg_group_pivot2.dropna(axis=1, how="all").sum().sort_values(ascending=False,inplace=False).index

        if len(col_order) < 2:
            continue

        s3 = lg_group_pivot.unstack(level=0).T
        s3 = s3[col_order]
        s3 = s3.T.stack(level=1).T

        s3 = s3.loc[cell_umi_count.index]

        s3_2 = lg_group_pivot2.dropna(axis=1, how="all").sum().sort_values(ascending=False, inplace=False)[col_order]

        n_unique_alleles = n_unique_alleles.loc[col_order]
        s3_intBCs = col_order
        s3_cellBCs = s3.index.tolist()


        # Plot heatmap
        h = plt.figure(figsize=(14,10))

        ax = h.add_axes([0.3, 0.1, 0.6, 0.8],frame_on=True)
        im = ax.matshow(s3, aspect='auto', origin ="lower", cmap=s_cmap)

        axx1 = plt.xticks(range(1, len(col_order)*3, 3), col_order, rotation='vertical', family="monospace")

        ax3 = h.add_axes([0.2, 0.1, 0.1, 0.8], frame_on=True)
        plt.barh(range(s3.shape[0]), cell_umi_count["UMI"])
        plt.ylim([0, s3.shape[0]])
        ax3.autoscale(tight=True)


        axy0 = ax3.set_yticks(range(len(s3_cellBCs)))
        axy1 = ax3.set_yticklabels(s3_cellBCs, family='monospace')

        w = (1/3)
        x = np.arange(len(s3_intBCs))
        ax2 = h.add_axes([0.3, 0, 0.6, 0.1], frame_on = False)
        b1 = ax2.bar(x - w, n_unique_alleles["r1"], width = w, label="r1")
        b2 = ax2.bar(x, n_unique_alleles["r2"], width = w, label="r2")
        b3 = ax2.bar(x + w, n_unique_alleles["r3"], width = w, label='r3')
        ax2.set_xlim([0, len(s3_intBCs)])
        ax2.set_ylim(ymin=0, ymax=(max(n_unique_alleles["r1"].max(), n_unique_alleles["r2"].max(), n_unique_alleles["r3"].max()) + 10))
        ax2.set_xticks([])
        ax2.yaxis.tick_right()
        ax2.invert_yaxis()
        ax2.autoscale(tight=True)
        plt.legend()

        #plt.gcf().subplots_adjust(bottom=0.15)
        plt.tight_layout()
        plt.savefig(outputdir + "/lineageGrp_piv_heatmaps/lg_" + str(int(n)) + "_piv_heatmap.png")
        plt.close()


def collectAlleles(at, thresh = 0.05):

    lineageGrps = at["lineageGrp"].unique()
    at_piv = pd.pivot_table(at, index="cellBC", columns="intBC", values="UMI", aggfunc="count")
    at_piv.fillna(value = 0, inplace=True)
    at_piv[at_piv > 0] = 1


    lgs = []

    for i in tqdm(lineageGrps):

        lg = at[at["lineageGrp"] == i]
        cells = lg["cellBC"].unique()

        lg_pivot = at_piv.loc[cells]

        props = lg_pivot.apply(lambda x: pylab.sum(x) / len(x)).to_frame().reset_index()
        props.columns = ["iBC", "prop"]

        props = props.sort_values(by="prop", ascending=False)
        props.index = props["iBC"]
        
        p_bc = props[(props["prop"] > thresh) & (props["iBC"] != "NC")]

        lg_group = lg.loc[np.in1d(lg["intBC"], p_bc["iBC"])]
        lgs.append(lg_group)

    return lgs

def filteredLG2AT(filtered_lgs):

    final_df = pd.concat(filtered_lgs)

    final_df = final_df.groupby(["cellBC", "intBC", "allele", "r1", "r2", "r3", "r1.old", "r2.old", "r3.old", "lineageGrp"], as_index=False).agg({"UMI": "count", "readCount": "sum"})

    final_df["Sample"] = final_df.apply(lambda x: x.cellBC.split(".")[0], axis=1)

    return final_df

def filter_low_prop_intBCs(PIV_assigned, thresh = 0.2):

    master_intBCs = {}
    master_LGs = []

    for i, PIV_i in PIV_assigned.groupby(["lineageGrp"]):
        PIVi_bin = PIV_i.copy()
        PIVi_bin = PIVi_bin.drop(['lineageGrp'], axis=1) # drop the lineageGroup column
        PIVi_bin[PIVi_bin>0]=1

        intBC_sums = PIVi_bin.sum(0)
        ordered_intBCs = intBC_sums.sort_values(ascending=False).index.tolist()
        intBC_normsums = intBC_sums/max(intBC_sums)
        
        intBC_normsums_filt_i = intBC_normsums[intBC_normsums >= thresh]
        intBC_set_i = intBC_normsums_filt_i.index.tolist()
        
        # update masters
        master_intBCs[i] = intBC_set_i
        master_LGs.append(i)
    
    return master_LGs, master_intBCs

def filterCellBCs(moleculetable, outputdir, umiCountThresh = 10, verbose=True):
    """
    Filter out cell barcodes thmt have too few UMIs

    :param moleculetable: allele table
    :param outputdir: file pmth to output directory
    :return: filtered allele table, cellBC to number umis mapping
    """

    if verbose:
        with open(outputdir + "/lglog.txt", "a") as f:
            f.write("FILTER CELL BARCODES:\n")

            f.write("Initial:\n")
            f.write("# UMIs: " + str(moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(moleculetable["cellBC"]))) + "\n")

    tooFewUMI_UMI = []
    cellBC2nM = {}

    # Create a cell-filter dictionary for hash lookup lmter on when filling
    # in the table
    cell_filter = {}

    for n, group in tqdm(moleculetable.groupby(["cellBC"])):
        if np.sum(group["UMI"].values) <= umiCountThresh:
            cell_filter[n] = "bad"
            tooFewUMI_UMI.append(np.sum(group["UMI"].values))
        else:
            cell_filter[n] = "good"
            cellBC2nM[n] = np.sum(group["UMI"].values)

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
        with open(outputdir + "/lglog.txt", "a") as f:
            f.write("Post:\n")
            f.write("# UMIs: " + str(n_moleculetable.shape[0]) + "\n")
            f.write("# Cell BCs: " + str(len(np.unique(n_moleculetable["cellBC"]))) + "\n\n")


    return n_moleculetable, cellBC2nM

def merge_lineage_groups(at, outputdir, thresh=0.3):

    lg_intbc_piv = pd.pivot_table(at, index="lineageGrp", columns=["intBC"], values="UMI", aggfunc="count")
    lg_intbc_piv[lg_intbc_piv > 0] = 1
    lg_intbc_piv.fillna(value=0)

    lg_oMat = np.asarray(lg_utils.maxOverlap(lg_intbc_piv.T))

    lg_oMat = sp.spatial.distance.squareform(lg_oMat)

    for i in range(lg_oMat.shape[0]):
        lg_oMat[i, i] = 1.0

    to_collapse = []
    for i in range(lg_oMat.shape[0]):
        for j in range(i+1, lg_oMat.shape[0]):
            if lg_oMat[i, j] > thresh:
                coll = (i, j)
                to_collapse.append(coll)

    collapse_net = nx.Graph()
    for pair in to_collapse:

        collapse_net.add_edge(pair[0], pair[1])

    num_lg = len(at["lineageGrp"].unique())
    cc = list(nx.connected_components(collapse_net))
    for i, c in zip(range(1, len(cc)+1), cc):
    
        for n in c:

            at.loc[at["lineageGrp"] == n, "lineageGrp" ]= i + num_lg 

    lg_sizes = {}
    rename_lg = {}

    for n, g in at.groupby(["lineageGrp"]):

        lg_sizes[n] = len(g["cellBC"].unique())

    sorted_by_value = sorted(lg_sizes.items(), key = lambda kv: kv[1])[::-1]

    for i, tup in zip(range(len(sorted_by_value)), sorted_by_value):
    
        rename_lg[tup[0]] = float(i)

    at["lineageGrp"] = at.apply(lambda x: rename_lg[x.lineageGrp], axis=1)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write("Collapsing the following lineage groups:\n")
        for coll in to_collapse:
            f.write(str(coll) + "\n")

    return at

def filter_cells_by_kinship_scores(PIV, master_LGs, master_intBCs, outputdir):

    dfLG2intBC = pd.DataFrame()

    for i in range(len(master_LGs)):
        LGi = master_LGs[i]
        intBCsi = master_intBCs[LGi]
        dfi = pd.DataFrame(index=[LGi], columns=intBCsi, data=1)
        dfLG2intBC = dfLG2intBC.append(dfi,'sort=False')

    dfLG2intBC = dfLG2intBC.fillna(0)

    # reorder
    flat_master_intBCs = []
    intBC_dupl_check = set()
    for key in master_intBCs.keys():
        sublist = master_intBCs[key]
        for item in sublist:
            if item not in intBC_dupl_check:
                flat_master_intBCs.append(item)
                intBC_dupl_check.add(item)

    dfLG2intBC = dfLG2intBC[flat_master_intBCs]

    # Construct matrices for multiplication
    ## subPIVOT (cellBC vs. intBC, value = freq)
    subPIVOT = PIV[flat_master_intBCs]
    subPIVOT = subPIVOT.fillna(0)

    # Matrix math
    dfCellBC2LG = subPIVOT.dot(dfLG2intBC.T)
    max_kinship = dfCellBC2LG.max(axis=1)

    max_kinship_ind = dfCellBC2LG.idxmax(axis=1).to_frame()
    max_kinship_frame = max_kinship.to_frame()

    max_kinship_LG = pd.concat([max_kinship_frame, max_kinship_ind+1], axis=1)
    max_kinship_LG.columns = ["maxOverlap","lineageGrp"]

    #max_kinship_LG_filt = max_kinship_LG[max_kinship_LG['maxOverlap'] >= 0.75]

    #with open(outputdir + "/lglog.txt", "a") as f:
    #    f.write(str(max_kinship_LG.shape[0] - max_kinship_LG_filt.shape[0]) + " cells filtered by kinship\n")

    return max_kinship_LG
    


def main():

    # Read in parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('molecule_table', type=str, help="MoleculeTable to be processed")
    parser.add_argument('output_fp', type=str, help="Output name for AlleleTable, to be saved in output directory")
    parser.add_argument("output_dir", type=str, help="File path to output directory for all results")
    parser.add_argument("--min_cluster_prop", default=0.005, help="Minimum proportion of cells that can fall into a cluster for lineage group calling") 
    parser.add_argument("--min_intbc_thresh", default=0.05, help="Threshold to filter out poor intBC per LineageGroup, as a function of the proportion of cells that report that intBC in the LG")
    parser.add_argument("--kinship_thresh", default = 0.25, help="Threshold by which to exclude cells during lineage group calling, based on their overlap (or kinship) of intBCs in that lineage group.")
    parser.add_argument("--detect_doublets_inter", default=False, action='store_true', help="Perform Inter-Doublet (from different  LGs) Detection")
    parser.add_argument("--doublet_threshold", default=0.35, help="Threshold at which to call intra-doublets")
    parser.add_argument("--verbose", "-v", default=False, action="store_true", help="Verbose output")
    parser.add_argument("--cell_umi_filter", default=10, help="Minimum UMIs per cell for final alleleTable")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot summaries at end of process")

    args = parser.parse_args()

    alleleTable_fp = args.molecule_table
    output_fp = args.output_fp
    outputdir = args.output_dir
    min_cluster_prop = float(args.min_cluster_prop)
    min_intbc_thresh = float(args.min_intbc_thresh)
    kinship_thresh = float(args.kinship_thresh)
    verbose = args.verbose
    detect_doublets = args.detect_doublets_inter
    doublet_thresh = float(args.doublet_threshold)
    cell_umi_filter = int(args.cell_umi_filter)
    plot = args.plot
 
    t0 = time.time()

    outputdir = create_output_dir(outputdir)

    print(">>> READING IN ALLELE TABLE...")
    mt = pd.read_csv(alleleTable_fp, sep='\t')

    if "allele" not in mt.columns:
        mt["allele"] = mt.apply(lambda x: x["r1"] + x["r2"] + x["r3"], axis=1)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write(str(mt.shape[0]) + " UMIs (rows), with " + str(mt.shape[1]) + " attributes (columns)\n")
        f.write(str(len(mt["cellBC"].unique())) + " Cells\n")

    PIV = pd.pivot_table(mt, index="cellBC",columns="intBC", values="UMI", aggfunc="count")
    PIV = PIV.div(PIV.sum(axis=1), axis=0)

    # reorder PIV columns by binarized intBC frequency
    PIVbin = PIV.copy()
    PIVbin[PIVbin>0]=1
    intBC_sums = PIVbin.sum(0)
    ordered_intBCs = intBC_sums.sort_values(ascending=False).index.tolist()
    PIV = PIV[ordered_intBCs]

    min_clust_size = int(min_cluster_prop * PIV.shape[0])
    print(">>> CLUSTERING WITH MINIMUM CLUSTER SIZE " + str(min_clust_size) + "...")
    PIV_assigned, master_intBC_list = iterative_lg_assign(PIV, min_clust_size, outputdir, min_intbc_thresh=min_intbc_thresh, kinship_thresh=kinship_thresh)

    print(">>> FILTERING OUT LOW PROPORTION INTBCs...")
    master_LGs, master_intBCs = filter_low_prop_intBCs(PIV_assigned, thresh = min_intbc_thresh)

    print(">>> COMPUTING KINSHIP MATRIX...")
    kinship_scores = filter_cells_by_kinship_scores(PIV_assigned, master_LGs, master_intBCs, outputdir)

    print(">>> ASSIGNING LINEAGE GROUPS...")
    at = assign_lineage_groups(mt, kinship_scores, master_intBCs)

    if detect_doublets:
        prop = doublet_thresh
        print(">>> FILTERING OUT INTRA-LINEAGE GROUP DOUBLETS WITH PROP "  + str(prop) + "...")
        at = lg_utils.filter_inter_doublets(at, "lglog.txt", outputdir, rule = prop)

    filtered_lgs = collectAlleles(at, thresh = min_intbc_thresh)

    at = filteredLG2AT(filtered_lgs)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write("Final LG assignments:\n")

        for n, g in at.groupby(["lineageGrp"]):
            f.write("LG " + str(n) + ": " + str(len(g["cellBC"].unique())) + " cells\n")

    print(">>> FILTERING OUT LOW-UMI CELLS...")
    at, cell2BCnM = filterCellBCs(at, outputdir, umiCountThresh=int(cell_umi_filter), verbose=verbose)

    print(">>> WRITING OUTPUT...")
    at.to_csv(outputdir + "/" + output_fp, sep='\t', index=False)

    if plot:
        print(">>> PRODUCING PLOTS...")
        at_piv = pd.pivot_table(at, index="cellBC", columns="intBC", values="UMI", aggfunc="count")
        at_pivot_I = at_piv
        at_pivot_I.fillna(value = 0, inplace=True)
        at_pivot_I[at_pivot_I > 0] = 1

        clusters = at[["cellBC", "lineageGrp"]].drop_duplicates()["lineageGrp"]

        print(">>> PRODUCING PIVOT TABLE HEATMAP...")
        plot_overlap_heatmap(at_pivot_I, at, outputdir)

        print(">>> PLOTTING FILTERED LINEAGE GROUP PIVOT TABLE HEATMAPS...")
        plot_overlap_heatmap_lg(at, at_pivot_I, outputdir)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write("Final allele table written to " + outputdir + "/" + output_fp + "\n")
        f.write("Total time: " + str(time.time() - t0))
