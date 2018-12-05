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
sys.path.append("/home/mattjones/TargetSeqAnalysis/process")
import lineageGroup_utils as lg_utils

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

    with open(outputdir + "/log_pickalleles.txt", "w") as f:
        f.write("INT BC MAPING LOG:\n")

    return outputdir



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

def analyze_overlap(oMat, outputdir):
    """
    Function to analyze overlap and produce basic QC information for downstream analysis.
    :param oMat: overlap matrix
    :param outputdir: file path to output directory
    :return: Dissimilarity Matrix
    """

    dm = 1 - oMat

    # remove old plots
    plt.close()

    h = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 2, 1)
    x1 = plt.hist(oMat)
    ax2 = plt.title("overlap mat")
    ax = plt.subplot(1, 2, 2)
    x1 = plt.hist(dm)
    ax3 = plt.title("distance mat")

    plt.savefig(outputdir + "/overlap_hist.png")


    return dm


def plot_heatmap(dm, xlink, ind1, outputdir, outSizeLen, outSizeHeight, maxD=1):
    """
    Function to produce the clustered heatmap. Taken from Michelle Chan's singleClusterAndPlotX function.

    :param dm:
    :param xlink:
    :param ind1:
    :param outSizeLen:
    :param outSizeHeight:
    :param maxD:
    :return:
    """

    # remove old plots
    plt.close()

    fClustThresh = 0.5
    h = plt.figure(1, figsize=(18,10))
    color_bar_w = 0.015
    cmap = "RdBu"

    # ax1, placement of dendrogram 1
    [ax1_x, ax1_y, ax1_w, ax1_h] = [0.05, 0.22, 0.2, 0.6]
    width_btn_ax1_axr = 0.004
    height_btn_ax1_axc = 0.004 # dst btn top color bar and mat

    # axr, placement of row side colorbar
    [axr_x, axr_y, axr_w, axr_h] = [0.31, 0.22, color_bar_w, 0.6]
    axr_x = ax1_x + ax1_w + width_btn_ax1_axr
    axc_y = ax1_y
    axc_h = ax1_h
    width_btn_axr_axm = 0.004

    # axc, placement of column side colorbar
    [axc_x, axc_y, axc_w, axc_h] = [0.4, 0.63, 0.5, color_bar_w]
    axc_x = axr_x + axr_w + width_btn_axr_axm
    axc_y = ax1_y + ax1_h + height_btn_ax1_axc
    height_btn_axc_ax2 = 0.004

    # axm, placement of heatmap
    [axm_x, axm_y, axm_w, axm_h] = [0.4, 0.9, 2.5, 0.5]
    axm_x = axr_x + axr_w + width_btn_axr_axm
    axm_y = ax1_y
    axm_h = ax1_h
    axm_w = 0.5

    # ax2, placement of dendrogram 2, on top of heatmap
    [ax2_x, ax2_y, ax2_w, ax2_h] = [0.3, 0.72, 0.6, 0.15]
    ax2_x = axr_x + axr_w + width_btn_axr_axm
    ax2_y = ax1_y + ax1_h + height_btn_ax1_axc + axc_h + height_btn_axc_ax2
    ax2_w = axc_w

    # axcb = placement of color legend
    [axcb_x, axcb_y, axcb_w, axcb_h] = [0.05, 0.88, 0.18, 0.09]

    # compute and plot left dendrogram
    ax1 = h.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True)
    sqfrmDistMat = sp.spatial.distance.squareform(dm)
    xdendro = sp.cluster.hierarchy.dendrogram(xlink, orientation='left', color_threshold=maxD)
    ax1.set_yticks([])

    ax2 = h.add_axes([ax2_x, ax2_y, ax2_w, ax2_h], frame_on=True)
    xdendro2 = sp.cluster.hierarchy.dendrogram(xlink, color_threshold=maxD)
    ax2.set_xticks([])

    axm = h.add_axes([axm_x, axm_y, axm_w, axm_h])
    idx1 = xdendro['leaves']
    im = axm.matshow(sqfrmDistMat[:, idx1][idx1], aspect='auto', origin='lower', cmap=cmap)
    axm.set_xticks([])
    axm.set_yticks([])

    axr = h.add_axes([axr_x, axr_y, axr_w, axr_h])
    dr = np.array(ind1[idx1], dtype=int)
    dr.shape = (len(ind1), 1)
    cmap_r = colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
    im_r = axr.matshow(dr, aspect='auto', origin='lower', cmap=cmap_r)
    axr.set_xticks([])
    axr.set_yticks([])

    axc = h.add_axes([axc_x, axc_y, axc_w, axc_h])
    dr.shape = (1, len(ind1))
    im_c = axc.matshow(dr, aspect='auto', origin='lower', cmap=cmap_r)
    axc.set_xticks([])
    axc.set_yticks([])

    axcb = h.add_axes([axcb_x, axcb_y, axcb_w, axcb_h], frame_on=False)
    cb = colorbar.ColorbarBase(axcb, cmap=cmap, orientation='horizontal')
    axcb.set_title("colorbar")

    plt.savefig(outputdir + "/alleleTable.filtered_lineageGroups.png")

def assign_lineage_groups(alleletable, ind1, df_pivot_I, outputdir):
    """
    Assign cells in the allele table to a lineage group

    :param alleletable: allele table
    :param ind1: clusterings
    :param df_pivot_I: binary pivot table relating cell BC to integration BC
    :return: allele table with lineage group assignments
    """

    alleletable["lineageGrp"] = -1
    lineageGrp2cellBC = dict(zip(df_pivot_I.index.tolist(), ind1))

    alleletable["lineageGrp"] = alleletable["cellBC"].map(lineageGrp2cellBC)

    #for cell_i in tqdm(lineageGrp2cellBC):
    #    cellBC_i = cell_i[1]
    #    LG_i = cell_i[0]
    #    alleletable.loc[alleletable["cellBC"] == cellBC_i, "lineageGrp"] = LG_i

    LGs, LG_counts = np.unique(ind1, return_counts=True)
    LG2cellCount = dict(zip(LGs, LG_counts))

    with open(outputdir + "/lglog.txt", "a") as f:
        for key in LG2cellCount.keys():
           f.write("Lineage Group " + str(key) + ": " + str(LG2cellCount[key]) + " cellBCs\n")

    return alleletable

def cluster_with_dtcut(ds, xlink, minClusterSize=1):
    base = importr("base")
    stats = importr("stats")
    dtc = importr("dynamicTreeCut")
    # mat = importr("Matrix")
    numpy2ri.activate()

    r = robjects.r
    base.gc()
    dmat = robjects.vectors.Matrix(ds)
    hc = r.hclust(stats.as_dist(dmat), method="average")

    clusters = dtc.cutreeDynamic(hc, distM = dmat,  minClusterSize=minClusterSize,
                                     respectSmallClusters=False, pamRespectsDendro=False, deepSplit = 0)
    return np.array(clusters)

def plot_lg_rank_iBCProps(df, curr_LG, title, output_file = None):

    # Get cells in curr_LG
    lg_group = df[df["lineageGrp"] == curr_LG]

    # get unique intBC list
    intbcs = np.unique(lg_group["intBC"])
    intbc_counter = pd.DataFrame({"iBC": intbcs, "counter": [0]*len(intbcs)})
    for cb in np.unique(lg_group["cellBC"]):
        # get unique intbcs in this cell
        mols = np.unique(lg_group.loc[lg_group["cellBC"] == cb, "intBC"])
        intbc_counter.loc[(intbc_counter["iBC"]).isin(mols), "counter"] += 1

    n_cells = len(np.unique(lg_group["cellBC"]))

    intbc_counter["counter"] /= n_cells
    intbc_counter = intbc_counter.sort_values(by="counter", ascending=False)
    intbc_counter.index = intbc_counter["iBC"]

    plt.figure()
    intbc_counter[:30].plot(kind="bar")
    plt.title(title)
    plt.tight_layout()


    if output_file is not None:
        plt.savefig(output_file)

    plt.close()


def plot_intbc_ranks(clusters, df, outputdir):

    if not os.path.exists(outputdir + "/rank_intbc_hists"):
        os.makedirs(outputdir + "/rank_intbc_hists")

    lineageGrps = np.unique(clusters)

    for lg in tqdm(clusters):
        name = "rank_iBC_prop_lg" + str(lg) + ".png"
        plot_lg_rank_iBCProps(df, lg, "Lineage Group " + str(lg), output_file = outputdir + "/rank_intbc_hists/" + name)


def plot_overlap_heatmap(df_pivot_I, clusters, outputdir, cell_xlink):

    # remove old plots
    plt.close()

    oMatIntBC = np.asarray(lg_utils.maxOverlap(df_pivot_I))
    dmIntBC = 1-oMatIntBC
    xlinkIntBC = sp.cluster.hierarchy.linkage(dmIntBC,method='average')
    xdendro = sp.cluster.hierarchy.dendrogram(xlinkIntBC,no_plot=True)

    idxIntBC = xdendro["leaves"]
    idxIntBC1 = df_pivot_I.columns[idxIntBC]

    cell_dendro = sp.cluster.hierarchy.dendrogram(cell_xlink, no_plot=True)
    idxcells = cell_dendro["leaves"]
    idxcells1 = df_pivot_I.index[idxcells]

    clusters = np.array(clusters)

    idx1 = np.argsort(clusters)

    idxCells = df_pivot_I.index[idx1]

    lg = clusters[idx1]
    lg.shape = (len(lg), 1)

    lineageGrps = np.unique(clusters)

    cmapDis = rand_cmap(len(lineageGrps), type='bright', first_color_black=True, last_color_black=False, verbose=False)

    h = plt.figure(figsize=(14,10))

    ax1 = h.add_axes([0.15, 0.05, 0.6, 0.9],frame_on=True)
    im = ax1.matshow(df_pivot_I.loc[idxCells,idxIntBC1], aspect='auto',origin='lower',cmap='Greys',vmin=0,vmax=1)
    ax1.set_yticks([])
    ax1.set_title("Clustered IntBC, Max Overlap")

    ax2 = h.add_axes([0.95,0.05,0.02,0.9],frame_on=True)
    im_1 = ax2.matshow(lg,aspect='auto',origin='lower',cmap=cmapDis)

    # ax3 = h.add_axes([0.15, 0.9, 0.6, 0.1])
    # xdendro = sp.cluster.hierarchy.dendrogram(xlinkIntBC, no_labels=True, color_threshold=2)
    # ax3.set_yticks([])
    # ax3.set_xticks([])
    #
    # ax4 = h.add_axes([0.05, 0.05, 0.1, 0.9])
    # xdendro2 = sp.cluster.hierarchy.dendrogram(cell_xlink, no_labels=True, color_threshold=2, orientation="left")
    # ax4.set_yticks([])
    # ax4.set_xticks([])


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

def plot_overlap_heatmap_lg(lgs, at_pivot_I, outputdir):

    if not os.path.exists(outputdir + "/lineageGrp_piv_heatmaps"):
        os.makedirs(outputdir + "/lineageGrp_piv_heatmaps")

    for i in tqdm(range(len(lgs))):

        plt.close()

        lg_group = add_cutsite_encoding(lgs[i])

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
        plt.savefig(outputdir + "/lineageGrp_piv_heatmaps/lg_" + str(i) + "_piv_heatmap.png")
        plt.close()


def collectAlleles(df, at_piv, clusters, outputdir, thresh = 0.05):

    lineageGrps = np.unique(clusters)

    lgs = []

    for i in tqdm(lineageGrps):
        lg_group, intbc_counter = get_lg_group(df, at_piv, i)

        p_bc = intbc_counter[(intbc_counter["prop"] > thresh) & (intbc_counter["iBC"] != "NC")]

        lg_group = lg_group.loc[np.in1d(lg_group["intBC"], p_bc["iBC"])]
        lgs.append(lg_group)

    return lgs

def filteredLG2AT(filtered_lgs):

    final_df = pd.concat(filtered_lgs)
    print(final_df.columns)

    final_df = final_df.groupby(["cellBC", "intBC", "allele", "r1", "r2", "r3", "r1.old", "r2.old", "r3.old", "lineageGrp"], as_index=False).agg({"UMI": "count"})
    final_df["Sample"] = "N"

    for i in final_df.index:
        final_df.loc[i, "Sample"] = final_df.loc[i, "cellBC"].split(".")[0]

    return final_df

def filter_low_prop_intBCs(at_pivot_I, at_pivot, outputdir, thresh = 0.0025, bycell=True):

    num_umi = at_pivot.values.sum()

    bc_cell_props = at_pivot_I.apply(lambda x: pylab.sum(x) / len(x))
    bc_umi_props = at_pivot.apply(lambda x: pylab.sum(x) / num_umi)
    to_keep = bc_cell_props[bc_cell_props > thresh].index

    h = plt.figure(figsize=(10,10))
    data = bc_cell_props.sort_values(ascending=False)
    ax = plt.plot(np.arange(data.shape[0]), data)
    plt.ylabel("Proportion")
    plt.yscale("log")
    plt.xscale('log')
    plt.title("Integration Barcode Proportion by Cell")
    plt.axhline(y=thresh, color="red", linestyle="dashed")
    plt.xticks([], [])
    plt.savefig(outputdir + "/intBC_cell_prop_ALL.png")
    plt.close()

    #h = plt.figure(figsize=(10,10))
    #ax = bc_umi_props.sort_values(ascending=False).plot(kind = "bar")
    #plt.ylabel("Proportion")
    #plt.title("Integration Barcode Proportion by UMI")
    #plt.xticks([],[])
    #plt.savefig(outputdir + "/intBC_umi_prop_ALL.png")
    #plt.close()

    #h = plt.figure(figsize=(10,10))
    #ax = plt.plot(bc_cell_props, bc_umi_props[bc_cell_props.index], "r.")
    #plt.loglog()
    #plt.ylabel("Molecule Proportion")
    #plt.xlabel("Cell Proportion")
    #plt.title("UMI vs Cell Proportion of IntBCs")
    #plt.savefig(outputdir + "/intBC_cell_vs_umi_prop.png")
    #plt.close()


    return at_pivot_I[to_keep]


if __name__ == "__main__":

    # Read in parameters
    inp = sys.argv[1]
    with open(inp, "r") as stream:
        params = yaml.load(stream)

    alleleTable_fp = params["sample_file"][0]
    output_fp = params["output_file"][0]
    outputdir = params["output_dir"][0]
    min_cluster_size = params["min_cluster_size"][0]
    min_intbc_thresh = params["min_intbc_thresh"][0]
    verbose = params["verbose"][0]
    filter_intbcs = params["filter_intbcs"][0]
    detect_doublets = params["detect_doublets"][0]


    t0 = time.time()

    outputdir = create_output_dir(outputdir)

    print(">>> READING IN ALLELE TABLE...")
    at = pd.read_csv(alleleTable_fp, sep='\t')

    if "allele" not in at.columns:
        at["allele"] = at.apply(lambda x: x["r1"] + x["r2"] + x["r3"], axis=1)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write(str(at.shape[0]) + " UMIs (rows), with " + str(at.shape[1]) + " attributes (columns)\n")
        f.write(str(len(at["cellBC"].unique())) + " Cells")

    
    if detect_doublets:
        prop = 0.35
        print(">>> FILTERING OUT INTRA-LINEAGE GROUP DOUBLETS WITH PROP "  + str(prop) + "...")
        at = lg_utils.filter_intra_doublets(at, outputdir, prop = prop)

    print(">>> MAPPING REMAINING INTEGRATION BARCODE CONFLICTS...")
    at = lg_utils.mapIntBCs(at, outputdir)
    print(at.columns)

    print(">>> CREATING PIVOT TABLE...")
    at_pivot = pd.pivot_table(at, index=["cellBC"], columns=["intBC"], values="UMI", aggfunc=pylab.size)

    at_pivot_I = at_pivot
    at_pivot_I[at_pivot_I > 0] = 1
    at_pivot_I.fillna(value=0)

    if filter_intbcs:
        print(">>> FILTERING OUT LOW PROPORTION INTEGRATION BARCODES...")
        at_pivot_I = filter_low_prop_intBCs(at_pivot_I, at_pivot, outputdir, thresh = 0.001)


    #print(">>> SUBSAMPLING 30% FOR MEMORY EFFICIENCY...")
    #at_pivot_I = at_pivot_I.sample(frac = 0.30)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write(str(at_pivot_I.shape[0]) + " cells and " + str(at_pivot_I.shape[1]) + " intBCs\n")

    print(">>> CALCULATING MAXIMUM OVERLAP MATRIX...")
    oMat = np.asarray(lg_utils.maxOverlap(at_pivot_I.T))

    #print(">>> SAVING OVERLAP MATRIX...")
    #np.savetxt(outputdir + "/overlap_matrix.txt", oMat, delimiter='\t') 

    print(">>> LOGGING CELL OVERLAP INFORMATION...")
    dm = analyze_overlap(oMat, outputdir)

    print(">>> CLUSTERING CELLS BASED ON OVERLAP...")
    xlink = sp.cluster.hierarchy.linkage(dm, method="average")

    print(">>> CALCULATING COPHENETIC DISTANCES...")
    c, coph_dists = sp.cluster.hierarchy.cophenet(xlink, dm)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write("Cophenetic Correlation Distance: " + str(c) + "\n")

    ds = sp.spatial.distance.squareform(dm)

    min_cluster_size = int(0.005 * at_pivot_I.shape[0])
    print(">>> CLUSTERING DENDROGRAM WITH MINIMUM CLUSTER SIZE OF " + str(min_cluster_size) + "...")
    clusters = cluster_with_dtcut(ds, xlink, minClusterSize=min_cluster_size)


    lineageGrps = np.unique(clusters)
    with open(outputdir + "/lglog.txt", "a") as f:
        f.write(str(len(lineageGrps)) + " Lineage Groups\n")

    print(">>> ASSIGNING LINEAGE GROUPS...")
    at = assign_lineage_groups(at, clusters, at_pivot_I, outputdir)

    if verbose:
        print(">>> PLOTTING INTBC COUNT HISTOGRAM...")
        plot_intBC_count_hist(clusters, at, outputdir)

        print(">>> PLOTTING INTBC RANK PROPORTION HISTOGRAMS...")
        plot_intbc_ranks(clusters, at, outputdir)

    print(">>> COLLECTING ALLELES...")
    filtered_lgs = collectAlleles(at, at_pivot_I, clusters, outputdir, thresh = min_intbc_thresh)

    print(">>> PRODUCING FINAL ALLELE TABLE...")
    at = filteredLG2AT(filtered_lgs)
    at.to_csv(outputdir + "/" + output_fp, sep='\t', index=False)

    print(">>> PRODUCING CLUSTERED HEATMAP...")
    plot_heatmap(dm, xlink, clusters, outputdir, 10, 10, maxD=2)

    print(">>> PRODUCING PIVOT TABLE HEATMAP...")
    plot_overlap_heatmap(at_pivot_I, clusters, outputdir, xlink)

    print(">>> PLOTTING FILTERED LINEAGE GROUP PIVOT TABLE HEATMAPS...")
    plot_overlap_heatmap_lg(filtered_lgs, at_pivot_I, outputdir)

    with open(outputdir + "/lglog.txt", "a") as f:
        f.write("Final allele table written to " + outputdir + "/" + output_fp + "\n")
        f.write("Total time: " + str(time.time() - t0))
