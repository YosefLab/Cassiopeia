"""
Basic utilities that can be used to generate visualizations of tree using the
iTOL tree plotting interface. See: https://itol.embl.de/ for more information
on the iTOL software and how to create an account.
"""

import os

import configparser
import shutil
import tempfile
from typing import Dict, List, Tuple, Optional

from bokeh import palettes
from itolapi import Itol, ItolExport
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cassiopeia.data import CassiopeiaTree
from cassiopeia.preprocess import utilities


class iTOLError(Exception):
    """Raises errors related to iTOL plotting
    """

    pass


def upload_and_export_itol(
    cassiopeia_tree: CassiopeiaTree,
    tree_name: str,
    export_filepath: str,
    itol_config: str = "~/.itolconfig",
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    meta_data: List[str] = [],
    allele_table: Optional[pd.DataFrame] = None,
    indel_colors: Optional[pd.DataFrame] = None,
    indel_priors: Optional[pd.DataFrame] = None,
    rect: bool = False,
    include_legend: bool = False,
    use_branch_lengths: bool = False,
    palette: List[str] = palettes.Category20[20],
    random_state: Optional[np.random.RandomState] = None,
    verbose: bool = True,
    **kwargs
):
    """Uploads a tree to iTOL and exports it.

    This function takes in a tree, plots it with iTOL, and exports it locally.
    A user can also specify meta data and allele tables to visualize alongside
    their tree. The function requires a user to have an account with iTOL and
    can pass these credentials to the function in one of two ways: first, the
    user can specify an `api_key` and a `project_name`, which corresponds to one
    in the user's iTOL account' second, the user can have a hidden `itolconfig`
    file that can store these credentials (the default location we will check
    is in ~/.itolconfig, though this can be overridden by the user). We
    preferentially take values passed in explicitly to the function in the
    `api_key` and `project_name` arguments.

    TODO(mgjones): Add the ability to pass in min/max colors for specific
        numeric meta data to use when creating the gradient files.

    Args:
        cassiopeia_tree: A CassiopeiaTree instance, populated with a tree.
        tree_name: Name of the tree. This is what the tree will be called
            within your project directory on iTOL
        export_filepath: Output file path to save your tree. Must end with
            one of the following suffixes: ['png', 'svg', 'eps', 'ps', 'pdf'].
        itol_config: A configuration file that a user can maintain for storing
            iTOL account details (specifically, an API key and a project name
            for uploading trees). We assume that the information is stored under
            the [DEFAULT] header. We also will be default check the path
            `~/itolconfig` but the user can pass in another path should they
            wish. If an `api_key` and `project_name` are also passed in, we
            will preferentially take those values.
        api_key: API key linking to your iTOL account
        project_name: Project name to upload to.
        meta_data: Meta data to plot alongside the tree, which must be columns
            in the CassiopeiaTree.cell_meta variable.
        allele_table: Alleletable to plot alongside the tree.
        indel_colors: Color mapping to use for plotting the alleles for each
            cell. Only necessary if `allele_table` is specified.
        indel_priors: Prior probabilities for each indel. Only useful if an
            allele table is to be plotted and `indel_colors` is None.
        rect: Boolean indicating whether or not to save your tree as a circle
            or rectangle.
        use_branch_lengths: Whether or not to use branch lengths when exporting
            the tree.
        include_legend: Plot legend along with meta data.
        palette: A palette of colors in hex format.
        random_state: A random state for reproducibility
        verbose: Include extra print statements.

    Raises:
        iTOLError if iTOL credentials cannot be found, if the output format is
            not supported, if meta data to be plotted cannot be found, or if
            an error with iTOL is encountered.
    """

    # create temporary directory for storing files we'll upload to iTOL
    temporary_directory = tempfile.mkdtemp()

    if (api_key is None or project_name is None):
        if os.path.exists(os.path.expanduser(itol_config)):

            config = configparser.ConfigParser()
            with open(os.path.expanduser(itol_config), "r") as f:
                config_string = f.read()
            config.read_string(config_string)

            try:
                api_key = config["DEFAULT"]["api_key"]
                project_name = config["DEFAULT"]["project_name"]
            except KeyError:
                raise iTOLError("Error reading the itol config file passed in.")

        else:
            raise iTOLError(
                "Specify an api_key and project_name, or a valid iTOL "
                "config file."
            )

    with open(os.path.join(temporary_directory, "tree_to_plot.tree"), "w") as f:
        f.write(cassiopeia_tree.get_newick(record_branch_lengths = True))

    file_format = export_filepath.split("/")[-1].split(".")[-1]

    if file_format not in ["png", "svg", "eps", "ps", "pdf"]:
        raise iTOLError(
            "File format must be one of " "'png', 'svg', 'eps', 'ps', 'pdf']"
        )

    itol_uploader = Itol()
    itol_uploader.add_file(
        os.path.join(temporary_directory, "tree_to_plot.tree")
    )

    files = []
    if allele_table is not None:
        files += create_indel_heatmap(
            allele_table,
            cassiopeia_tree,
            f"{tree_name}.allele",
            temporary_directory,
            indel_colors,
            indel_priors,
            random_state,
        )

    for meta_item in meta_data:
        if meta_item not in cassiopeia_tree.cell_meta.columns:
            raise iTOLError("Meta data item not in CassiopeiaTree cell meta.")

        values = cassiopeia_tree.cell_meta[meta_item]

        if pd.api.types.is_numeric_dtype(values):
            files.append(
                create_gradient_from_df(
                    values,
                    cassiopeia_tree,
                    f"{tree_name}.{meta_item}",
                    temporary_directory,
                )
            )

        if pd.api.types.is_string_dtype(values):
            colors = palette[: len(values.unique())]
            colors = [hex_to_rgb(color) for color in colors]
            colormap = dict(zip(np.unique(values), colors))

            files.append(
                create_colorbar(
                    values,
                    cassiopeia_tree,
                    colormap,
                    f"{tree_name}.{meta_item}",
                    temporary_directory,
                    create_legend=include_legend,
                )
            )

    for _file in files:
        itol_uploader.add_file(_file)

    itol_uploader.params["treeName"] = tree_name
    itol_uploader.params["APIkey"] = api_key
    itol_uploader.params["projectName"] = project_name

    good_upload = itol_uploader.upload()
    if not good_upload:
        raise iTOLError(itol_uploader.comm.upload_output)

    if verbose:
        print("iTOL output: " + str(itol_uploader.comm.upload_output))
        print("Tree Web Page URL: " + itol_uploader.get_webpage())
        print("Warnings: " + str(itol_uploader.comm.warnings))

    tree_id = itol_uploader.comm.tree_id

    itol_exporter = ItolExport()

    # set parameters:
    itol_exporter.set_export_param_value("tree", tree_id)
    itol_exporter.set_export_param_value("format", file_format)
    if rect:
        # rectangular tree settings
        itol_exporter.set_export_param_value("display_mode", 1)
    else:
        # circular tree settings
        itol_exporter.set_export_param_value("display_mode", 2)
        itol_exporter.set_export_param_value("arc", 359)
        itol_exporter.set_export_param_value("rotation", 270)

    itol_exporter.set_export_param_value("leaf_sorting", 1)
    itol_exporter.set_export_param_value("label_display", 0)
    itol_exporter.set_export_param_value("internal_marks", 0)
    itol_exporter.set_export_param_value(
        "ignore_branch_length", 1 - int(use_branch_lengths)
    )

    itol_exporter.set_export_param_value(
        "datasets_visible", ",".join([str(i) for i in range(len(files))])
    )

    itol_exporter.set_export_param_value("horizontal_scale_factor", 1)

    for key, value in kwargs.items():
        itol_exporter.set_export_param_value(key, value)

    # export!
    itol_exporter.export(export_filepath)

    # remove intermediate files
    shutil.rmtree(temporary_directory)


def create_gradient_from_df(
    df: pd.Series,
    tree: CassiopeiaTree,
    dataset_name: str,
    output_directory: str = "./tmp/",
    color_min: str = "#ffffff",
    color_max: str = "#000000",
) -> str:
    """Creates a gradient file for the iTOL batch uploader

    Creates a gradient file for iTOL from numerical data. This will write out
    the file to the specified location, which can then be uploaded to iTOL.

    Args:
        df: A pandas series with numerical data
        tree: CassiopeiaTree
        dataset_name: Name for the dataset
        output_directory: Where to write the output file
        color_min: Minimum color for gradient
        color_max: Maximum color for gradient

    Returns:
        The filepath to new gradient file.
    """

    _leaves = tree.leaves
    df = df.loc[_leaves].copy()

    outdf = pd.DataFrame()
    outdf["cellBC"] = _leaves
    outdf["gradient"] = df.values

    header = [
        "DATASET_GRADIENT",
        "SEPARATOR TAB",
        "COLOR\t#00000",
        f"COLOR_MIN\t{color_min}",
        f"COLOR_MAX\t{color_max}",
        "MARGIN\t100",
        f"DATASET_LABEL\t{df.name}",
        "STRIP_WIDTH\t50",
        "SHOW_INTERNAL\t0",
        "DATA",
        "",
    ]

    outfp = os.path.join(output_directory, f"{dataset_name}.{df.name}.txt")
    with open(outfp, "w") as fOut:
        for line in header:
            fOut.write(line + "\n")
        df_writeout = outdf.to_csv(None, sep="\t", header=False, index=False)
        fOut.write(df_writeout)
    return outfp


def create_colorbar(
    labels: pd.DataFrame,
    tree: CassiopeiaTree,
    colormap: Dict[str, Tuple[int, int, int]],
    dataset_name: str,
    output_directory: str = ".tmp/",
    create_legend: bool = False,
) -> str:
    """Creates a colorbar file for the iTOL batch uploader

    Creates a colorbar file for iTOL from categorical data. This will write out
    the file to the specified location, which can then be uploaded to iTOL.

    Args:
        labels: A pandas series with categorical data (can be represented as strings
            or categories)
        tree: CassiopeiaTree
        colormap: A mapping from category to RGB colors
        dataset_name: Name for the dataset
        output_directory: Where to write the output file
        create_legend: Include legend for this colorbar.

    Returns:
        The filepath to new colorbar file.
    """

    _leaves = tree.leaves
    labelcolors_iTOL = []
    for i in labels.loc[_leaves].values:
        colors_i = colormap[i]
        color_i = (
            "rgb("
            + str(colors_i[0])
            + ","
            + str(colors_i[1])
            + ","
            + str(colors_i[2])
            + ")"
        )
        labelcolors_iTOL.append(color_i)
    dfCellColor = pd.DataFrame()
    dfCellColor["cellBC"] = _leaves
    dfCellColor["color"] = labelcolors_iTOL

    # save file with header
    header = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        "COLOR\t#FF0000",
        "MARGIN\t100",
        f"DATASET_LABEL\t{dataset_name}",
        "STRIP_WIDTH\t100",
        "SHOW_INTERNAL\t0",
        "",
    ]

    outfp = os.path.join(output_directory, f"{dataset_name}.txt")
    with open(outfp, "w") as SIDout:
        for line in header:
            SIDout.write(line + "\n")

        if create_legend:
            number_of_items = len(colormap)

            SIDout.write(f"LEGEND_TITLE\t{dataset_name} legend\n")
            SIDout.write("LEGEND_SHAPES")
            for _ in range(number_of_items):
                SIDout.write("\t1")

            SIDout.write("\nLEGEND_COLORS")
            for col in colormap.values():
                SIDout.write(f"\t{rgb_to_hex(col)}")

            SIDout.write("\nLEGEND_LABELS")
            for key in colormap.keys():
                SIDout.write(f"\t{key}")
            SIDout.write("\n")

        SIDout.write("\nDATA\n")
        df_writeout = dfCellColor.to_csv(
            None, sep="\t", header=False, index=False
        )
        SIDout.write(df_writeout)

    return outfp


def create_indel_heatmap(
    alleletable: pd.DataFrame,
    tree: CassiopeiaTree,
    dataset_name: str,
    output_directory: str,
    indel_colors: Optional[pd.DataFrame] = None,
    indel_priors: Optional[pd.DataFrame] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> List[str]:
    """Creates a set of files for displaying an indel heatmap with iTOL

    Creates a set of files, each one corresponding to a character, that can be
    used to display an allele heatmap alongside a tree in iTOL. If neither indel
    colors nor indel priors are provided, a random color mapping is created for
    each unique indel.

    Args:
        alleletable: An AlleleTable for the tree
        tree: CassiopeiaTree
        dataset_name: Name for the dataset
        output_directory: Where to write the output files
        indel_colors: Mapping of indels to colors.
        indel_priors: Prior probabilities for each indel. Only `indel_colors`
            are not provided, in which case a new indel color mapping is created
            by displaying low-probability indels with bright colors and
            high-probability ones with dull colors.
        random_state: Random state for reproducibility

    Returns:
        A set of filepaths to each cut-site's color bar file for iTOL batch
            uploading.
    """

    _leaves = tree.leaves

    lineage_profile = utilities.convert_alleletable_to_lineage_profile(
        alleletable
    )
    clustered_linprof = lineage_profile.loc[_leaves[::-1]]

    if indel_colors is None:
        if indel_priors is None:
            indel_colors = get_random_indel_colors(
                lineage_profile, random_state
            )
        else:
            indel_colors = get_indel_colors(indel_priors, random_state)

    # Convert colors and make colored alleleTable (rgb_heatmap)
    r, g, b = (
        np.zeros(clustered_linprof.shape),
        np.zeros(clustered_linprof.shape),
        np.zeros(clustered_linprof.shape),
    )
    for i in tqdm(range(clustered_linprof.shape[0])):
        for j in range(clustered_linprof.shape[1]):
            ind = str(clustered_linprof.iloc[i, j])
            if ind == "nan":
                r[i, j], g[i, j], b[i, j] = 1, 1, 1
            elif "none" in ind.lower():
                r[i, j], g[i, j], b[i, j] = 192 / 255, 192 / 255, 192 / 255
            else:
                col = hsv_to_rgb(tuple(indel_colors.loc[ind, "color"]))
                r[i, j], g[i, j], b[i, j] = col[0], col[1], col[2]

    rgb_heatmap = np.stack((r, g, b), axis=2)

    allele_files = []
    for j in range(0, rgb_heatmap.shape[1]):
        item_list = []
        for i in rgb_heatmap[:, j]:
            item = (
                "rgb("
                + str(int(round(255 * i[0])))
                + ","
                + str(int(round(255 * i[1])))
                + ","
                + str(int(round(255 * i[2])))
                + ")"
            )
            item_list.append(item)
        dfAlleleColor = pd.DataFrame()
        dfAlleleColor["cellBC"] = clustered_linprof.index.values
        dfAlleleColor["color"] = item_list

        if j == 0:
            header = [
                "DATASET_COLORSTRIP",
                "SEPARATOR TAB",
                "COLOR\t#000000",
                "MARGIN\t100",
                "DATASET_LABEL\tallele" + str(j),
                "STRIP_WIDTH\t50",
                "SHOW_INTERNAL\t0",
                "DATA",
                "",
            ]
        else:
            header = [
                "DATASET_COLORSTRIP",
                "SEPARATOR TAB",
                "COLOR\t#000000",
                "DATASET_LABEL\tallele" + str(j),
                "STRIP_WIDTH\t50",
                "SHOW_INTERNAL\t0",
                "DATA",
                "",
            ]

        if len(str(j)) == 1:
            alleleLabel_fileout = os.path.join(
                output_directory, f"indelColors_0{j}.txt"
            )
        elif len(str(j)) == 2:
            alleleLabel_fileout = os.path.join(
                output_directory, f"indelColors_{j}.txt"
            )
        with open(alleleLabel_fileout, "w") as ALout:
            for line in header:
                ALout.write(line + "\n")
            df_writeout = dfAlleleColor.to_csv(
                None, sep="\t", header=False, index=False
            )
            ALout.write(df_writeout)

        allele_files.append(alleleLabel_fileout)

    return allele_files


def get_random_indel_colors(
    lineage_profile: pd.DataFrame,
    random_state: Optional[np.random.RandomState] = None,
) -> pd.DataFrame:
    """Assigns random color to each unique indel.

    Assigns a random HSV value to each indel observed in the specified
    lineage profile.

    Args:
        lineage_profile: An NxM lineage profile reporting the indels observed
            at each cut site in a cell.
        random_state: A random state for reproducibility

    Returns:
        A mapping from indel to HSV color.
    """

    lineage_profile.fillna("missing", inplace=True)
    unique_indels = np.unique(
        np.hstack(lineage_profile.apply(lambda x: x.unique(), axis=0))
    )

    # color families
    redmag = [0.5, 1, 0, 0.5, 0, 1]
    grnyel = [0, 1, 0.5, 1, 0, 0.5]
    cynblu = [0, 0.5, 0, 1, 0.5, 1]
    colorlist = [redmag, grnyel, cynblu]

    # construct dictionary of indels-to-RGBcolors
    indel2color = {}
    for indel in unique_indels:
        if "none" in indel.lower():
            indel2color[indel] = rgb_to_hsv((0.75, 0.75, 0.75))
        elif indel == "NC":
            indel2color[indel] = rgb_to_hsv((0, 0, 0))
        elif indel == "missing":
            indel2color[indel] = rgb_to_hsv((1, 1, 1))
        else:
            # randomly pick a color family and then draw random colors
            # from that family
            if random_state:
                rgb_i = random_state.choice(range(len(colorlist)))
                color_ranges = colorlist[rgb_i]
                indel2color[indel] = rgb_to_hsv(
                    generate_random_color(
                        color_ranges[:2],
                        color_ranges[2:4],
                        color_ranges[4:6],
                        random_state,
                    )
                )
            else:
                rgb_i = np.random.choice(range(len(colorlist)))
                color_ranges = colorlist[rgb_i]
                indel2color[indel] = rgb_to_hsv(
                    generate_random_color(
                        color_ranges[:2], color_ranges[2:4], color_ranges[4:6]
                    )
                )

    indel_colors = pd.DataFrame(columns=["color"])
    for indel in indel2color.keys():
        indel_colors.loc[indel, "color"] = indel2color[indel]
    return pd.DataFrame(indel_colors["color"])


def get_indel_colors(
    indel_priors: pd.DataFrame,
    random_state: Optional[np.random.RandomState] = None,
) -> pd.DataFrame:
    """Map indel to HSV colors using prior probabilities.

    Given prior probabilities of indels, map each indel to a color reflecting
    its relative likelihood. Specifically, indels that are quite frequent will
    have dull colors and indels that are rare will be bright.

    Args:
        indel_priors: DataFrame mapping indels to probabilities
        random_state: Random state for reproducibility

    Returns:
        DataFrame mapping indel to color
    """

    def assign_color(prob, random_state):
        """Samples a HSV color, with saturation proportional to probability."""
        if random_state:
            H = random_state.rand()
        else:
            H = np.random.rand()
        S = prob
        V = 0.5 + 0.5 * S
        return (H, S, V)

    indel_priors_copy = indel_priors.copy()
    indel_priors_copy["NormFreq"] = indel_priors_copy["freq"]
    indel_priors_copy["NormFreq"] = indel_priors_copy.apply(
        lambda x: (indel_priors_copy["NormFreq"].max() - x.NormFreq), axis=1
    )
    indel_priors_copy["NormFreq"] /= indel_priors_copy["NormFreq"].max()
    indel_priors_copy["color"] = indel_priors_copy.apply(
        lambda x: assign_color(x.NormFreq, random_state), axis=1
    )
    return pd.DataFrame(indel_priors_copy["color"])


def hex_to_rgb(value) -> Tuple[int, int, int]:
    """Converts Hex color code to RGB.

    Args:
        values: hex values (beginning with "#")

    Returns:
        A tuple denoting (r, g, b)
    """
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb) -> str:
    """Converts (r, g, b) tuple to hex

    Args:
        rgb: A tuple denoting (R, G, B) values

    Returns:
        A hex string.
    """

    r, g, b = rgb[0], rgb[1], rgb[2]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def generate_random_color(
    r_range: Tuple[float, float],
    g_range: Tuple[float, float],
    b_range: Tuple[float, float],
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[int, int, int]:
    """Generates a random color from ranges of RGB.

    Args:
        r_range: Range of values for the R value
        g_range: Range of value for the G value
        b_range: Range of values for the B value
        random_state: Random state for reproducibility

    Returns:
        An (R, G, B) tuple sampled from the ranges passed in.
    """

    if random_state:
        red = random_state.uniform(r_range[0], r_range[1])
        grn = random_state.uniform(g_range[0], g_range[1])
        blu = random_state.uniform(b_range[0], b_range[1])
    else:
        red = np.random.uniform(r_range[0], r_range[1])
        grn = np.random.uniform(g_range[0], g_range[1])
        blu = np.random.uniform(b_range[0], b_range[1])
    return (red, grn, blu)
