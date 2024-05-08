"""
Functions for use when plotting trees locally. Unlike itol_utilities.py, which
plots trees using iTOL, a cloud tree plotting service, this file implements
functions that plot trees without requiring a subscription to iTOL. Currently,
trees may be plotted either statically (using Matplotlib) or dynamically
(using Plotly).
"""
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import hsv_to_rgb
from typing_extensions import Literal
from tqdm import tqdm

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import PlottingError, PlottingWarning
from cassiopeia.plotting import utilities
from cassiopeia.preprocess import utilities as preprocess_utilities


def compute_colorstrip_size(
    node_coords: Dict[str, Tuple[float, float]],
    anchor_coords: Dict[str, Tuple[float, float]],
    loc: Literal["left", "right", "up", "down", "polar"],
) -> float:
    """Compute the size of colorstrip boxes.

    This function computes two sizes: the width and height of the colorstrip
    boxes. The height is selected to be the value such that the boxes are
    tightly arranged next to one another. The width is selected to be 5% of the
    tree depth.

    Args:
        node_coords: Node coordinates.
        anchor_coords: Anchor coordinates.
        loc: The location of the box relative to the anchor.

    Returns:
        Two floats representing the width and height of the box.
    """
    min_pos = np.inf
    max_pos = -np.inf
    min_depth = np.inf
    max_depth = -np.inf
    for x, y in node_coords.values():
        pos, depth = x, y
        if loc in ("left", "right"):
            pos, depth = y, x
        min_pos = min(min_pos, pos)
        max_pos = max(max_pos, pos)
        min_depth = min(min_depth, depth)
        max_depth = max(max_depth, depth)
    return (
        (max_depth - min_depth) * 0.05,
        (max_pos - min_pos + 1) / len(anchor_coords),
    )


def create_categorical_colorstrip(
    values: Dict[str, str],
    anchor_coords: Dict[str, Tuple[float, float]],
    width: float,
    height: float,
    spacing: float,
    loc: Literal["left", "right", "up", "down", "polar"],
    cmap: Union[str, mpl.colors.Colormap] = "tab10",
    value_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[
    Dict[str, Tuple[List[float], List[float], Tuple[float, float, float], str]],
    Dict[str, Tuple[float, float]],
]:
    """Create a colorstrip for a categorical variable.

    Args:
        values: Dictionary containing node-category pairs.
        anchor_coords: Anchor coordinates for the colorstrip boxes.
        width: Width of the colorstrip.
        height: Height of each box of the colorstrip.
        spacing: The amount of space to "skip" before placing each box
            (i.e. padding)
        loc: Location of the boxes relative to the anchors.
        cmap: Colormap. Defaults to the `tab10` colormap of Matplotlib.
        value_mapping: An optional dictionary containing string values to their
            integer mappings. These mappings are used to assign colors by
            calling the `cmap` with the designated integer mapping. By default,
            the values are assigned pseudo-randomly (whatever order the set()
            operation returns).

    Returns:
        Dictionary of box coordinates and a dictionary of new anchor coordinates.
    """
    if type(cmap) == str:
        cm = plt.colormaps[cmap]
    elif isinstance(cmap, mpl.colors.Colormap):
        cm = cmap
    else:
        raise PlottingError(
            "Colormap must be a string or a matplotlib colormap."
        )
    unique_values = set(values.values())
    value_mapping = value_mapping or {
        val: i for i, val in enumerate(unique_values)
    }

    boxes, next_anchor_coords = utilities.place_colorstrip(
        anchor_coords, width, height, spacing, loc
    )
    colorstrip = {}
    for leaf, val in values.items():
        v = value_mapping[val] % cm.N
        colorstrip[leaf] = boxes[leaf] + (cm(v)[:-1], f"{leaf}\n{val}")
    return colorstrip, next_anchor_coords


def create_continuous_colorstrip(
    values: Dict[str, float],
    anchor_coords: Dict[str, Tuple[float, float]],
    width: float,
    height: float,
    spacing: float,
    loc: Literal["left", "right", "up", "down", "polar"],
    cmap: Union[str, mpl.colors.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[
    Dict[str, Tuple[List[float], List[float], Tuple[float, float, float], str]],
    Dict[str, Tuple[float, float]],
]:
    """Create a colorstrip for a continuous variable.

    Args:
        values: Dictionary containing node-value pairs.
        anchor_coords: Anchor coordinates for the colorstrip boxes.
        width: Width of the colorstrip.
        height: Height of each box of the colorstrip.
        spacing: The amount of space to "skip" before placing each box
            (i.e. padding)
        loc: Location of the boxes relative to the anchors.
        cmap: Colormap. Defaults to the `viridis` colormap of Matplotlib.
        vmin: Value representing the lower limit of the color scale.
        vmax: Value representing the upper limit of the color scale.

    Returns:
        Dictionary of box coordinates and a dictionary of new anchor coordinates.
    """
    if type(cmap) == str:
        cm = plt.colormaps[cmap]
    elif isinstance(cmap, mpl.colors.Colormap):
        cm = cmap
    else:
        raise PlottingError(
            "Colormap must be a string or a matplotlib colormap."
        )
    max_value = vmax if vmax is not None else max(values.values())
    min_value = vmin if vmin is not None else min(values.values())
    if min_value >= max_value:
        warnings.warn(
            f"Min value is {min_value} and max value is {max_value}.",
            PlottingWarning,
        )

    boxes, next_anchor_coords = utilities.place_colorstrip(
        anchor_coords, width, height, spacing, loc
    )
    colorstrip = {}
    for leaf, val in values.items():
        v = np.clip((val - min_value) / (max_value - min_value), 0, 1)
        colorstrip[leaf] = boxes[leaf] + (cm(v)[:-1], f"{leaf}\n{val}")
    return colorstrip, next_anchor_coords


def create_indel_heatmap(
    allele_table: pd.DataFrame,
    anchor_coords: Dict[str, Tuple[float, float]],
    width: float,
    height: float,
    spacing: float,
    loc: Literal["left", "right", "up", "down", "polar"],
    indel_colors: Optional[pd.DataFrame] = None,
    indel_priors: Optional[pd.DataFrame] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[
    List[
        Dict[
            str,
            Tuple[List[float], List[float], Tuple[float, float, float], str],
        ]
    ],
    Dict[str, Tuple[float, float]],
]:
    """Create an indel heatmap.

    Args:
        allele_table: Allele table containing indels.
        anchor_coords: Anchor coordinates for the colorstrip boxes.
        width: Width of the colorstrip.
        height: Height of each box of the colorstrip.
        spacing: The amount of space to "skip" before placing each box
            (i.e. padding)
        loc: Location of the boxes relative to the anchors.
        indel_colors: Mapping of indels to colors.
        indel_priors: Prior probabilities for each indel. Only if `indel_colors`
            are not provided, in which case a new indel color mapping is created
            by displaying low-probability indels with bright colors and
            high-probability ones with dull colors.
        random_state: Random state for reproducibility

    Returns:
        List of colorstrips (where each colorstrip is a dictionary of
            coordinates) and a dictionary of new anchor coordinates.
    """
    clustered_linprof, _indel_colors = utilities.prepare_alleletable(
        allele_table, list(anchor_coords.keys()), indel_priors, random_state
    )
    if indel_colors is None:
        indel_colors = _indel_colors

    heatmap = []
    for site_idx in tqdm(range(clustered_linprof.shape[1])):
        cut_site = clustered_linprof.columns[site_idx]
        boxes, anchor_coords = utilities.place_colorstrip(
            anchor_coords, width, height, spacing, loc
        )
        colorstrip = {}
        for i in range(clustered_linprof.shape[0]):
            leaf = clustered_linprof.index[i]
            ind = str(clustered_linprof.iloc[i, site_idx])
            if ind == "nan":
                col = (1, 1, 1)
            elif "none" in ind.lower():
                col = (192 / 255, 192 / 255, 192 / 255)
            else:
                c = hsv_to_rgb(tuple(indel_colors.loc[ind, "color"]))
                col = (c[0], c[1], c[2])
            text = f"{leaf}\n{cut_site}\n{ind}"
            colorstrip[leaf] = boxes[leaf] + (col, text)
        heatmap.append(colorstrip)

    return heatmap, anchor_coords


def create_clade_colors(
    tree: CassiopeiaTree, clade_colors: Dict[str, Tuple[float, float, float]]
) -> Tuple[
    Dict[str, Tuple[float, float, float]],
    Dict[Tuple[str, str], Tuple[float, float, float]],
]:
    """Assign colors to nodes and branches by clade.

    Args:
        tree: The CassiopeiaTree.
        clade_colors: Dictionary containing internal node-color mappings. These
            colors will be used to color all the paths from this node to the
            leaves the provided color.

    Returns:
        Two dictionaries. The first contains the node colors, and the second
            contains the branch colors.
    """
    # Deal with clade colors.
    descendants = {}
    for node in clade_colors.keys():
        descendants[node] = set(tree.depth_first_traverse_nodes(node))
    if len(set.union(*list(descendants.values()))) != sum(
        len(d) for d in descendants.values()
    ):
        warnings.warn(
            "Some clades specified with `clade_colors` are overlapping. "
            "Colors may be overridden.",
            PlottingWarning,
        )

    # Color by largest clade first
    node_colors = {}
    branch_colors = {}
    for node in sorted(
        descendants, key=lambda x: len(descendants[x]), reverse=True
    ):
        color = clade_colors[node]
        for n1, n2 in tree.depth_first_traverse_edges(node):
            node_colors[n1] = node_colors[n2] = color
            branch_colors[(n1, n2)] = color
    return node_colors, branch_colors


def place_tree_and_annotations(
    tree: CassiopeiaTree,
    depth_key: Optional[str] = None,
    meta_data: Optional[List[str]] = None,
    allele_table: Optional[pd.DataFrame] = None,
    indel_colors: Optional[pd.DataFrame] = None,
    indel_priors: Optional[pd.DataFrame] = None,
    orient: Union[Literal["up", "down", "left", "right"], float] = 90.0,
    extend_branches: bool = True,
    angled_branches: bool = True,
    add_root: bool = False,
    colorstrip_width: Optional[float] = None,
    colorstrip_spacing: Optional[float] = None,
    clade_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    continuous_cmap: Union[str, mpl.colors.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    categorical_cmap: Union[str, mpl.colors.Colormap] = "tab10",
    value_mapping: Optional[Dict[str, int]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[Dict, Dict, Dict, Dict, List]:
    """Helper function to place the tree and all requested annotations.

    Args:
        tree: The CassiopeiaTree to plot.
        depth_key: The node attribute to use as the depth of the nodes. If
            not provided, the distances from the root is used by calling
            `tree.get_distances`.
        meta_data: Meta data to plot alongside the tree, which must be columns
            in the CassiopeiaTree.cell_meta variable.
        allele_table: Alleletable to plot alongside the tree.
        indel_colors: Color mapping to use for plotting the alleles for each
            cell. Only necessary if `allele_table` is specified.
        indel_priors: Prior probabilities for each indel. Only useful if an
            allele table is to be plotted and `indel_colors` is None.
        orient: The orientation of the tree. Valid arguments are `left`, `right`,
            `up`, `down` to display a rectangular plot (indicating the direction
            of going from root -> leaves) or any number, in which case the
            tree is placed in polar coordinates with the provided number used
            as an angle offset. Defaults to 90.
        extend_branches: Extend branch lengths such that the distance from the
            root to every node is the same. If `depth_key` is also provided, then
            only the leaf branches are extended to the deepest leaf.
        angled_branches: Display branches as angled, instead of as just a
            line from the parent to a child.
        add_root: Add a root node so that only one branch connects to the
            start of the tree. This node will have the name `synthetic_root`.
        colorstrip_width: Width of the colorstrip. Width is defined as the
            length in the direction of the leaves. Defaults to 5% of the tree
            depth.
        colorstrip_spacing: Space between consecutive colorstrips. Defaults to
            half of `colorstrip_width`.
        clade_colors: Dictionary containing internal node-color mappings. These
            colors will be used to color all the paths from this node to the
            leaves the provided color.
        continuous_cmap: Colormap to use for continuous variables. Defaults to
            `viridis`.
        vmin: Value representing the lower limit of the color scale. Only applied
            to continuous variables.
        vmax: Value representing the upper limit of the color scale. Only applied
            to continuous variables.
        categorical_cmap: Colormap to use for categorical variables. Defaults to
            `tab10`.
        value_mapping: An optional dictionary containing string values to their
            integer mappings. These mappings are used to assign colors by
            calling the `cmap` with the designated integer mapping. By default,
            the values are assigned pseudo-randomly (whatever order the set()
            operation returns). Only applied for categorical variables.
        random_state: A random state for reproducibility

    Returns:
        Four dictionaries (node coordinates, branch coordinates, node
            colors, branch colors) and a list of colorstrips.
    """
    meta_data = meta_data or []

    # Place tree on the appropriate coordinate system.
    node_coords, branch_coords = utilities.place_tree(
        tree,
        depth_key=depth_key,
        orient=orient,
        extend_branches=extend_branches,
        angled_branches=angled_branches,
        add_root=add_root,
    )

    # Compute first set of anchor coords, which are just the coordinates of
    # all the leaves.
    anchor_coords = {
        node: coords
        for node, coords in node_coords.items()
        if tree.is_leaf(node)
    }
    is_polar = isinstance(orient, (float, int))
    loc = "polar" if is_polar else orient
    tight_width, tight_height = compute_colorstrip_size(
        node_coords, anchor_coords, loc
    )
    width = colorstrip_width or tight_width
    spacing = colorstrip_spacing or tight_width / 2

    # Place indel heatmap
    colorstrips = []
    if allele_table is not None:
        heatmap, anchor_coords = create_indel_heatmap(
            allele_table,
            anchor_coords,
            width,
            tight_height,
            spacing,
            loc,
            indel_colors,
            indel_priors,
            random_state,
        )
        colorstrips.extend(heatmap)

    # Any other annotations
    if type(meta_data) == str:
        meta_data = [meta_data]
        
    for meta_item in meta_data:
        if meta_item not in tree.cell_meta.columns:
            raise PlottingError(
                "Meta data item not in CassiopeiaTree cell meta."
            )

        values = tree.cell_meta[meta_item]
        if pd.api.types.is_numeric_dtype(values):
            colorstrip, anchor_coords = create_continuous_colorstrip(
                values.to_dict(),
                anchor_coords,
                width,
                tight_height,
                spacing,
                loc,
                continuous_cmap,
                vmin,
                vmax,
            )
        elif pd.api.types.is_string_dtype(
            values
        ) or pd.api.types.is_categorical_dtype(values):
            colorstrip, anchor_coords = create_categorical_colorstrip(
                values.to_dict(),
                anchor_coords,
                width,
                tight_height,
                spacing,
                loc,
                categorical_cmap,
                value_mapping,
            )
        else:
            raise PlottingError(
                f"Column {meta_item} has unrecognized dtype {pd.api.types.infer_dtype(values)}. "
                "Only numeric, string, and categorical dtypes are supported."
            )
        colorstrips.append(colorstrip)

    # Clade colors
    node_colors = {}
    branch_colors = {}
    if clade_colors:
        node_colors, branch_colors = create_clade_colors(tree, clade_colors)
    return node_coords, branch_coords, node_colors, branch_colors, colorstrips


def plot_matplotlib(
    tree: CassiopeiaTree,
    depth_key: Optional[str] = None,
    meta_data: Optional[List[str]] = None,
    allele_table: Optional[pd.DataFrame] = None,
    indel_colors: Optional[pd.DataFrame] = None,
    indel_priors: Optional[pd.DataFrame] = None,
    orient: Union[Literal["up", "down", "left", "right"], float] = 90.0,
    extend_branches: bool = True,
    angled_branches: bool = True,
    add_root: bool = False,
    figsize: Tuple[float, float] = (7.0, 7.0),
    colorstrip_width: Optional[float] = None,
    colorstrip_spacing: Optional[float] = None,
    clade_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    internal_node_kwargs: Optional[Dict] = None,
    leaf_kwargs: Optional[Dict] = None,
    branch_kwargs: Optional[Dict] = None,
    colorstrip_kwargs: Optional[Dict] = None,
    continuous_cmap: Union[str, mpl.colors.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    categorical_cmap: Union[str, mpl.colors.Colormap] = "tab10",
    value_mapping: Optional[Dict[str, int]] = None,
    ax: Optional[plt.Axes] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate a static plot of a tree using Matplotlib.

    Args:
        tree: The CassiopeiaTree to plot.
        depth_key: The node attribute to use as the depth of the nodes. If
            not provided, the distances from the root is used by calling
            `tree.get_distances`.
        meta_data: Meta data to plot alongside the tree, which must be columns
            in the CassiopeiaTree.cell_meta variable.
        allele_table: Allele table to plot alongside the tree.
        indel_colors: Color mapping to use for plotting the alleles for each
            cell. Only necessary if `allele_table` is specified.
        indel_priors: Prior probabilities for each indel. Only useful if an
            allele table is to be plotted and `indel_colors` is None.
        orient: The orientation of the tree. Valid arguments are `left`, `right`,
            `up`, `down` to display a rectangular plot (indicating the direction
            of going from root -> leaves) or any number, in which case the
            tree is placed in polar coordinates with the provided number used
            as an angle offset. Defaults to 90.
        extend_branches: Extend branch lengths such that the distance from the
            root to every node is the same. If `depth_key` is also provided, then
            only the leaf branches are extended to the deepest leaf.
        angled_branches: Display branches as angled, instead of as just a
            line from the parent to a child.
        add_root: Add a root node so that only one branch connects to the
            start of the tree. This node will have the name `synthetic_root`.
        figsize: Size of the plot. Defaults to (7., 7.,)
        colorstrip_width: Width of the colorstrip. Width is defined as the
            length in the direction of the leaves. Defaults to 5% of the tree
            depth.
        colorstrip_spacing: Space between consecutive colorstrips. Defaults to
            half of `colorstrip_width`.
        clade_colors: Dictionary containing internal node-color mappings. These
            colors will be used to color all the paths from this node to the
            leaves the provided color.
        internal_node_kwargs: Keyword arguments to pass to `plt.scatter` when
            plotting internal nodes.
        leaf_kwargs: Keyword arguments to pass to `plt.scatter` when
            plotting leaf nodes.
        branch_kwargs: Keyword arguments to pass to `plt.plot` when plotting
            branches.
        colorstrip_kwargs: Keyword arguments to pass to `plt.fill` when plotting
            colorstrips.
        continuous_cmap: Colormap to use for continuous variables. Defaults to
            `viridis`.
        vmin: Value representing the lower limit of the color scale. Only applied
            to continuous variables.
        vmax: Value representing the upper limit of the color scale. Only applied
            to continuous variables.
        categorical_cmap: Colormap to use for categorical variables. Defaults to
            `tab10`.
        value_mapping: An optional dictionary containing string values to their
            integer mappings. These mappings are used to assign colors by
            calling the `cmap` with the designated integer mapping. By default,
            the values are assigned pseudo-randomly (whatever order the set()
            operation returns). Only applied for categorical variables.
        ax: Matplotlib axis to place the tree. If not provided, a new figure is
            initialized.
        random_state: A random state for reproducibility

    Returns:
        If `ax` is provided, `ax` is returned. Otherwise, a tuple of (fig, ax)
            of the newly initialized figure and axis.
    """
    is_polar = isinstance(orient, (float, int))
    (
        node_coords,
        branch_coords,
        node_colors,
        branch_colors,
        colorstrips,
    ) = place_tree_and_annotations(
        tree,
        depth_key,
        meta_data,
        allele_table,
        indel_colors,
        indel_priors,
        orient,
        extend_branches,
        angled_branches,
        add_root,
        colorstrip_width,
        colorstrip_spacing,
        clade_colors,
        continuous_cmap,
        vmin,
        vmax,
        categorical_cmap,
        value_mapping,
        random_state,
    )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.set_axis_off()

    # Plot all nodes
    _leaf_kwargs = dict(x=[], y=[], s=5, c="black")
    _node_kwargs = dict(x=[], y=[], s=0, c="black")
    _leaf_kwargs.update(leaf_kwargs or {})
    _node_kwargs.update(internal_node_kwargs or {})
    for node, (x, y) in node_coords.items():
        if node in node_colors:
            continue
        if is_polar:
            x, y = utilities.polar_to_cartesian(x, y)
        if tree.is_leaf(node):
            _leaf_kwargs["x"].append(x)
            _leaf_kwargs["y"].append(y)
        else:
            _node_kwargs["x"].append(x)
            _node_kwargs["y"].append(y)
    ax.scatter(**_leaf_kwargs)
    ax.scatter(**_node_kwargs)

    _leaf_colors = []
    _node_colors = []
    _leaf_kwargs.update({"x": [], "y": []})
    _node_kwargs.update({"x": [], "y": []})
    for node, color in node_colors.items():
        x, y = node_coords[node]
        if is_polar:
            x, y = utilities.polar_to_cartesian(x, y)
        if tree.is_leaf(node):
            _leaf_kwargs["x"].append(x)
            _leaf_kwargs["y"].append(y)
            _leaf_colors.append(color)
        else:
            _node_kwargs["x"].append(x)
            _node_kwargs["y"].append(y)
            _node_colors.append(color)

    _leaf_kwargs["c"] = _leaf_colors
    _node_kwargs["c"] = _node_colors
    ax.scatter(**_leaf_kwargs)
    ax.scatter(**_node_kwargs)

    # Plot all branches
    _branch_kwargs = dict(linewidth=1, c="black")
    _branch_kwargs.update(branch_kwargs or {})
    for branch, (xs, ys) in branch_coords.items():
        if branch in branch_colors:
            continue
        if is_polar:
            xs, ys = utilities.polars_to_cartesians(xs, ys)
        ax.plot(xs, ys, **_branch_kwargs)

    for branch, color in branch_colors.items():
        _branch_kwargs["c"] = color
        xs, ys = branch_coords[branch]
        if is_polar:
            xs, ys = utilities.polars_to_cartesians(xs, ys)
        ax.plot(xs, ys, **_branch_kwargs)

    # Colorstrips
    _colorstrip_kwargs = dict(linewidth=0)
    _colorstrip_kwargs.update(colorstrip_kwargs or {})
    for colorstrip in colorstrips:
        # Last element is text, but this can not be shown in static plotting.
        for xs, ys, c, _ in colorstrip.values():
            _colorstrip_kwargs["c"] = c
            if is_polar:
                xs, ys = utilities.polars_to_cartesians(xs, ys)
            ax.fill(xs, ys, **_colorstrip_kwargs)

    return (fig, ax) if fig is not None else ax


def plot_plotly(
    tree: CassiopeiaTree,
    depth_key: Optional[str] = None,
    meta_data: Optional[List[str]] = None,
    allele_table: Optional[pd.DataFrame] = None,
    indel_colors: Optional[pd.DataFrame] = None,
    indel_priors: Optional[pd.DataFrame] = None,
    orient: Union[Literal["up", "down", "left", "right"], float] = 90.0,
    extend_branches: bool = True,
    angled_branches: bool = True,
    add_root: bool = False,
    width: float = 500.0,
    height: float = 500.0,
    colorstrip_width: Optional[float] = None,
    colorstrip_spacing: Optional[float] = None,
    clade_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    internal_node_kwargs: Optional[Dict] = None,
    leaf_kwargs: Optional[Dict] = None,
    branch_kwargs: Optional[Dict] = None,
    colorstrip_kwargs: Optional[Dict] = None,
    continuous_cmap: Union[str, mpl.colors.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    categorical_cmap: Union[str, mpl.colors.Colormap] = "tab10",
    value_mapping: Optional[Dict[str, int]] = None,
    figure: Optional[go.Figure] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> go.Figure:
    """Generate a static plot of a tree using Plotly.

    Args:
        tree: The CassiopeiaTree to plot.
        depth_key: The node attribute to use as the depth of the nodes. If
            not provided, the distances from the root is used by calling
            `tree.get_distances`.
        meta_data: Meta data to plot alongside the tree, which must be columns
            in the CassiopeiaTree.cell_meta variable.
        allele_table: Allele table to plot alongside the tree.
        indel_colors: Color mapping to use for plotting the alleles for each
            cell. Only necessary if `allele_table` is specified.
        indel_priors: Prior probabilities for each indel. Only useful if an
            allele table is to be plotted and `indel_colors` is None.
        orient: The orientation of the tree. Valid arguments are `left`, `right`,
            `up`, `down` to display a rectangular plot (indicating the direction
            of going from root -> leaves) or any number, in which case the
            tree is placed in polar coordinates with the provided number used
            as an angle offset. Defaults to 90.
        extend_branches: Extend branch lengths such that the distance from the
            root to every node is the same. If `depth_key` is also provided, then
            only the leaf branches are extended to the deepest leaf.
        angled_branches: Display branches as angled, instead of as just a
            line from the parent to a child.
        add_root: Add a root node so that only one branch connects to the
            start of the tree. This node will have the name `synthetic_root`.
        width: Width of the figure.
        height: Height of the figure.
        colorstrip_width: Width of the colorstrip. Width is defined as the
            length in the direction of the leaves. Defaults to 5% of the tree
            depth.
        colorstrip_spacing: Space between consecutive colorstrips. Defaults to
            half of `colorstrip_width`.
        clade_colors: Dictionary containing internal node-color mappings. These
            colors will be used to color all the paths from this node to the
            leaves the provided color.
        internal_node_kwargs: Keyword arguments to pass to `plt.scatter` when
            plotting internal nodes.
        leaf_kwargs: Keyword arguments to pass to `plt.scatter` when
            plotting leaf nodes.
        branch_kwargs: Keyword arguments to pass to `plt.plot` when plotting
            branches.
        colorstrip_kwargs: Keyword arguments to pass to `plt.fill` when plotting
            colorstrips.
        continuous_cmap: Colormap to use for continuous variables. Defaults to
            `viridis`.
        vmin: Value representing the lower limit of the color scale. Only applied
            to continuous variables.
        vmax: Value representing the upper limit of the color scale. Only applied
            to continuous variables.
        categorical_cmap: Colormap to use for categorical variables. Defaults to
            `tab10`.
        value_mapping: An optional dictionary containing string values to their
            integer mappings. These mappings are used to assign colors by
            calling the `cmap` with the designated integer mapping. By default,
            the values are assigned pseudo-randomly (whatever order the set()
            operation returns). Only applied for categorical variables.
        figure: Plotly figure to plot the tree.
        random_state: A random state for reproducibility

    Returns:
        The Plotly figure.
    """
    # Warn user if there are many leaves
    if len(tree.leaves) > 2000:
        warnings.warn(
            "Tree has greater than 2000 leaves. This may take a while.",
            PlottingWarning,
        )

    is_polar = isinstance(orient, (float, int))
    (
        node_coords,
        branch_coords,
        node_colors,
        branch_colors,
        colorstrips,
    ) = place_tree_and_annotations(
        tree,
        depth_key,
        meta_data,
        allele_table,
        indel_colors,
        indel_priors,
        orient,
        extend_branches,
        angled_branches,
        add_root,
        colorstrip_width,
        colorstrip_spacing,
        clade_colors,
        continuous_cmap,
        vmin,
        vmax,
        categorical_cmap,
        value_mapping,
        random_state,
    )
    figure = figure if figure is not None else go.Figure()

    # Plot all nodes
    _leaf_kwargs = dict(
        x=[],
        y=[],
        text=[],
        marker_size=3,
        marker_color="black",
        mode="markers",
        showlegend=False,
        hoverinfo="text",
    )
    # NOTE: setting marker_size=0 has no effect for some reason?
    _node_kwargs = dict(
        x=[],
        y=[],
        text=[],
        marker_size=0.1,
        marker_color="black",
        mode="markers",
        showlegend=False,
        hoverinfo="text",
    )
    _leaf_kwargs.update(leaf_kwargs or {})
    _node_kwargs.update(internal_node_kwargs or {})
    for node, (x, y) in node_coords.items():
        if node in node_colors:
            continue
        text = f"<b>NODE</b><br>{node}"
        if is_polar:
            x, y = utilities.polar_to_cartesian(x, y)
        if tree.is_leaf(node):
            _leaf_kwargs["x"].append(x)
            _leaf_kwargs["y"].append(y)
            _leaf_kwargs["text"].append(text)
        else:
            _node_kwargs["x"].append(x)
            _node_kwargs["y"].append(y)
            _node_kwargs["text"].append(text)
    figure.add_trace(go.Scatter(**_leaf_kwargs))
    figure.add_trace(go.Scatter(**_node_kwargs))

    _leaf_colors = []
    _node_colors = []
    _leaf_kwargs.update({"x": [], "y": [], "text": []})
    _node_kwargs.update({"x": [], "y": [], "text": []})
    for node, color in node_colors.items():
        x, y = node_coords[node]
        text = f"<b>NODE</b><br>{node}"
        if is_polar:
            x, y = utilities.polar_to_cartesian(x, y)
        if tree.is_leaf(node):
            _leaf_kwargs["x"].append(x)
            _leaf_kwargs["y"].append(y)
            _leaf_kwargs["text"].append(text)
            _leaf_colors.append(color)
        else:
            _node_kwargs["x"].append(x)
            _node_kwargs["y"].append(y)
            _node_kwargs["text"].append(text)
            _node_colors.append(color)

    _leaf_kwargs["marker_color"] = _leaf_colors
    _node_kwargs["marker_color"] = _node_colors
    figure.add_trace(go.Scatter(**_leaf_kwargs))
    figure.add_trace(go.Scatter(**_node_kwargs))

    # Plot all branches
    _branch_kwargs = dict(
        x=[],
        y=[],
        text=[],
        line_color="black",
        line_width=1,
        mode="lines",
        showlegend=False,
        hoverinfo="text",
    )
    _branch_kwargs.update(branch_kwargs or {})
    for branch, (xs, ys) in branch_coords.items():
        if branch in branch_colors:
            continue
        _branch_kwargs["x"], _branch_kwargs["y"] = xs, ys
        text = f"<b>BRANCH</b><br>{branch[0]}<br>{branch[1]}"
        if is_polar:
            (
                _branch_kwargs["x"],
                _branch_kwargs["y"],
            ) = utilities.polars_to_cartesians(xs, ys)
        _branch_kwargs["text"] = [text] * len(xs)
        figure.add_trace(go.Scatter(**_branch_kwargs))

    for branch, color in branch_colors.items():
        xs, ys = branch_coords[branch]
        _branch_kwargs["x"], _branch_kwargs["y"] = xs, ys
        _branch_kwargs["line_color"] = color
        text = f"<b>BRANCH</b><br>{branch[0]}<br>{branch[1]}"
        if is_polar:
            (
                _branch_kwargs["x"],
                _branch_kwargs["y"],
            ) = utilities.polars_to_cartesians(xs, ys)
        _branch_kwargs["text"] = [text] * len(xs)
        figure.add_trace(go.Scatter(**_branch_kwargs))

    # Colorstrips
    _colorstrip_kwargs = dict(
        x=[],
        y=[],
        text=[],
        line_width=0,
        fill="toself",
        mode="lines",
        showlegend=False,
        hoverinfo="text",
        hoveron="fills",
    )
    _colorstrip_kwargs.update(colorstrip_kwargs or {})
    for colorstrip in colorstrips:
        for xs, ys, c, text in colorstrip.values():
            _colorstrip_kwargs["x"], _colorstrip_kwargs["y"] = xs, ys
            _colorstrip_kwargs["fillcolor"] = mpl.colors.to_hex(c)
            if is_polar:
                (
                    _colorstrip_kwargs["x"],
                    _colorstrip_kwargs["y"],
                ) = utilities.polars_to_cartesians(xs, ys)
            _colorstrip_kwargs["text"] = text.replace("\n", "<br>")
            figure.add_trace(go.Scatter(**_colorstrip_kwargs))

    figure.update_layout(
        width=width,
        height=height,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return figure
