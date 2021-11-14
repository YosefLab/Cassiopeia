import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    Returns:
        Dictionary of box coordinates and a dictionary of new anchor coordinates.
    """
    cm = plt.cm.get_cmap(cmap)
    unique_values = set(values.values())
    value_mapping = {val: i for i, val in enumerate(unique_values)}

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

    Returns:
        Dictionary of box coordinates and a dictionary of new anchor coordinates.
    """
    cm = plt.cm.get_cmap(cmap)
    max_value = max(values.values())
    min_value = min(values.values())

    boxes, next_anchor_coords = utilities.place_colorstrip(
        anchor_coords, width, height, spacing, loc
    )
    colorstrip = {}
    for leaf, val in values.items():
        v = (val - min_value) / (max_value - min_value)
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
        indel_priors: Prior probabilities for each indel. Only `indel_colors`
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
    categorical_cmap: Union[str, mpl.colors.Colormap] = "tab10",
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
        categorical_cmap: Colormap to use for categorical variables. Defaults to
            `tab10`.
        ax: Matplotlib axis to place the tree. If not provided, a new figure is
            initialized.
        random_state: A random state for reproducibility

    Returns:
        If `ax` is provided, `ax` is returned. Otherwise, a tuple of (fig, ax)
        of the newly initialized figure and axis.
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
            )

        if pd.api.types.is_string_dtype(values):
            colorstrip, anchor_coords = create_categorical_colorstrip(
                values.to_dict(),
                anchor_coords,
                width,
                tight_height,
                spacing,
                loc,
                categorical_cmap,
            )
        colorstrips.append(colorstrip)

    # Clade colors
    node_colors = {}
    branch_colors = {}
    if clade_colors:
        node_colors, branch_colors = create_clade_colors(tree, clade_colors)

    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.set_axis_off()

    # Plot all nodes
    _leaf_kwargs = dict(s=5, c="black")
    _node_kwargs = dict(s=0, c="black")
    _leaf_kwargs.update(leaf_kwargs or {})
    _node_kwargs.update(internal_node_kwargs or {})
    leaves = ([], [])
    nodes = ([], [])
    for node, (x, y) in node_coords.items():
        if node in node_colors:
            continue
        if is_polar:
            x, y = utilities.polar_to_cartesian(x, y)
        if tree.is_leaf(node):
            leaves[0].append(x)
            leaves[1].append(y)
        else:
            nodes[0].append(x)
            nodes[1].append(y)
    ax.scatter(*leaves, **_leaf_kwargs)
    ax.scatter(*nodes, **_node_kwargs)

    _leaf_colors = []
    _node_colors = []
    leaves = ([], [])
    nodes = ([], [])
    for node, color in node_colors.items():
        x, y = node_coords[node]
        if is_polar:
            x, y = utilities.polar_to_cartesian(x, y)
        if tree.is_leaf(node):
            leaves[0].append(x)
            leaves[1].append(y)
            _leaf_colors.append(color)
        else:
            nodes[0].append(x)
            nodes[1].append(y)
            _node_colors.append(color)

    _leaf_kwargs["c"] = _leaf_colors
    _node_kwargs["c"] = _node_colors
    ax.scatter(*leaves, **_leaf_kwargs)
    ax.scatter(*nodes, **_node_kwargs)

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
            if is_polar:
                xs, ys = utilities.polars_to_cartesians(xs, ys)

            ax.fill(xs, ys, color=c)

    return (fig, ax) if fig is not None else ax


def plot_interactive():
    pass
