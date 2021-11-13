"""
Utility functions for use when plotting trees. Functions here support "local"
plotting, as opposed to itol_utilities, which are for visualization through
iTOL.
"""
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from typing_extensions import Literal

from cassiopeia.data import CassiopeiaTree


def degrees_to_radians(degrees) -> float:
    """Convert degrees to radians.

    Args:
        degrees: Degrees

    Returns:
        Degrees converted to radians.
    """
    return degrees * np.pi / 180


def polar_to_cartesian(theta: float, r: float) -> Tuple[float, float]:
    """Convert polar coordinates to cartesian coordinates.

    Args:
        theta: The angle (in radians)
        r: The radius

    Returns:
        A tuple of x and y coordinates
    """
    return r * np.cos(theta), r * np.sin(theta)


def place_tree(
    tree: CassiopeiaTree,
    depth_key: Optional[str] = None,
    orient: Union[Literal["down", "up", "left", "right"], float] = "down",
    depth_scale: float = 1.0,
    width_scale: float = 1.0,
    extend_branches: bool = True,
    angled_branches: bool = True,
    polar_interpolation_threshold: float = 30.0,
    polar_interpolation_step: float = 5.0,
) -> Tuple[
    Dict[str, Tuple[float, float]],
    Dict[Tuple[str, str], Tuple[List[float], List[float]]],
]:
    """Given a tree, computes the coordinates of the nodes and branches.

    This function computes the x and y coordinates of all nodes and branches (as
    lines) to be used for visualization. Several options are provided to
    modify how the elements are placed. This function returns two dictionaries,
    where the first has nodes as keys and an (x, y) tuple as the values.
    Similarly, the second dictionary has (parent, child) tuples as keys denoting
    branches in the tree and a tuple (xs, ys) of two lists containing the x and
    y coordinates to draw the branch.

    This function also provides functionality to place the tree in polar
    coordinates as a circular plot when the `orient` is a number. In this case,
    the returned dictionaries have (thetas, radii) as its elements, which are
    the angles and radii in polar coordinates respectively.

    Note:
        This function only *places* (i.e. computes the x and y coordinates) a
        tree on a coordinate system and does no real plotting.

    Args:
        tree: The CassiopeiaTree to place on the coordinate grid.
        depth_key: The node attribute to use as the depth of the nodes. If
            not provided, the distances from the root is used by calling
            `tree.get_distances`.
        orient: The orientation of the tree. Valid arguments are `left`, `right`,
            `up`, `down` to display a rectangular plot (indicating the direction
            of going from root -> leaves) or any number, in which case the
            tree is placed in polar coordinates with the provided number used
            as an angle offset.
        depth_scale: Scale the depth of the tree by this amount. This option
            has no effect in polar coordinates.
        width_scale: Scale the width of the tree by this amount. This option
            has no effect in polar coordinates.
        extend_branches: Extend branch lengths such that the distance from the
            root to every node is the same. If `depth_key` is also provided, then
            only the leaf branches are extended to the deepest leaf.
        angled_branches: Display branches as angled, instead of as just a
            line from the parent to a child.
        polar_interpolation_threshold: When displaying in polar coordinates,
            many plotting frameworks (such as Plotly) just draws a straight line
            from one point to another, instead of scaling the radius appropriately.
            This effect is most noticeable when two points are connected that
            have a large angle between them. When the angle between two connected
            points in polar coordinates exceed this amount (in degrees), this function
            adds additional points that smoothly interpolate between the two
            endpoints.
        polar_interpolation_step: Interpolation step. See above.

    Returns:
        Two dictionaries, where the first contains the node coordinates and
        the second contains the branch coordinates.
    """
    root = tree.root
    depths = None
    if depth_key:
        depths = {
            node: tree.get_attribute(node, depth_key) for node in tree.nodes
        }
    else:
        depths = tree.get_distances(root)

    placement_depths = {}
    positions = {}
    leaf_i = 0
    leaves = set()
    for node in tree.depth_first_traverse_nodes(postorder=False):
        if tree.is_leaf(node):
            positions[node] = leaf_i
            leaf_i += 1
            leaves.add(node)
            placement_depths[node] = depths[node]
            if extend_branches:
                placement_depths[node] = (
                    max(depths.values()) if depth_key else 0
                )

    # Place nodes by depth
    for node in sorted(depths, key=lambda k: depths[k], reverse=True):
        # Leaves have already been placed
        if node in leaves:
            continue

        # Find all immediate children and place at center.
        min_position = np.inf
        max_position = -np.inf
        min_depth = np.inf
        for child in tree.children(node):
            min_position = min(min_position, positions[child])
            max_position = max(max_position, positions[child])
            min_depth = min(min_depth, placement_depths[child]) - 1
        positions[node] = (min_position + max_position) / 2
        placement_depths[node] = (
            min_depth if extend_branches and not depth_key else depths[node]
        )

    polar = isinstance(orient, (float, int))
    polar_depth_offset = -min(placement_depths.values())
    polar_angle_scale = 360 / (len(leaves) + 1)

    # Define some helper functions to modify coordinate system.
    def reorient(pos, depth):
        pos *= width_scale
        depth *= depth_scale
        if orient == "down":
            return (pos, -depth)
        elif orient == "right":
            return (depth, pos)
        elif orient == "left":
            return (-depth, pos)
        # default: up
        return (pos, depth)

    def polarize(pos, depth):
        # angle, radius
        return (
            (pos + 1) * polar_angle_scale + orient,
            depth + polar_depth_offset,
        )

    node_coords = {}
    for node in tree.nodes:
        pos = positions[node]
        depth = placement_depths[node]
        coords = polarize(pos, depth) if polar else reorient(pos, depth)
        node_coords[node] = coords

    branch_coords = {}
    for parent, child in tree.edges:
        parent_pos, parent_depth = positions[parent], placement_depths[parent]
        child_pos, child_depth = positions[child], placement_depths[child]

        middle_x = (
            child_pos if angled_branches else (parent_pos + child_pos) / 2
        )
        middle_y = (
            parent_depth
            if angled_branches
            else (parent_depth + child_depth) / 2
        )
        _xs = [parent_pos, middle_x, child_pos]
        _ys = [parent_depth, middle_y, child_depth]

        xs = []
        ys = []
        for _x, _y in zip(_xs, _ys):
            if polar:
                x, y = polarize(_x, _y)

                if xs:
                    # Interpolate to prevent weird angles if the angle exceeds threshold.
                    prev_x = xs[-1]
                    prev_y = ys[-1]
                    if abs(x - prev_x) > polar_interpolation_threshold:
                        num = int(abs(x - prev_x) / polar_interpolation_step)
                        for inter_x, inter_y in zip(
                            np.linspace(prev_x, x, num)[1:-1],
                            np.linspace(prev_y, y, num)[1:-1],
                        ):
                            xs.append(inter_x)
                            ys.append(inter_y)
            else:
                x, y = reorient(_x, _y)
            xs.append(x)
            ys.append(y)

        branch_coords[(parent, child)] = (xs, ys)

    return node_coords, branch_coords


def place_colorstrip(
    anchor_coords: Dict[str, Tuple[float, float]],
    width: float,
    height: float,
    loc: Literal["left", "right", "up", "down", "polar"],
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Compute the coordinates of the boxes that represent a colorstrip.

    This function computes the x and y coordinates (or the angles and radii)
    of colored boxes, which together form a colorstrip used to annotate leaves
    of a tree.

    Args:
        anchor_coords: Dictionary of nodes-to-coordinate tuples that contain
            the "anchor" point to start the colorstrip. When `loc=left`, this
            is the center right of each box, when `loc=right`, this is the
            center left of each box, etc.
        width: Width of the box. The width is defined as the length of the
            box in the same direction as the leaves.
        height: Height of the box. The height is defined as the length of the
            box in the direction perpendicular to the leaves.
        loc: Where to place each box relative to the anchors. Valid options are:
            `left`, `right`, `up`, `down`, `polar`.

    Returns:
        A dictionary of node-to-coordinate tuples for each box.
    """
    size_x, size_y = (
        (width, height) if loc in ("left", "right") else (height, width)
    )
    coef_x, coef_y = 0, 1  # default: up / polar
    if loc == "left":
        coef_x, coef_y = -1, 0
    elif loc == "right":
        coef_x, coef_y = 1, 0
    elif loc == "down":
        coef_x, coef_y = 0, -1

    boxes = {}
    for anchor, (x, y) in anchor_coords.items():
        center_x, center_y = x + coef_x * size_x / 2, y + coef_y * size_y / 2
        xs = [
            center_x + size_x / 2,
            center_x - size_x / 2,
            center_x - size_x / 2,
            center_x + size_x / 2,
            center_x + size_x / 2,
        ]
        ys = [
            center_y + size_y / 2,
            center_y + size_y / 2,
            center_y - size_y / 2,
            center_y - size_y / 2,
            center_y + size_y / 2,
        ]
        boxes[anchor] = (xs, ys)
    return boxes


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
