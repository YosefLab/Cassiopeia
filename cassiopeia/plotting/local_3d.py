import warnings
from collections import deque
from functools import partial
from hashlib import sha256
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgb, to_rgba, to_rgba_array

from ..data import CassiopeiaTree
from ..mixins import PlottingError, PlottingWarning, try_import

# Optional dependencies that are required for 3D plotting
cv2 = try_import('cv2')
pv = try_import('pyvista')
measure = try_import('skimage.measure')
neighbors = try_import('sklearn.neighbors')

# Don't want to have to set scanpy as a dependency just to use its color palette.
# https://github.com/scverse/scanpy/blob/master/scanpy/plotting/palettes.py
godsnot_102 = [
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]

def interpolate_branch(parent: Tuple[float, float, float], child: Tuple[float, float, float]) -> np.ndarray:
    """Helper function to interpolate the branch between a parent and child.

    The branch is interpolated in such a way that there is a 90 degree angle.

    Args:
        parent: Parent coordinates as a triplet
        child: Child coordinates as a triplet

    Returns:
        Numpy array containing x, y, z coordinates of the branch.
    """
    x1, y1, z1 = parent
    x2, y2, z2 = child
    return np.array([
        (x1, y1, z1),
        (x2, y2, z1),
        (x2, y2, z2),
    ])

def polyline_from_points(points: np.ndarray) -> "pv.PolyData":
    """Helper function to create a Pyvista object connected a set of points.

    Args:
        points: Points to create the Pyvista object from

    Returns:
        A Pyvista.PolyData object
    """
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def average_mixing(*c):
    """Helper function to mix a set of colors by taking the average of each channel.
    """
    return tuple(to_rgba_array(c)[:,:3].mean(axis=0))

def highlight(c) -> Tuple[float, float, float]:
    """Helper function to highlight a certain color.
    """
    hsv = rgb_to_hsv(c)
    # hsv[1] = min(hsv[1] + 0.2, 1)
    hsv[2] = min(hsv[2] + 0.5, 1)
    return hsv_to_rgb(hsv)

def lowlight(c) -> Tuple[float, float, float]:
    """Helper function to dim out a certain color.
    """
    hsv = rgb_to_hsv(c)
    # hsv[1] = max(hsv[1] - 0.2, 0)
    hsv[2] = max(hsv[2] - 0.5, 0)
    return hsv_to_rgb(hsv)

def labels_from_coordinates(
    tree: CassiopeiaTree,
    attribute_key: str = "spatial",
    angle_distribution: Callable[[], float] = lambda: 0,
    long_axis_distribution: Callable[[], float] = lambda: 1,
    short_axis_distribution: Optional[Callable[[], float]] = None,
    absolute: bool = False
) -> np.ndarray:
    """Helper function to create a (synthetic) labels Numpy array for use with 3D plotting.

    This function is useful when your spatial data only provides XY coordinates and
    not an actual labels array. There are various options to control what each leaf
    will look like. An additional column in the cell meta will be added in the form
    `{attribute_key}_label`, indicating what label in the returned array corresponds
    to each cell.

    Internally, each cell is represented as an ellipse. An ellipse is parameterized with
    three values: angle, long axis and short axis. The distribution arguments to this function
    control how each cell will look. By default, all cells are perfect circles.

    Args:
        tree: CassiopieaTree to generate labels for
        attribute_key: Attribute name in the `cell_meta` of the tree containing coordinates.
            All columns of the form `{attribute_key}_i` where `i` is an integer `0...` will be used.
        angle_distribution: Callable that takes no arguments and returns a float between 0 and 360,
            which is the angle of the ellipse.
        long_axis_distribution: Callable that generates the of the long axis length. If `absolute=True`,
            the values generated from this function are scaled to half of the minimum nearest-neighbor
            distance between any two coordinates. Otherwise, the values returned are used as literal
            lengths.
        short_axis_distribution: Similar to `long_axis_distribution`, but for the short axis. If this
            argument is not provided, `long_axis_distribution` is used for this purpose.
        absolute: Whether or not the lengths sampled from `long_axis_distribution` and
            `short_axis_distribution` should be considered as absolute lenghts, or should be scaled
            such that no two cells should overlap when all axes are 1.

    Returns:
        A synthetic labels array that can be used for 3D plotting.

    Raises:
        PlottingError if there are not exactly two spatial coordinates.
    """
    if short_axis_distribution is None:
        short_axis_distribution = long_axis_distribution

    meta = tree.cell_meta.copy()
    if meta is None:
        raise PlottingError("CassiopeiaTree must contain cell meta.")
    columns = []
    i = 0
    while True:
        column = f'{attribute_key}_{i}'
        if column in meta.columns:
            columns.append(column)
            i += 1
        else:
            break
    if len(columns) != 2:
        raise PlottingError(f"Only 2-dimensional data is supported, but found {len(columns)} dimensions.")

    coordinates = meta[columns].values

    # Compute scale if not absolute
    scale = 1
    if not absolute:
        nn = neighbors.NearestNeighbors(n_neighbors=1)
        nn.fit(coordinates)
        distances = nn.kneighbors(return_distance=True)[0].flatten()
        scale = max(1, distances.min() / 2)

    shape = tuple(coordinates.astype(int).max(axis=0) + 1)
    labels = np.zeros(shape, dtype=int)
    leaf_to_label = {}
    for leaf, coord in zip(meta.index, coordinates):
        angle = angle_distribution()
        long_axis = max(int(long_axis_distribution() * scale), 1)
        short_axis = max(int(short_axis_distribution() * scale), 1)
        center = tuple(coord.astype(int))
        ellipse = cv2.ellipse(
            np.zeros(shape, dtype=np.uint8), center, (long_axis, short_axis), angle, 0, 360, 1, -1
        ).astype(bool)
        ellipse[center] = True

        label = len(leaf_to_label) + 1
        labels[ellipse] = label
        leaf_to_label[leaf] = label

    # Make sure centers of each leaf is always that leaf
    for leaf, coord in zip(meta.index, coordinates):
        labels[tuple(coord.astype(int))] = leaf_to_label[leaf]

    # Add label column
    meta[f'{attribute_key}_label'] = meta.index.map(leaf_to_label)
    tree.cell_meta = meta

    return labels

class Tree3D:
    """Create a 3D projection of a tree onto a 2D surface.

    This class provides various wrappers around Pyvista, which is used for 3D rendering.

    Example:
        # When labels aren't available, they can be synthetically created by using
        # `labels_from_coordinates`, as so. The tree must contain spatial coordinates
        # in the cell meta.
        labels = cas.pl.labels_from_coordinates(tree)

        tree3d = cas.pl.Tree3D(tree, labels)
        tree3d.add_image(img)  # img is a Numpy array with the same shape as labels
        tree3d.plot()

    Hotkeys:
        1-9: Cut the tree to this many branches.
        p: Select a subclone and highlight it. Branches and cells not in this subclone
            are dimmed out. Only available when "Enable node selection" is checked.
        r: Reset root of the displayed tree to be the root of the actual tree, and reset the view.
        h: Unselect selected node (which was selected using p).
        s: Set the root of the displayed tree as the selected node (which was selected using p).


    Args:
        tree: The Cassiopeia tree to plot. The leaf names must be string-casted integers.
        labels: A Numpy array containing cell labels on a 2D surface. This array must contain all the
            cells in the `tree`, but as integers.
        offset: Offset to give to tree and subclone shading. This option exists because in some cases
            if the tree and subclone shading is placed at the same height as the image, weird clipping
            happens.
        downscale: Downscale all images by this amount. This option is recommended for more responsive
            visualization.
        cmap: Colormap to use. Defaults to the Godsnot color palette, as defined at
            https://github.com/scverse/scanpy/blob/master/scanpy/plotting/palettes.py
        attribute_key: Attribute key to use as the integer labels for each leaf node.
            A column with the name `{attribute_key}_label` will be looked up in the cell meta.
    """
    def __init__(
        self,
        tree: CassiopeiaTree,
        labels: np.ndarray,
        offset: float = 1.,
        downscale: float = 1.,
        cmap: Optional[List[str]] = None,
        attribute_key: str = "spatial",
    ):
        # Check optional dependencies.
        if None in (cv2, pv, measure):
            raise PlottingError(
                "Some required modules were not found. Make sure you installed Cassiopeia with "
                "the `spatial` extras, or run `pip install cassiopeia-lineage[spatial]`."
            )

        # Caches. These come first because initialization may cache stuff.
        self.cut_tree_cache = {}
        self.place_nodes_cache = {}
        self.tree = tree
        self.labels = labels  # cell labels image
        self.offset = offset
        self.downscale = downscale
        self.scale = max(*labels.shape) * downscale

        self.init_label_mapping(f'{attribute_key}_label')

        self.plotter = pv.Plotter()
        self.node_actors = {}
        self.branch_actors = {}
        self.subclone_actor = None
        self.text_actors = {}

        self.node_colors = {}

        resized_labels = cv2.resize(labels, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_NEAREST)
        self.images = {}
        self.image_dims = resized_labels.shape + (1,)
        self.image_actors = {}

        self.init_nodes()
        self.init_subclones()

        # Initial visualization values
        self.root = tree.root
        self.time = self.tree.get_time(self.root)
        self.shown_images = []
        self.subclone_sigma = self.scale / 40
        self.leaves = sorted(self.cut_tree(self.root, self.time))
        self.show_nodes = False
        self.show_text = False
        self.selected_node = None

        # Widget values
        self.checkbox_size = 30
        self.checkbox_border_size = 2

        # Colormap to use. Colors are cycled through when more are needed.
        self.cmap = cmap or godsnot_102

    def init_label_mapping(self, key):
        """Initialize label mappings."""
        # Construct leaf-to-label mapping
        self.leaf_to_label = {}
        if self.tree.cell_meta is None or key not in self.tree.cell_meta:
            warnings.warn(
                f"Failed to locate {key} column in cell meta. "
                "Leaf names casted as integers will be used as the labels.", PlottingWarning
            )
            self.leaf_to_label = {leaf: int(leaf) for leaf in self.tree.leaves}
        else:
            self.leaf_to_label = dict(self.tree.cell_meta[key])
        self.label_to_leaf = {label: leaf for leaf, label in self.leaf_to_label.items()}

        # Check labels
        if not np.isin([self.leaf_to_label[leaf] for leaf in self.tree.leaves], self.labels).all():
            raise PlottingError("Label array must contain all leaves in the tree.")

        self.regionprops = {prop.label: prop for prop in measure.regionprops(self.labels)}


    def init_nodes(self):
        """Initialize node information.
        """
        self.nodes = self.tree.nodes
        self.times = self.tree.get_times()
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.node_coordinates = np.full((len(self.tree.nodes), 3), np.nan)

        areas = np.zeros(self.node_coordinates.shape[0], dtype=int)
        for label, props in self.regionprops.items():
            leaf = self.label_to_leaf[label]
            if leaf in self.tree.leaves:
                i = self.node_index[leaf]
                areas[i] = props.area
                self.node_coordinates[i] = props.centroid + (0,)

        queue = deque(set(self.tree.parent(leaf) for leaf in self.tree.leaves))
        processed = set(self.tree.leaves)

        while queue:
            node = queue.popleft()

            children = self.tree.children(node)
            if not all(child in processed for child in children):
                queue.append(node)
                continue

            i = self.node_index[node]
            children_indices = [self.node_index[child] for child in children]
            child_coordinates = self.node_coordinates[children_indices]
            child_areas = areas[children_indices]
            self.node_coordinates[i] = (child_coordinates * child_areas.reshape(-1, 1)).sum(axis=0) / child_areas.sum()
            areas[i] = child_areas.sum()

            processed.add(node)
            if not self.tree.is_root(node):
                queue.append(self.tree.parent(node))
        assert not np.isnan(self.node_coordinates).any()

    def init_subclones(self):
        """Initialize subclone grid.
        """
        self.subclones = self.create_grid()
        self.subclones.origin = (0, 0, 1)

    def get_mask(self, node: str) -> np.ndarray:
        """Helper function to get a boolean mask of where certain subclades are.

        Args:
            node: Node name to select cells

        Returns:
            Boolean mask where True indicates cells in the subclade marked by `node`
        """
        regionprops = []
        if self.tree.is_leaf(node):
            regionprops.append(self.regionprops[self.leaf_to_label[node]])
        else:
            for leaf in self.tree.leaves_in_subtree(node):
                regionprops.append(self.regionprops[self.leaf_to_label[leaf]])

        mask = np.zeros(self.labels.shape, dtype=bool)
        for props in regionprops:
            label = props.label
            min_row, min_col, max_row, max_col = props.bbox
            label_mask = self.labels[min_row:max_row, min_col:max_col] == label
            mask[min_row:max_row, min_col:max_col][label_mask] = True

        return mask

    def create_grid(self) -> "pv.UniformGrid":
        """Helper function to create a Pyvista UniformGrid object with the appropriate
        shape.
        """
        return pv.UniformGrid(dims=self.image_dims)

    def add_image(self, key: str, img: np.ndarray):
        """Add an image so that it may be displayed with the tree.

        Args:
            key: Identifiable name of the image.
            img: Image as a Numpy array
        """
        if img.ndim not in (2, 3):
            raise PlottingError("Only 2- and 3-dimensional images are supported.")
        if img.shape[:2] != self.labels.shape:
            raise PlottingError(
                f"The first two dimensions of the image must have shape {self.labels.shape}."
            )

        img = (cv2.resize(img, None, fx=self.downscale, fy=self.downscale) * 255).astype(np.uint8)

        # Immediately convert to mesh
        grid = self.create_grid()
        grid.point_data['values'] = img.reshape(np.prod(self.image_dims), -1, order='F')
        self.images[key] = grid

        # Always show the first one
        if len(self.images) == 1:
            self.shown_images = [key]

    def cut_tree(self, root: str, time: float) -> List[str]:
        """Cut the tree at a specific time after the time specified by the root.

        Args:
            root: Root to calculate the time delta from.
            time: Time from root to cut the tree at.

        Returns:
            A list of leaf nodes for the cut tree. These nodes may be internal nodes.
        """
        key = (root, time)
        if key in self.cut_tree_cache:
            return self.cut_tree_cache[key]

        root_time = self.tree.get_time(root)

        leaves = []
        for n1, n2 in self.tree.breadth_first_traverse_edges(root):
            time1 = self.tree.get_time(n1) - root_time
            time2 = self.tree.get_time(n2) - root_time

            if (time1 <= time and time2 > time) or (self.tree.is_leaf(n2) and time2 <= time):
                leaves.append(n2)
        self.cut_tree_cache[key] = leaves
        return leaves

    def place_nodes(self, root: str, leaves: List[str]) -> np.ndarray:
        """Place a subset of nodes in xyz coordinates.

        Only nodes in the path between `root` and each node in `leaves` are placed.

        Args:
            root: Root node of the tree. Does not have to be the actual root of the tree.
            leaves: List of nodes at the bottom of the tree. May be internal nodes.

        Returns:
            A Numpy array of N x 3, where N is the total number of nodes in the tree.
            Nodes that were not placed will have `np.nan` as its three coordinates.
            Each row is indexed to each node according to `self.node_index`.
        """
        if not all(root in self.tree.get_all_ancestors(leaf) for leaf in leaves):
            raise PlottingError(f"The desired root {root} is not an ancestor of all the leaves {leaves}.")

        key = (root, frozenset(leaves))
        if key in self.place_nodes_cache:
            return self.place_nodes_cache[key]

        coordinates = np.full(self.node_coordinates.shape, np.nan)
        leaf_indices = [self.node_index[leaf] for leaf in leaves]
        coordinates[leaf_indices] = self.node_coordinates[leaf_indices]

        queue = deque(set(self.tree.parent(leaf) for leaf in leaves))
        processed = set(leaves)

        while queue:
            node = queue.popleft()
            children = self.tree.children(node)

            if node != root:
                queue.append(self.tree.parent(node))

            if not all(child in processed for child in children):
                queue.append(node)
                continue

            i = self.node_index[node]
            children_indices = [self.node_index[child] for child in children]
            child_coordinates = coordinates[children_indices]
            coordinates[i] = child_coordinates.mean(axis=0)
            coordinates[i,2] = child_coordinates[:,2].min() - 1
            processed.add(node)
        self.place_nodes_cache[key] = coordinates
        return coordinates

    def place_branches(self, root: str, coordinates: np.ndarray) -> Dict[Tuple[str, str], np.ndarray]:
        """Place a subset of branches in xyz coordinates.

        Only nodes in the path between `root` and each node that have valid coordinates
        in `coordinates` are placed.

        Args:
            root: Root node of the tree. Does not have to be the actual root of the tree.
            coordinates: Coordinates of nodes as produced by `self.place_nodes`.

        Returns:
            Dictionary of branch tuples (node1, node2) as keys and branch coordinates as
            a Numpy arrays as values.
        """
        branches = {}
        for n1, n2 in self.tree.breadth_first_traverse_edges(root):
            i1 = self.node_index[n1]
            i2 = self.node_index[n2]
            if np.isnan(coordinates[i1]).any() or np.isnan(coordinates[i2]).any():
                continue

            branch_coords = interpolate_branch(coordinates[i1], coordinates[i2])
            branches[(n1, n2)] = branch_coords
        return branches

    def render_node(self, coords: np.ndarray, radius: float) -> "pv.Sphere":
        """Helper function to create a Pyvista object representing a node.

        Args:
            coords: XYZ coordinates as a 1-dimensional Numpy array
            radius: Radius of sphere

        Returns:
            Pyvista object representing a node.
        """
        coords = coords.copy()
        coords[0] *= self.downscale
        coords[1] *= self.downscale
        coords[2] *= -self.scale * 0.1
        coords[2] += self.offset
        return pv.Sphere(center=coords, radius=radius)

    def render_branch(self, branch_coords: np.ndarray, radius: float) -> "pv.Tube":
        """Helper function to create a Pyvista object representing a branch.

        Args:
            coords: XYZ coordinates as a 2-dimensional Numpy array
            radius: Radius of tube

        Returns:
            Pyvista object representing a branch.
        """
        coords = branch_coords.copy()
        coords[:,0] *= self.downscale
        coords[:,1] *= self.downscale
        coords[:,2] *= -self.scale * 0.1
        coords[:,2] += self.offset
        branch = polyline_from_points(coords)
        return branch.tube(radius=radius)

    def clear_node_actors(self):
        """Clear nodes from visualization."""
        for actor in self.node_actors.values():
            self.plotter.remove_actor(actor)
        self.node_actors = {}

    def clear_branch_actors(self):
        """Clear branches from visualization."""
        for actor in self.branch_actors.values():
            self.plotter.remove_actor(actor)
        self.branch_actors = {}

    def clear_subclone_actor(self):
        """Clear subclone shades from visualization."""
        if self.subclone_actor is not None:
            self.plotter.remove_actor(self.subclone_actor)
        self.subclone_actor = None

    def clear_image_actors(self):
        """Clear images from visualization."""
        for actor in self.image_actors.values():
            self.plotter.remove_actor(actor)
        self.image_actors = {}

    def set_subclone_sigma(self, sigma: float):
        """Set subclone shade blur strength.

        Args:
            sigma: Blur strength
        """
        if sigma == self.subclone_sigma:
            return
        self.subclone_sigma = sigma
        self.update_subclones()

    def set_height(self, height: int):
        """Set the height of the tree.

        The height is defined as the number of branches from the root.

        Args:
            height: Cutoff height as an integer
        """
        times = sorted(set(
            self.times[node] for node in self.tree.depth_first_traverse_nodes(source=self.root)
        ))
        if height <= len(times):
            self.set_time(times[height])
        else:
            self.set_time(times[-1])

    def set_time(self, time: float):
        """Set the time of the tree.

        Args:
            time: Cutoff time
        """
        if time == self.time:
            return
        self.time = time
        leaves = sorted(self.cut_tree(self.root, self.time))
        if self.leaves != leaves:
            self.leaves = leaves
            self.update_branches()
            self.update_subclones()
            if f'node:{self.selected_node}' in self.node_actors:
                self.select_node(self.selected_node)
                self.clear_picked_mesh()
            else:
                self.reset_selected_node()
            self.update_texts()

    def set_node_picking(self, flag: bool):
        """Helper function to setup node selection.

        Args:
            flag: True to enable node selection, False otherwise.
        """
        self.show_nodes = flag
        for actor in self.node_actors.values():
            actor.SetVisibility(flag)
        if not flag:
            self.plotter.remove_actor('_mesh_picking_selection')

    def set_shown_image(self, key: str, show: bool):
        """Helper function to show an image.

        Args:
            key: Image key
            show: True to show, False otherwise.
        """
        update = False
        if show:
            if key not in self.shown_images:
                self.shown_images.append(key)
                update = True
        else:
            if key in self.shown_images:
                self.shown_images.remove(key)
                update = True
        if update:
            self.update_images()

    def set_root(self, root: str):
        """Helper function to set the root of the tree.

        Args:
            root: Desired root node
        """
        self.reset_selected_node()
        self.update_texts()
        if root == self.root:
            return
        self.root = root
        self.leaves = sorted(self.cut_tree(self.root, self.time))
        self.update_branches()
        self.update_subclones()

    def set_selected_node_as_root(self):
        """Helper function to set the selected node as the root."""
        mesh = self.plotter.picked_mesh
        if mesh is not None:
            node = self.nodes[self.leaf_to_label[mesh.field_data['node'][0]]]
            self.set_root(node)

    def select_node_mesh(self, mesh):
        """Helper function remember the selected node."""
        node = self.nodes[self.leaf_to_label[mesh.field_data['node'][0]]]
        self.select_node(node)

    def select_node(self, node: str):
        """Helper function to select a node.

        When a node is selected, its children nodes and branches are highlighted,
        including subclone shading. All other elements are dimmed out.

        Args:
            node: Selected node
        """
        self.selected_node = node
        self.update_texts()

        reset = node is None
        selected = set(self.tree.depth_first_traverse_nodes(
            source=node if not reset else self.root
        ))
        for name, actor in self.node_actors.items():
            node = name[len('node:'):]
            func = highlight if node in selected else lowlight
            color = func(self.node_colors[node]) if not reset else self.node_colors[node]
            actor.GetProperty().SetColor(color)

            if node == self.root:
                branch_name = 'branch:synthetic_root'
                branch_actor = self.branch_actors[branch_name]
                branch_actor.GetProperty().SetColor(color)
            elif node != self.tree.root:
                branch_name = f'branch:{self.tree.parent(node)}-{node}'
                if branch_name in self.branch_actors:
                    branch_actor = self.branch_actors[branch_name]
                    branch_actor.GetProperty().SetColor(color)

        key = sha256(';'.join(self.leaves).encode('utf-8')).hexdigest()
        labels_key = f'labels:{key}'
        leaf_labels = np.array(self.subclones.point_data[labels_key]).reshape(*self.image_dims[:2], order='F')
        colors = [(1, 1, 1)]
        for i, leaf in enumerate(self.leaves):
            func = highlight if leaf in selected else lowlight
            colors.append(func(to_rgb(self.cmap[i % len(self.cmap)])) if not reset else to_rgb(self.cmap[i % len(self.cmap)]))
        colors = np.pad(np.array(colors), ((0, 0), (0, 1)))
        colors[1:,3] = 1

        leaf_colors = colors[leaf_labels]
        mask = leaf_colors[:,:,3] > 0
        blur = cv2.GaussianBlur(leaf_colors, (0, 0), sigmaX=self.subclone_sigma)
        alpha = cv2.GaussianBlur(mask.astype(float), (0, 0), sigmaX=self.subclone_sigma)
        alpha -= alpha.min()
        alpha /= alpha.max()
        blur[:,:,3] = alpha
        self.subclones.point_data['values'] = blur.reshape(np.prod(self.image_dims), -1, order='F')
        self.subclones.set_active_scalars('values')
        self.subclone_actor = self.plotter.add_mesh(self.subclones, rgba=True, name='subclones', pickable=False)

    def clear_picked_mesh(self):
        """Helper function to clear the selected mesh."""
        self.plotter.remove_actor('_mesh_picking_selection')
        self.plotter._picked_mesh = None

    def reset_selected_node(self):
        """Helper function to clear the selected node."""
        self.clear_picked_mesh()
        self.select_node(None)
        self.update_texts()

    def update_actors(self, actors: Dict[str, 'vtk.vtkActor'], new_actors: Dict[str, 'vtk.vtkActor']):
        """Helper function to update a set of actors.

        Args:
            actors: Dictionary of actors that are currently displayed.
            new_actors: Dictionary of actors to replace the existing actors with.
        """
        for name, actor in actors.items():
            if name not in new_actors:
                self.plotter.remove_actor(actor)
        actors.clear()
        actors.update(new_actors)

    def update_texts(self):
        """Update displayed text.

        The following text is updated.
        * The current root
        * The current time
        * The selected node (if one is selected)
        """
        new_actors = {}

        # Root
        name = 'root'
        actor = self.plotter.add_text(
            f'Root: {self.root}',
            position=(0.2, 0.9),
            viewport=True,
            color='black',
            font_size=self.checkbox_size * (1 / 4),
            name=name
        )
        actor.SetVisibility(self.show_text)
        new_actors[name] = actor

        # Times
        name = 'times'
        actor = self.plotter.add_text(
            f'Time range: {self.tree.get_time(self.root)} - {min(self.tree.get_time(leaf) for leaf in self.leaves)}',
            position=(0.2, 0.87),
            viewport=True,
            color='black',
            font_size=self.checkbox_size * (1 / 4),
            name=name
        )
        actor.SetVisibility(self.show_text)
        new_actors[name] = actor

        # Selected node
        if self.selected_node is not None:
            name = 'node'
            actor = self.plotter.add_text(
                f'Selected: {self.selected_node}',
                position=(0.2, 0.84),
                viewport=True,
                color='black',
                font_size=self.checkbox_size * (1 / 4),
                name=name
            )
            actor.SetVisibility(self.show_text)
            new_actors[name] = actor
        self.update_actors(self.text_actors, new_actors)

    def update_images(self):
        """update displayed image(s). """
        new_actors = {}
        for i, key in enumerate(self.shown_images):
            self.images[key].origin = (0, 0, -i * self.scale * 0.25)
            name = f'image:{key}'
            actor = self.plotter.add_mesh(self.images[key], rgba=True, name=name, pickable=False)
            new_actors[name] = actor
        self.update_actors(self.image_actors, new_actors)

    def update_branches(self):
        """Update displayed branches."""
        root = self.root
        leaves = self.leaves
        cmap = self.cmap

        coordinates = self.place_nodes(root, leaves)
        branches = self.place_branches(root, coordinates)

        # NODES
        self.node_colors = {
            leaf: to_rgb(cmap[i % len(cmap)]) for i, leaf in enumerate(leaves)
        }
        queue = deque(leaves)
        new_actors = {}
        while queue:
            node = queue.popleft()

            if node not in self.node_colors:
                children = self.tree.children(node)
                if not all(child in self.node_colors for child in children):
                    queue.append(node)
                    continue
                color = average_mixing(*[self.node_colors[child] for child in children])
                self.node_colors[node] = color
            else:
                color = self.node_colors[node]

            name = f'node:{node}'
            i = self.node_index[node]
            sphere = self.render_node(
                coordinates[i],
                self.scale * 0.00175 * np.log2(max(len(self.tree.leaves_in_subtree(node)), 2))
            )
            sphere.add_field_data(i, 'node')
            actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True, name=name, pickable=True)
            actor.SetVisibility(self.show_nodes)
            new_actors[name] = actor

            if node != self.root:
                queue.append(self.tree.parent(node))
        self.update_actors(self.node_actors, new_actors)

        # BRANCHES
        new_actors = {}
        for (n1, n2), branch_coords in branches.items():
            branch = self.render_branch(
                branch_coords,
                self.scale * 0.001 * np.log2(max(len(self.tree.leaves_in_subtree(n2)), 2))
            )

            name = f'branch:{n1}-{n2}'
            actor = self.plotter.add_mesh(branch, color=self.node_colors[n2], smooth_shading=True, name=name, pickable=False)
            new_actors[name] = actor

        # Synthetic root
        root_coords = coordinates[self.node_index[root]]
        branch_coords = np.array([
            (root_coords[0], root_coords[1], root_coords[2] - 1), root_coords
        ])
        branch = self.render_branch(
            branch_coords,
            self.scale * 0.001 * np.log2(max(len(self.tree.leaves_in_subtree(root)), 2))
        )

        name = 'branch:synthetic_root'
        actor = self.plotter.add_mesh(branch, color=self.node_colors[root], smooth_shading=True, name=name, pickable=False)
        new_actors[name] = actor
        self.update_actors(self.branch_actors, new_actors)

    def update_subclones(self):
        """Update displayed subclone shading."""
        leaves = self.leaves
        cmap = self.cmap

        key = sha256(';'.join(leaves).encode('utf-8')).hexdigest()
        labels_key = f'labels:{key}'
        colors_key = f'colors:{key}'
        if colors_key not in self.subclones.point_data:
            colors = np.array([to_rgba(cmap[i % len(cmap)]) for i in range(len(leaves))])
            colors = np.insert(colors, 0, [1, 1, 1, 0], axis=0)
            leaf_labels = np.zeros(self.labels.shape, dtype=int)
            for i, leaf in enumerate(leaves):
                mask = self.get_mask(leaf)
                leaf_labels[mask] = i+1
            leaf_labels = cv2.resize(leaf_labels, None, fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_NEAREST)
            self.subclones.point_data[labels_key] = leaf_labels.flatten(order='F')
            leaf_colors = colors[leaf_labels]
            self.subclones.point_data[colors_key] = leaf_colors.reshape(np.prod(self.image_dims), -1, order='F')

        leaf_colors = np.array(self.subclones.point_data[colors_key]).reshape(self.image_dims[0], self.image_dims[1], -1, order='F')
        mask = leaf_colors[:,:,3] > 0

        blur = cv2.GaussianBlur(leaf_colors, (0, 0), sigmaX=self.subclone_sigma)
        alpha = cv2.GaussianBlur(mask.astype(float), (0, 0), sigmaX=self.subclone_sigma)
        alpha -= alpha.min()
        alpha /= alpha.max()
        blur[:,:,3] = alpha
        self.subclones.point_data['values'] = blur.reshape(np.prod(self.image_dims), -1, order='F')
        self.subclones.set_active_scalars('values')
        self.subclone_actor = self.plotter.add_mesh(self.subclones, rgba=True, name='subclones', pickable=False)

    def add_blur_slider(self):
        """Add slider to control subclone blur strength."""
        self.plotter.add_slider_widget(
            self.set_subclone_sigma,
            (1, self.scale / 20),
            self.subclone_sigma,
            title='Blur',
            color='black',
            pointa=(0.7, 0.9),
            pointb=(0.9, 0.9)
        )

    def add_time_slider(self):
        """Add slider to control current time."""
        self.plotter.add_slider_widget(
            self.set_time,
            (self.tree.get_time(self.root), min(self.tree.get_max_depth_of_tree() - self.tree.get_time(self.root), 5)),
            self.time,
            title='Time',
            color='black',
            pointa=(0.85, 0.6),
            pointb=(0.85, 1.0)
        )

    def add_height_key_events(self):
        """Add key events such that pressing numbers from 1 through 9 controls the tree height."""
        for i in range(1, 10):
            self.plotter.add_key_event(str(i), partial(self.set_height, i-1))

    def add_image_checkboxes(self):
        for i, key in enumerate(self.images):
            self.plotter.add_checkbox_button_widget(
                partial(self.set_shown_image, key),
                value=key in self.shown_images,
                position=(10., 10. + (i+1) * self.checkbox_size * 1.1),
                size=self.checkbox_size,
                border_size=self.checkbox_border_size,
                color_on='black',
                color_off='lightgrey',
                background_color='grey',
            )
            self.plotter.add_text(
                f'Show {key}', position=(10 + self.checkbox_size * 1.1, 10. + (i+1) * self.checkbox_size * 1.1),
                color='black', font_size=self.checkbox_size * (2 / 5)
            )

    def add_node_picking(self):
        """Enable node selection."""
        self.plotter.enable_mesh_picking(
            self.select_node_mesh,
            show=True, show_message=False, style='surface'
        )
        self.plotter.add_key_event('h', self.reset_selected_node)
        self.plotter.add_key_event('r', partial(self.set_root, self.tree.root))

        self.plotter.add_checkbox_button_widget(
            self.set_node_picking,
            value=self.show_nodes,
            position=(10., 10.),
            size=self.checkbox_size,
            border_size=self.checkbox_border_size,
            color_on='black',
            color_off='lightgrey',
            background_color='grey',
        )
        self.plotter.add_text(
            'Enable node selection', position=(10 + self.checkbox_size * 1.1, 10.),
            color='black', font_size=self.checkbox_size * (2 / 5)
        )

        self.plotter.add_key_event('s', self.set_selected_node_as_root)

    def add_widgets(self):
        """Add widgets."""
        self.add_blur_slider()
        self.add_height_key_events()
        # self.add_time_slider()
        self.add_image_checkboxes()
        self.add_node_picking()

    def plot(self, plot_tree: bool = True, add_widgets: bool = True, show: bool = True):
        """Display 3D render.

        Args:
            plot_tree: Immediately render the tree.
                If False, the initial plot will not have any tree rendered.
            add_widgets: Add widgets to scene.
            show: Whether to show the plot immmediately.
        """
        self.update_images()

        if add_widgets:
            self.add_widgets()
            self.show_text = True

        if plot_tree:
            self.update_subclones()
            self.update_branches()
            self.update_texts()

        self.plotter.set_background('white')
        self.plotter.add_axes(viewport=(0, 0.75, 0.2, 0.95))
        self.plotter.enable_lightkit()
        self.plotter.enable_anti_aliasing()
        if show:
            self.plotter.show()

    def reset(self):
        """Helper function to reset everything."""
        self.branch_actors = {}
        self.subclone_actor = None
        self.image_actors = {}
        self.shown_images = []
        self.plotter.clear()