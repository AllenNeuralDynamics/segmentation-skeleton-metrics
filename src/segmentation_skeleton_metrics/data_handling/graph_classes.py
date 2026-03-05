"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of NetworkX.Graph called SkeletonGraph.

"""

from collections import defaultdict
from io import StringIO
from scipy.spatial import distance, KDTree

import networkx as nx
import numpy as np
import os

from segmentation_skeleton_metrics.utils import util


class SkeletonGraph(nx.Graph):
    """
    A subclass of the NetworkX.Graph designed for graphs built from SWC files.
    This class extends the functionality of the standard Graph class by adding
    support for handling node labels and voxel coordinates. In this subclass,
    node IDs serve as direct indices for accessing the labels and voxels
    attributes.

    Attributes
    ----------
    anisotropy : numpy.ndarray
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    filename : str
        Filename of SWC file that graph is built from.
    run_length : float
        Physical path length of the graph.
    voxels : numpy.ndarray
        A 3D array that contains a voxel coordinate for each node.
    """

    colors = [
        "# COLOR 1.0 0.0 1.0",  # pink
        "# COLOR 0.0 1.0 1.0",  # cyan
        "# COLOR 1.0 1.0 0.0",  # yellow
        "# COLOR 0.0 0.5 1.0",  # blue
        "# COLOR 1.0 0.5 0.0",  # orange
        "# COLOR 0.5 0.0 1.0",  # purple
        "# COLOR 0.0 0.8 0.8",  # teal
        "# COLOR 0.6 0.0 0.6",  # plum
    ]

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), name=None):
        """
        Initializes a SkeletonGraph, including setting the anisotropy and
        initializing the run length attributes.

        Parameters
        ----------
        anisotropy : ArrayLike, optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        name : str, optional
            Name of the SWC file that graph is built from.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.anisotropy = np.array(anisotropy)
        self.filename = None
        self.kdtree = None
        self.name = name
        self.node_voxel = None
        self.run_length = 0

    def set_kdtree(self):
        """
        Builds a KD-Tree from the node physical coordinates.
        """
        self.kdtree = KDTree(self.node_voxel * self.anisotropy)

    def set_voxels(self, voxels):
        """
        Sets the "node_voxel" attribute for the graph.

        Parameters
        ----------
        voxels : ArrayLike
            Voxel coordinates for each node in the graph.
        """
        self.node_voxel = np.array(voxels, dtype=np.int32)

    def set_filename(self, filename):
        """
        Sets the filename attribute which corresponds to the SWC file that the
        graph is built from.

        Parameters
        ----------
        filename : str
            Name of the SWC file that the graph is built from.
        """
        self.filename = filename

    def set_nodes(self, num_nodes):
        """
        Adds nodes to the graph. The nodes are assigned indices from 0 to
        "num_nodes".

        Parameters
        ----------
        num_nodes : int
            Number of nodes to be added to the graph.
        """
        self.add_nodes_from(np.arange(num_nodes))

    # --- Getters ---
    def dist(self, i, j):
        """
        Computes the Euclidean distance between the voxel coordinates
        cooresponding to the given nodes.

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.

        Returns
        -------
        float
            Distance between voxel coordinates of the given nodes.
        """
        return distance.euclidean(self.node_voxel[i], self.node_voxel[j])

    def physical_dist(self, i, j):
        """
        Computes the Euclidean distance between the physical coordinates
        cooresponding to the given nodes.

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.

        Returns
        -------
        float
            Euclidean distance between physical coordinates of the given
            nodes.
        """
        return distance.euclidean(self.node_xyz(i), self.node_xyz(j))

    def node_xyz(self, i):
        """
        Gets the physical coordinate of the given node.

        Parameters
        ----------
        i : int
            Node ID.

        Returns
        -------
        numpy.ndarray
            Physical coordinate of the given node.
        """
        return self.node_voxel[i][::-1] * self.anisotropy

    def color(self):
        """
        Gets the display color of the skeleton to be written to an SWC file.

        Returns
        -------
        str
            String representing the color in the format "# COLOR R G B".
        """
        return util.sample_once(SkeletonGraph.colors)

    def radius(self):
        """
        Gets the radius of the skeleton to be written to an SWC file.

        Returns
        -------
        float
            Radius of the skeleton.
        """
        return 3

    def prune_branches(self):
        """
        Placeholder method to be implemented by subclasses.
        """
        pass

    def run_length_from(self):
        """
        Placeholder method to be implemented by subclasses.
        """
        pass

    # --- Writers ---
    def to_zipped_swcs(self, zip_writer, use_color=True):
        """
        Writes the graph to an SWC file format, which is then stored in a ZIP
        archive.

        Parameters
        ----------
        zip_writer : zipfile.ZipFile
            A ZipFile object that will store the generated SWC file.
        use_color : bool, optional
            Indication of whether to use the class color. Default is True.
        """
        for cnt, nodes in enumerate(map(list, nx.connected_componets(self))):
            filename = f"{self.name}.{cnt}.swc"
            zip_writer.writestr(filename, self._generate_swc_text(nodes[0]))

    def to_swcs(self, output_dir, use_color=True):
        """
        Writes the graph to an SWC file format, which is then stored in the
        given local directory.
        archive.

        Parameters
        ----------
        output_dir : str
            Path to directory that SWC files will be written to.
        use_color : bool, optional
            Indication of whether to use the class color. Default is True.
        """
        for cnt, nodes in enumerate(map(list, nx.connected_components(self))):
            path = os.path.join(output_dir, f"{self.name}.{cnt}.swc")
            with open(path, "w") as file:
                file.write(self._generate_swc_text(nodes[0], use_color))

    def _generate_swc_text(self, root, use_color=True):
        """
        Generates the text to store that graph as an SWC file.

        Parameters
        ----------
        root : int
            Node ID used as root of SWC file.
        use_color : bool, optional
            Indication of whether to use the class color. Default is True.
        """

        # Subroutines
        def write_entry(node, parent):
            """
            Writes a line in an SWC file

            Parameters
            ----------
            node : int
                Node ID.
            parent : int
                Node ID of the parent.
            """
            x, y, z = self.node_xyz(i)
            r = self.radius()
            node_id = cnt
            parent_id = node_to_idx[parent]
            node_to_idx[node] = node_id
            text_buffer.write(f"\n{node_id} 2 {x} {y} {z} {r} {parent_id}")

        # Create writer
        text_buffer = StringIO()
        text_buffer.write(self.color()) if self.use_color else None
        text_buffer.write("\n# id, type, z, y, x, r, pid")

        # Write entries
        cnt = 1
        node_to_idx = defaultdict(lambda: -1)
        for i, j in nx.dfs_edges(self, source=root):
            # Special Case: Root
            if len(node_to_idx) == 0:
                write_entry(i, -1)

            # General Case: Non-Root
            cnt += 1
            write_entry(j, i)
        return text_buffer.getvalue()


class LabeledGraph(SkeletonGraph):
    """
    Subclass of SkeletonGraph with the provides support for node-level
    labeling and tracking label assignments.
    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), name=None):
        """
        Instantiates a LabeledGraph object.

        Parameters
        ----------
        anisotropy : ArrayLike
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope.
        name : str or None
            Name of the graph which is derived from the SWC filename. Default
            is None.
        """
        # Call parent class
        super().__init__(anisotropy=anisotropy, name=name)

        # Instance attributes
        self.labels_with_merge = set()
        self.labeled_run_length = 0

    def init_node_labels(self):
        """
        Initializes the "node_label" attribute for the graph.
        """
        error_msg = "Graph must have nodes to initialize node labels!"
        assert self.number_of_nodes() > 0, error_msg
        self.node_label = np.zeros((self.number_of_nodes()), dtype=int)

    # --- Core Routines ---
    def node_labels(self):
        """
        Gets the unique non-zero label values in the "labels" attribute.

        Returns
        -------
        node_labels : Set[int]
            Unique non-zero label values assigned to nodes in the graph.
        """
        node_labels = set(np.unique(self.node_label))
        node_labels.discard(0)
        return node_labels

    def nodes_with_label(self, label):
        """
        Gets the IDs of nodes that have the specified label value.

        Parameters
        ----------
        label : int
            Label value to search for in the "node_label" attribute.

        Returns
        -------
        numpy.ndarray
            Node IDs that have the specified label.
        """
        return np.where(self.node_label == label)[0]

    def remove_nodes_with_label(self, label):
        """
        Removes nodes with the given label

        Parameters
        ----------
        label : int
            Label to be deleted from graph.
        """
        nodes = self.nodes_with_label(label)
        self.remove_nodes_from(nodes)

    def run_length_from(self, root):
        """
        Computes the physical path length of the label-based connected
        component that contains "root".

        Parameters
        ----------
        root : int
            Node contained in connected component to compute run length of.

        Returns
        -------
        run_length : float
            Physical path length.
        """
        # Initializations
        root_label = self.node_label[root]
        run_length = 0

        # Main
        queue = [(root, root)]
        visited = set([root])
        while queue:
            # Visit node
            i, j = queue.pop()
            run_length += self.physical_dist(i, j)

            # Update queue
            for k in self.neighbors(j):
                if k not in visited and self.node_label[k] == root_label:
                    queue.append((j, k))
                    visited.add(k)
        return run_length

    def update_node_labels(self, nodes, label):
        """
        Updates the label of the given nodes with a specified label.

        Parameters
        ----------
        nodes : List[int]
            Nodes to be updated.
        label : int
            New label of nodes.
        """
        for i in nodes:
            self.node_label[i] = label

    # --- Helpers ---
    def get_bbox(self, nodes):
        """
        Calculates the minimal bounding box containing the voxel coordinates
        for a given collection of nodes. Useful for labelling the graph by
        reading patches from the segmentation.

        Parameters
        ----------
        nodes : Container
            Node indices for which to compute the bounding box.

        Returns
        -------
        Dict[str, Tuple[int]
            Dictionary containing the bounding box coordinates:
            - "min": minimum voxel coordinates along each axis.
            - "max": maximum voxel coordinates along each axis.
        """
        bbox_min = np.inf * np.ones(3)
        bbox_max = np.zeros(3)
        for i in nodes:
            bbox_min = np.minimum(bbox_min, self.node_voxel[i])
            bbox_max = np.maximum(bbox_max, self.node_voxel[i] + 1)
        return {"min": bbox_min.astype(int), "max": bbox_max.astype(int)}

    def color(self):
        """
        Gets the display color of the skeleton to be written to an SWC file.

        Returns
        -------
        str
            String representing the color in the format "# COLOR R G B".
        """
        return "# COLOR 1.0 1.0 1.0"

    def radius(self):
        """
        Gets the radius of the skeleton to be written to an SWC file.

        Returns
        -------
        float
            Radius of the skeleton.
        """
        return 2


class FragmentGraph(SkeletonGraph):
    """
    Subclass of SkeletonGraph for skeletons obtained from a segmentation.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        name=None,
        label=None,
        segment_id=None,
    ):
        """
        Instantiates a FragmentGraph object.

        Parameters
        ----------
        anisotropy : ArrayLike, optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        name : str or None, optional
            Name of the graph which is derived from the SWC filename. Default
            is None.
        label : int, optional
            Graph-level label that corresponds to a node-level label given to
            a ground truth graph. Default is None.
        segment_id : int, optional
            Segment ID of the segment the given skeleton was obtained from.
            Default is None.
        """
        # Call parent class
        super().__init__(anisotropy=anisotropy, name=name)

        # Instance attributes
        self.label = label
        self.segment_id = segment_id

    def prune_branches(self, depth=24):
        """
        Prunes branches with length less than "depth" microns.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.
        depth : float
            Length of branches that are pruned.
        """
        for leaf in util.get_leafs(self):
            branch = [leaf]
            length = 0
            for i, j in nx.dfs_edges(self, source=leaf):
                # Visit edge
                length += self.physical_dist(i, j)
                if length > depth:
                    break

                # Check whether to continue search
                if self.degree(j) == 2:
                    branch.append(j)
                elif self.degree(j) > 2:
                    self.remove_nodes_from(branch)
                    break

    def run_length_from(self, root):
        """
        Computes the physical path length of the connected component that
        contains "root".

        Parameters
        ----------
        root : int
            Node contained in connected component to compute run length of.

        Returns
        -------
        run_length : float
            Physical path length.
        """
        run_length = 0
        queue = [(root, root)]
        visited = set([root])
        while queue:
            # Visit node
            i, j = queue.pop()
            run_length += self.physical_dist(i, j)

            # Update queue
            for k in self.neighbors(j):
                if k not in visited:
                    queue.append((j, k))
                    visited.add(k)
        return run_length
