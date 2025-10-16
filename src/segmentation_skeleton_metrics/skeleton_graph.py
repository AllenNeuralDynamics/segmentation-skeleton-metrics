"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of NetworkX.Graph called SkeletonGraph.

"""

from collections import defaultdict
from io import StringIO
from scipy.spatial import distance

import networkx as nx
import numpy as np

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
    is_groundtruth : bool
        Indication of whether this graph corresponds to a ground truth
        tracing.
    labels : numpy.ndarray
        A 1D array that contains a label value associated with each node.
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

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), is_groundtruth=False):
        """
        Initializes a SkeletonGraph, including setting the anisotropy and
        initializing the run length attributes.

        Parameters
        ----------
        anisotropy : ArrayLike, optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        is_groundtruth : bool, optional
            Indication of whether this graph corresponds to a ground truth
            tracing. Default is False.
        """
        # Call parent class
        super(SkeletonGraph, self).__init__()

        # Instance attributes
        self.anisotropy = np.array(anisotropy)
        self.filename = None
        self.is_groundtruth = is_groundtruth
        self.labels = None
        self.run_length = 0
        self.voxels = None

    def init_labels(self):
        """
        Initializes the "labels" attribute for the graph.
        """
        error_msg = "Graph must have nodes to initialize labels!"
        assert self.number_of_nodes() > 0, error_msg
        self.labels = np.zeros((self.number_of_nodes()), dtype=int)

    def init_voxels(self, voxels):
        """
        Initializes the "voxels" attribute for the graph.

        Parameters
        ----------
        voxels : ArrayLike
            Voxel coordinates for each node in the graph.
        """
        self.voxels = np.array(voxels, dtype=np.int32)

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
    def get_labels(self):
        """
        Gets the unique non-zero label values in the "labels" attribute.

        Returns
        -------
        Set[int]
            Unique non-zero label values assigned to nodes in the graph.
        """
        labels = set(np.unique(self.labels))
        labels.discard(0)
        return labels

    def nodes_with_label(self, label):
        """
        Gets the IDs of nodes that have the specified label value.

        Parameters
        ----------
        label : int
            Label value to search for in the "labels" attribute.

        Returns
        -------
        numpy.ndarray
            Node IDs that have the specified label.
        """
        return np.where(self.labels == label)[0]

    # --- Computation ---
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
        return distance.euclidean(self.voxels[i], self.voxels[j])

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
        xyz_i = self.voxels[i] * self.anisotropy
        xyz_j = self.voxels[j] * self.anisotropy
        return distance.euclidean(xyz_i, xyz_j)

    def get_bbox(self, nodes):
        """
        Calculates the minimal bounding box containing the voxel coordinates
        for a given collection of nodes.

        Parameters
        ----------
        nodes : Container
            Node indices for which to compute the bounding box.

        Returns
        -------
        dict
            Dictionary containing the bounding box coordinates:
            - "min": minimum voxel coordinates along each axis.
            - "max": maximum voxel coordinates along each axis.
        """
        bbox_min = np.inf * np.ones(3)
        bbox_max = np.zeros(3)
        for i in nodes:
            bbox_min = np.minimum(bbox_min, self.voxels[i])
            bbox_max = np.maximum(bbox_max, self.voxels[i] + 1)
        return {"min": bbox_min.astype(int), "max": bbox_max.astype(int)}

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

    def run_lengths(self):
        """
        Computes the path length of each connected component.

        Returns
        -------
        numpy.ndarray
            Array containing run lengths of each connected component.
        """
        run_lengths = []
        if self.number_of_nodes() > 0:
            for nodes in nx.connected_components(self):
                root = util.sample_once(nodes)
                run_lengths.append(self.run_length_from(root))
        else:
            run_lengths.append(0)
        return np.array(run_lengths)

    def run_length_from(self, root):
        """
        Computes the path length of the connected component that contains
        "root".

        Parameters
        ----------
        root : int
            Node contained in connected component to compute run length of.

        Returns
        -------
        float
            Path length.
        """
        run_length = 0
        for i, j in nx.dfs_edges(self, source=root):
            run_length += self.physical_dist(i, j)
        return run_length

    def upd_labels(self, nodes, label):
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
            self.labels[i] = label

    def to_zipped_swc(self, zip_writer):
        """
        Writes the graph to an SWC file format, which is then stored in a ZIP
        archive.

        Parameters
        ----------
        zip_writer : zipfile.ZipFile
            A ZipFile object that will store the generated SWC file.
        """
        # Subroutines
        def write_entry(node, parent):
            x, y, z = tuple(self.voxels[i] * self.anisotropy)
            r = 2 if self.is_groundtruth else 3
            node_id = cnt
            parent_id = node_to_idx[parent]
            node_to_idx[node] = node_id
            text_buffer.write(f"\n{node_id} 2 {x} {y} {z} {r} {parent_id}")

        # Main
        with StringIO() as text_buffer:
            # Preamble
            text_buffer.write(self.get_color())
            text_buffer.write("\n# id, type, z, y, x, r, pid")

            # Write entries
            cnt = 1
            node_to_idx = defaultdict(lambda: -1)
            for i, j in nx.dfs_edges(self):
                # Special Case: Root
                if len(node_to_idx) == 0:
                    write_entry(i, -1)

                # General Case: Non-Root
                cnt += 1
                write_entry(j, i)

            # Finish
            zip_writer.writestr(self.filename, text_buffer.getvalue())

    def get_color(self):
        """
        Gets the display color of the skeleton to be written to an SWC file.

        Returns
        -------
        str
            String representing the color in the format "# COLOR R G B".
        """
        if self.is_groundtruth:
            return "# COLOR 1.0 1.0 1.0"
        else:
            return util.sample_once(SkeletonGraph.colors)
