"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of a custom subclass of NetworkX.Graph called SkeletonGraph.

"""

from io import StringIO
from scipy.spatial import distance

import networkx as nx
import numpy as np

from segmentation_skeleton_metrics.utils import util


class SkeletonGraph(nx.Graph):
    """
    A subclass of the NetworkX.Graph designed for graphs built from SWC files.
    This class extends the functionality of the standard Graph by adding
    support for handling node labels and voxel coordinates. In this subclass,
    node IDs serve as direct indices for accessing the labels and voxels
    attributes.

    Attributes
    ----------
    anisotropy : np.ndarray
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    run_length : float
        Physical path length of the graph.
    labels : numpy.ndarray
        A 1D array that contains a label value associated with each node.
    voxels : numpy.ndarray
        A 3D array that contains a voxel coordinate for each node.

    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0)):
        """
        Initializes a SkeletonGraph, including setting the anisotropy and
        initializing the run length attributes.

        Parameters
        ----------
        anisotropy : ArrayLike, optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is (1.0, 1.0, 1.0).

        Returns
        -------
        None

        """
        # Call parent class
        super(SkeletonGraph, self).__init__()

        # Instance attributes
        self.anisotropy = np.array(anisotropy)
        self.filename = None
        self.labels = None
        self.run_length = 0
        self.voxels = None

    def init_labels(self):
        """
        Initializes the "labels" attribute for the graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

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

        Returns
        -------
        None

        """
        self.voxels = np.array(voxels, dtype=np.int32)

    def set_filename(self, filename):
        """
        Sets the filename attribute which corresponds to the SWC file that the
        graph is built from.

        Parameters
        ----------
        filename : str
            Name of SWC file that graph is built from.

        Returns
        -------
        None

        """
        self.filename = filename

    def set_nodes(self, num_nodes):
        """
        Adds nodes to the graph. The nodes are assigned indices from 0 to
        "num_nodes".

        Parameters
        ----------
        num_nodes

        Returns
        -------
        None

        """
        self.add_nodes_from(np.arange(num_nodes))

    # --- Getters ---
    def get_labels(self):
        """
        Gets the unique non-zero label values in the "labels" attribute.

        Parameters
        ----------
        None

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
            A 1D array of node IDs that have the specified label.

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

        Returns
        -------
        None

        """
        nodes = self.nodes_with_label(label)
        self.remove_nodes_from(nodes)

    def run_lengths(self):
        """
        Computes the path length of each connected component.

        Parameters
        ----------
        None

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

        Returns
        -------
        None

        """
        for i in nodes:
            self.labels[i] = label

    def to_zipped_swc(self, zip_writer, color=None):
        """
        Writes the graph to an SWC file format, which is then stored in a ZIP
        archive.

        Parameters
        ----------
        zip_writer : zipfile.ZipFile
            A ZipFile object that will store the generated SWC file.
        color : str, optional
            A string representing the color (e.g., "[1.0 0.0 0.0]") of the SWC
            file. The default is None.

        Returns
        -------
        None

        """
        with StringIO() as text_buffer:
            # Preamble
            text_buffer.write("# COLOR " + color) if color else None
            text_buffer.write("# id, type, z, y, x, r, pid")

            # Write entries
            node_to_idx = dict()
            r = 6 if color else 3
            for i, j in nx.dfs_edges(self):
                # Special Case: Root
                x, y, z = tuple(self.voxels[i] * self.anisotropy)
                if len(node_to_idx) == 0:
                    parent = -1
                    node_to_idx[i] = 1
                    text_buffer.write("\n" + f"1 2 {x} {y} {z} {r} {parent}")

                # General Case
                node = len(node_to_idx) + 1
                parent = node_to_idx[i]
                node_to_idx[j] = node
                text_buffer.write("\n" + f"{node} 2 {x} {y} {z} {r} {parent}")

            # Finish
            zip_writer.writestr(self.filename, text_buffer.getvalue())
