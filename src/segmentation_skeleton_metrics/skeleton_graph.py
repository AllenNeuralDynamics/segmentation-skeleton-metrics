"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of a custom subclass of NetworkX.Graph called SkeletonGraph.

"""

from scipy.spatial import distance

import networkx as nx
import numpy as np

from segmentation_skeleton_metrics.utils import util


class SkeletonGraph(nx.Graph):
    """
    A subclass of the NetworkX.Graph that represents a skeleton graph with
    additional functionality for handling labels and voxel coordinates
    corresponding to the nodes. Note that node IDs directly index into the
    "labels" and "voxels" attributes.

    Attributes
    ----------
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    run_length : float
        Physical path length of the graph.
    labels : numpy.ndarray
        A 1D array that contains a label value associated with each node.
    voxels : numpy.ndarray
        A 3D array that contains a voxel coordinate of each node.

    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0)):
        """
        Initializes a SkeletonGraph, including setting the anisotropy and
        initializing the run length attribute.

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
        self.run_length = 0

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
        self.labels = np.zeros((self.number_of_nodes()), dtype=int)

    def init_voxels(self, voxels):
        """
        Initializes the "voxels" attribute for the graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.voxels = np.array(voxels, dtype=np.int32)

    def set_nodes(self):
        """
        Adds nodes to the graph. The nodes are assigned indices from 0 to the
        total number of voxels in the image.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        num_nodes = len(self.voxels)
        self.add_nodes_from(np.arange(num_nodes))

    # --- Getters ---
    def get_labels(self):
        """
        Gets the unique label values in the "labels" attribute.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            A 1D array of unique labels assigned to nodes in the graph.

        """
        return np.unique(self.labels)

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
            Euclidea distance between physical coordinates of the given nodes.

        """
        xyz_i = self.voxels[i] * self.anisotropy
        xyz_j = self.voxels[j] * self.anisotropy
        return distance.euclidean(xyz_i, xyz_j)

    def get_bbox(self, nodes):
        """
        Calculates the bounding box that contains the voxel coordinates for a
        given collection of nodes.

        Parameters
        ----------
        nodes : Container
            A collection of node indices for which to compute the bounding box.

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
        graph : networkx.Graph
            Graph to be parsed.

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
        Updates the label of each node in "nodes" with "label".

        Parameters
        ----------
        nodes : List[int]
            Nodes to be updated.
        label : int
            Updated label.

        Returns
        -------
        None

        """
        for i in nodes:
            self.labels[i] = label
