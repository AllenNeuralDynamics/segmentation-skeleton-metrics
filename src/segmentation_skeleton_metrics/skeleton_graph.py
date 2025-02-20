from scipy.spatial import distance

import networkx as nx
import numpy as np

from segmentation_skeleton_metrics.utils import util


class SkeletonGraph(nx.Graph):

    def __init__(self, anisotropy=(1.0, 1.0, 1.0)):
        # Call parent class
        super(SkeletonGraph, self).__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.run_length = 0

    def set_labels(self):
        self.labels = np.zeros((self.number_of_nodes()), dtype=int)

    def set_nodes(self):
        num_nodes = len(self.voxels)
        self.add_nodes_from(np.arange(num_nodes))

    def set_voxels(self, voxels):
        self.voxels = np.array(voxels, dtype=np.int32)

    # --- Getters ---
    def get_labels(self):
        return np.unique(self.labels)

    def nodes_with_label(self, label):
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
            Distance between physical coordinates of the given nodes.

        """
        xyz_i = self.voxels[i] * self.anisotropy
        xyz_j = self.voxels[j] * self.anisotropy
        return distance.euclidean(xyz_i, xyz_j)

    def get_bbox(self, nodes):
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
