# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from random import sample
from scipy.spatial import distance
from tqdm import tqdm

import networkx as nx
import numpy as np
import sys

from segmentation_skeleton_metrics.utils import img_util, swc_util, util

ANISOTROPY = np.array([0.748, 0.748, 1.0])


class GraphBuilder:
    """
    A class that builds and processes graphs from SWC files.

    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        coords_only=False,
        label_mask=None,
        selected_ids=None,
        use_anisotropy=True,
    ):
        # Instance attributes
        self.anisotropy = anisotropy
        self.coords_only = coords_only
        self.label_mask = label_mask
        self.selected_ids = selected_ids

        # Reader
        anisotropy = anisotropy if use_anisotropy else (1.0, 1.0, 1.0)
        self.swc_reader = swc_util.Reader(anisotropy, selected_ids=selected_ids)

    def run(self, swc_pointer):
        graphs = self._build_graphs_from_swcs(swc_pointer)
        graphs = self._label_graphs_with_segmentation(graphs)
        return graphs

    # --- Build Graphs ---
    def _build_graphs_from_swcs(self, swc_pointer):
        with ThreadPoolExecutor() as executor:
            # Assign processes
            processes = list()
            swc_dicts = self.swc_reader.load(swc_pointer)
            while len(swc_dicts) > 0:
                swc_dict = swc_dicts.pop()
                #if self._process_swc_dict(swc_dict["swc_id"]):
                processes.append(executor.submit(self.to_graph, swc_dict))

            # Store results
            graphs = dict()
            pbar = tqdm(total=len(processes), desc="Build Graphs")
            for process in as_completed(processes):
                graphs.update(process.result())
                pbar.update(1)
        return graphs

    def _process_swc_dict(self, swc_id):
        if self.selected_ids:
            segment_id = get_segment_id(swc_id)
            return True if segment_id in self.selected_ids else False
        else:
            return True

    def to_graph(self, swc_dict):
        """
        Builds a graph from a dictionary that contains the contents of an SWC
        file.

        Parameters
        ----------
        swc_dict : dict
            ...

        Returns
        -------
        networkx.Graph
            Graph built from an SWC file.

        """
        # Initialize graph
        graph = SkeletonGraph(anisotropy=self.anisotropy)
        graph.set_voxels(swc_dict["voxel"])

        # Build graph
        if not self.coords_only:
            #graph.set_nodes()
            id_lookup = dict()
            run_length = 0
            for i, id_i in enumerate(swc_dict["id"]):
                id_lookup[id_i] = i
                if swc_dict["pid"][i] != -1:
                    parent = id_lookup[swc_dict["pid"][i]]
                    graph.add_edge(i, parent)
                    graph.run_length += graph.dist(i, parent)

            # Set graph-level attributes
            graph.graph["n_edges"] = graph.number_of_edges()
        return {swc_dict["swc_id"]: graph}

    # --- Label Graphs ---
    def _label_graphs_with_segmentation(self, graphs):
        return graphs

    def _label_graph(self, key):
        pass


class SkeletonGraph(nx.Graph):

    def __init__(self, anisotropy=(1.0, 1.0, 1.0)):
        # Call parent class
        super(SkeletonGraph, self).__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.run_length = 0

    def set_nodes(self):
        num_nodes = len(self.voxels)
        self.add_nodes_from(np.arange(num_nodes))

    def set_voxels(self, voxels):
        self.voxels = np.array(voxels, dtype=np.int32)

    # --- Getters ---
    def get_labels(self):
        return np.unique(self.graph["label"])

    def nodes_with_label(self, label):
        return np.where(self.graph["label"] == label)[0]

    # --- Computation ---
    def dist(self, i, j):
        return distance.euclidean(self.voxels[i], self.voxels[j])

    def physical_dist(self, i, j):
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

    def run_lengths(self):
        """
        Computes the path length of each connected component.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be parsed.

        Returns
        -------
        run_lengths : numpy.ndarray
            Array containing run lengths of each connected component in "graph".

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
        run_length = 0
        for i, j in nx.dfs_edges(self, source=root):
            run_length += self.physical_dist(i, j)
        return run_length


class LabelHandler:
    def __init__(self, connections_path=None, valid_labels=None):
        """
        Initializes the label handler and builds the label mappings
        if a connections path is provided.

        Parameters
        ----------
        connections_path : str, optional
            Path to a file containing pairs of segment ids that were merged.
        valid_labels : Set[int], optional
            Set of valid segment ids to be considered during processing.

        Returns
        -------
        None

        """
        self.mapping = dict()  # Maps label to equivalent class id
        self.inverse_mapping = dict()  # Maps class id to list of labels
        self.processed_labels = set()
        self.valid_labels = valid_labels or set()
        if connections_path:
            self.init_label_mappings(connections_path)

    # --- Constructor Helpers ---
    def init_label_mappings(self, connections_path):
        """
        Initializes dictionaries that map between segment IDs and equivalent
        class IDS.

        Parameters
        ----------
        path : str
            Path to a file containing pairs of segment ids that were merged.

        Returns
        -------
        None

        """
        self.mapping = {0: 0}
        self.inverse_mapping = {0: [0]}
        labels_graph = self.build_labels_graph(connections_path)
        for i, labels in enumerate(nx.connected_components(labels_graph)):
            class_id = i + 1
            self.inverse_mapping[class_id] = set()
            for label in labels:
                self.mapping[label] = class_id
                self.inverse_mapping[class_id].add(label)

    def build_labels_graph(self, connections_path):
        """
        Builds a graph from a list of labels and connection data. The nodes
        are initialized with "self.valid_labels", then edges are added between
        nodes based on a list of connections specified in a file.

        Parameters
        ----------
        connections_path : str
            Path to a text file containing connections. Each line represents a
            connection between two segmentation ids.

        Returns
        -------
        networkx.Graph
            Graph with nodes that represent labels and edges are based on the
            connections read from the "connections_path".

        """
        # Initializations
        assert self.valid_labels is not None, "Must provide valid labels!"
        labels_graph = nx.Graph()
        labels_graph.add_nodes_from(self.valid_labels)

        # Main
        for line in util.read_txt(connections_path):
            ids = line.split(",")
            id_1 = int(ids[0])
            id_2 = int(ids[1])
            labels_graph.add_edge(id_1, id_2)
        return labels_graph

    # --- Main ---
    def get(self, label):
        if self.use_mapping():
            return self.mapping.get(label, 0)
        elif self.valid_labels:
            return 0 if label not in self.valid_labels else label
        return label

    def get_class(self, label):
        return self.inverse_mapping[label] if self.use_mapping() else [label]

    def use_mapping(self):
        return True if len(self.mapping) > 0 else False


# --- Update graph ---
def remove_nodes(graph, target_label):
    """
    Deletes nodes in the given graph whose label is "target_label".

    Parameters
    ----------
    graph : networkx.Graph
        Graph with a graph-level attribute called "label".
    target_label : int
        Label to be deleted from graph.

    Returns
    -------
    networkx.Graph
        Updated graph.

    """
    nodes = np.where(graph.graph["label"] == target_label)[0]
    graph.remove_nodes_from(nodes)
    return graph


# -- Miscellaneous --
def count_splits(graph):
    """
    Counts the number of splits in "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be evaluated.

    Returns
    -------
    int
        Number of splits in "graph".

    """
    return max(nx.number_connected_components(graph) - 1, 0)


def get_segment_id(swc_id):
    return int(swc_id.split(".")[0])


def sample_leaf(graph):
    """
    Samples leaf node from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be sampled from.

    Returns
    -------
    int
        Leaf node of "graph"

    """
    leafs = [i for i in graph.nodes if graph.degree[i] == 1]
    return sample(leafs, 1)[0]
