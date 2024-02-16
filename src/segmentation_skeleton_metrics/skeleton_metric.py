# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
import networkx as nx
import numpy as np
import random
import tensorstore as ts

from segmentation_skeleton_metrics import graph_utils as gutils, utils
from segmentation_skeleton_metrics.swc_utils import to_graph
from toolbox.utils import progress_bar

SUPPORTED_FILETYPES = ["tif", "n5", "neuroglancer_precomputed"]


class SkeletonMetric:
    """
    Class that evaluates a segmentation in terms of the number of
    splits and merges.

    """

    def __init__(
        self,
        swc_paths,
        labels,
        anisotropy=[1.0, 1.0, 1.0],
        filetype=None,
        valid_ids=None,
        equivalent_ids=None,
    ):
        """
        Constructs object that evaluates a predicted segmentation.

        Parameters
        ----------
        ...

        Returns
        -------
        None.

        """
        self.valid_ids = valid_ids
        self.graphs = self.init_graphs(swc_paths, anisotropy)
        self.pred_graphs = []
        if type(labels) is str:
            self.labels = self.init_labels(labels, filetype)
        else:
            self.labels = labels

    def init_graphs(self, paths, anisotropy):
        graphs = []
        for path in paths:            
            graphs.append(to_graph(path, anisotropy=anisotropy))
        return graphs

    def init_labels(self, path, filetype):
        """
        Initializes a volume by uploading file with extension "filetype".

        Parameters
        ----------
        path : str
            Path to image volume.
        filetype : str
            Extension of file to be uploaded, supported values include tif, n5,
            and tensorstore.

        Returns
        -------
        dict
            Image volume.

        """
        assert filetype is not None, "Must provide filetype to upload image!"
        assert filetype in SUPPORTED_FILETYPES, "Filetype is not supported!"
        return utils.read_img(path, filetype)

    def get_labels(self):
        """
        Gets list of all unique labels in "self.labels".

        Parameters
        ----------
        None

        Returns
        -------
        list[int]
            List of all unique labels in "self.labels".

        """
        if type(self.labels) == np.array:
            labels = np.unique(self.labels)
        else:
            labels = np.unique(list(self.labels.keys()))
        return [label for label in labels if label != 0]

    def get_label(self, graph, i):
        """
        Gets label of node "i".

        Parameters
        ----------
        graph : networkx.Graph
            Graph which represents a neuron.
        i : int
            Node in "graph".

        Returns
        -------
        int
           Label of node "i".

        """
        label = self.__get_label(gutils.get_xyz(graph, i))
        return self.adjust_label(label)

    def __get_label(self, xyz):
        """
        Gets label at image coordinates "xyz".

        Parameters
        ----------
        xyz : tuple[int]
            Coordinates that index into "self.labels".

        Returns
        -------
        int
           Label at image coordinates "xyz".

        """
        if type(self.labels) == dict:
            return 0 if xyz not in self.labels.keys() else self.labels[xyz]
        elif type(self.labels) == ts.TensorStore:
            return int(self.labels[xyz].read().result())
        else:
            return self.labels[xyz]

    def get_labels(self, graph, i, j):
        """
        Gets labels of nodes "i" and "j".

        Parameters
        ----------
        graph : networkx.Graph
            Graph which represents a neuron.
        i : int
            Node in "graph".
        j : int
            Node in "graph".

        Returns
        -------
        int
           Label of node "i".
        int
           Label of node "j".

        """
        return self.get_label(graph, i), self.get_label(graph, j)

    def adjust_label(self, label):
        if self.valid_ids:
            if label not in self.valid_ids:
                return 0
        return label

    def set_labels(self, label_i, label_j):
        return self.set_label(label_i), self.set_label(label_j)

    def count_edges(self):
        """
        Counts number of edges in all graphs in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        int

        """
        total_edges = 0
        for graph in self.graphs:
            total_edges += graph.number_of_edges()
        return total_edges

    def is_mistake(self, a, b):
        """
        Checks if "a" and "b" are positive and not equal.

        Parameters
        ----------
        a : int
            label at node i.
        b : int
            label at node j.

        Returns
        -------
        bool
            Indicates whether there is a mistake.

        """
        if self.valid_ids is not None:
            if a not in self.valid_ids or b not in self.valid_ids:
                return False
        return (a != 0 and b != 0) and (a != b)

    def detect_mistakes(self):
        """
        Detects splits in the predicted segmentation.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for t, graph in enumerate(self.graphs):
            # Initializations
            progress_bar(t, len(self.graphs))
            pred_graph = graph.copy()
            pred_graph.graph.update({"pred_ids": set()})

            # Sample root
            r = gutils.sample_leaf(graph)
            label_r = self.get_label(graph, r)
            pred_graph = upd_node(pred_graph, r, label_r)

            # Run dfs
            dfs_edges = list(nx.dfs_edges(graph, source=r))
            while len(dfs_edges) > 0:
                # Visit edge
                (i, j) = dfs_edges.pop(0)
                label_i, label_j = self.get_labels(graph, i, j)
                if self.is_mistake(label_i, label_j):
                    pred_graph = upd_edge(pred_graph, i, j, label_j)
                elif label_j == 0:
                    dfs_edges, pred_graph = self.mistake_search(
                        graph, pred_graph, dfs_edges, i, j
                    )
                else:
                    pred_graph = upd_node(pred_graph, j, label_j)

            pred_graph = remove_zeros(pred_graph)
            self.pred_graphs.append(pred_graph)
            self.site_cnt += count_splits(pred_graph)

        print("# Splits:", self.site_cnt)
        print("% Omit:", self.edge_cnt / self.count_edges())

    def mistake_search(self, graph, pred_graph, dfs_edges, nb, root):
        """
        Determines whether complex mistake is a split.

        Parameters
        ----------
        graph : networkx.Graph
            Graph with possible split at "root".
        dfs_edges : list[tuple]
            List of edges to be processed for mistake detection.
        root : int
            Node where possible split starts.

        Returns
        -------
        list[tuple].
            Updated "dfs_edges" with visited edges removed.

        """
        # Search
        queue = [root]
        visited = set()
        collision_labels = set()
        collision_nodes = set()
        while len(queue) > 0:
            j = queue.pop(0)
            label_j = self.get_label(graph, j)
            visited.add(j)
            if label_j != 0:
                collision_labels.add(label_j)
                pred_graph = upd_node(pred_graph, j, label_j)
            else:
                for k in [k for k in graph.neighbors(j) if k not in visited]:
                    if utils.check_edge(dfs_edges, (j, k)):
                        queue.append(k)
                        dfs_edges = utils.remove_edge(dfs_edges, (j, k))
                    elif k == nb:
                        queue.append(k)

        # Upd zero nodes
        visited = visited.difference(collision_nodes)
        label = collision_labels.pop() if len(collision_labels) == 1 else 0
        pred_graph = upd_nodes(pred_graph, visited, label)

        return dfs_edges, pred_graph

    
# -- utils --
def upd_edge(pred_graph, i, j, label_j):
    pred_graph.remove_edges_from([(i, j)])
    pred_graph = upd_node(pred_graph, j, label_j)
    return pred_graph


def upd_node(pred_graph, i, label):
    pred_graph.graph["pred_ids"].update(set([label]))
    pred_graph.nodes[i].update({"pred_id": label})
    return pred_graph


def upd_nodes(pred_graph, nodes, label):
    for i in nodes:
        pred_graph = upd_node(pred_graph, i, label)
    return pred_graph


def remove_zeros(pred_graph):
    delete_nodes = []
    for i in pred_graph.nodes:
        if pred_graph.nodes[i]["pred_id"] == 0:
            delete_nodes.append(i)
    pred_graph.remove_nodes_from(delete_nodes)
    return pred_graph

def count_splits(graph):
    return max(len(list(nx.connected_components(graph))) - 1, 0)
