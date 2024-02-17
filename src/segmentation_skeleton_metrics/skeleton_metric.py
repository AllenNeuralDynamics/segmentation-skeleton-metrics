# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import networkx as nx
import numpy as np
import tensorstore as ts
from toolbox.utils import progress_bar

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import utils
from segmentation_skeleton_metrics.swc_utils import to_graph


class SkeletonMetric:
    """
    Class that evaluates the quality of a predicted segmentation by comparing
    the ground truth skeletons to the predicted segmentation mask. The
    accuracy is then quantified by detecting splits and merges, then computing
    the following metrics:
        (1) Number of splits
        (2) Number of merges
        (3) Percentage of omit edges
        (4) Percentage of merged edges
        (5) Edge accuracy
        (6) Expected Run Length (ERL)

    """

    def __init__(
        self,
        swc_paths,
        labels,
        anisotropy=[1.0, 1.0, 1.0],
        equivalent_ids=None,
        valid_ids=None,
    ):
        """
        Constructs skeleton metric object that evaluates the quality of a
        predicted segmentation.

        Parameters
        ----------
        swc_paths : list[str]
            List of paths to swc files such that each file corresponds to a
            neuron in the ground truth.
        labels : numpy.ndarray or tensorstore.TensorStore
            Predicted segmentation mask.
        anisotropy : list[float], optional
            Image to real-world coordinates scaling factors applied to swc
            files. The default is [1.0, 1.0, 1.0]
        equivalent_ids : ...
            ...
        valid_ids : set
            ...

        Returns
        -------
        None.

        """
        self.valid_ids = valid_ids
        self.labels = labels
        self.target_graphs = self.init_target_graphs(swc_paths, anisotropy)
        self.pred_graphs = self.init_pred_graphs()

    def init_target_graphs(self, paths, anisotropy):
        target_graphs = dict()
        for path in paths:
            swc_id = os.path.basename(path).replace(".swc", "")
            target_graphs[swc_id] = to_graph(path, anisotropy=anisotropy)
        return target_graphs

    def init_pred_graphs(self):
        print("Labelling Target Graphs...")
        t0 = time()
        pred_graphs = dict()
        for cnt, (swc_id, graph) in enumerate(self.target_graphs.items()):
            progress_bar(cnt + 1, len(self.target_graphs))
            pred_graphs[swc_id] = self.label_graph(graph)
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")
        return pred_graphs

    def label_graph(self, target_graph):
        """
        Iterates over nodes in "target_graph" and stores the label in the
        predicted segmentation mask (i.e. "self.labels") which coincides with
        each node as a node-level attribute called "pred_id".
 
        Parameters
        ----------
        target_graph : networkx.Graph
            Graph that represents a neuron from the ground truth.

        Returns
        -------
        target_graph : networkx.Graph
            Updated graph with node-level attributes called "pred_id".

        """
        pred_graph = gutils.empty_copy(target_graph)
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = []
            for i in pred_graph.nodes:
                img_coord = gutils.get_coord(target_graph, i)
                threads.append(executor.submit(self.get_label, img_coord, i))
            # Store results
            for thread in as_completed(threads):
                i, label = thread.result()
                pred_graph.nodes[i].update({"pred_id": label})
        return pred_graph

    def get_label(self, img_coord, return_node=False):
        """
        Gets label of voxel at "img_coord".

        Parameters
        ----------
        img_coord : numpy.ndarray
            Image coordinate of voxel to be read.

        Returns
        -------
        int
           Label of voxel at "img_coord".

        """
        label = self.__read_label(img_coord)
        if return_node:
            return return_node, self.validate_label(label)
        else:
            return self.validate_label(label)

    def __read_label(self, coord):
        """
        Gets label at image coordinates "xyz".

        Parameters
        ----------
        coord : tuple[int]
            Coordinates that index into "self.labels".

        Returns
        -------
        int
           Label at image coordinates "xyz".

        """
        if type(self.labels) == ts.TensorStore:
            return int(self.labels[coord].read().result())
        else:
            return self.labels[coord]

    def validate_label(self, label):
        """
        Validates label by checking whether it is contained in
        "self.valid_ids".

        Parameters
        ----------
        label : int
            Label to be validated.

        Returns
        -------
        label : int
            There are two possibilities: (1) original label if either "label"
            is contained in "self.valid_ids" or "self.valid_labels" is None,
            or (2) 0 if "label" is not contained in self.valid_ids.

        """
        if self.valid_ids:
            if label not in self.valid_ids:
                return 0
        return label

    def compute_metrics(self):
        """
        Computes skeleton-based metrics.

        Parameters
        ----------
        None

        Returns
        -------
        ...

        """
        # Split evaluation
        print("Detecting Splits...")
        self.detect_splits()
        self.quantify_splits()

        # Merge evaluation
        print("Detecting Merges...")
        self.detect_merges()
        self.quantify_merges()

        # Compute metrics
        self.compile_results()

    def detect_splits(self):
        """
        Detects splits in the predicted segmentation, then deletes node and
        edges in "self.pred_graphs" that correspond to a split.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        t0 = time()
        target_graphs = self.target_graphs.items()
        for cnt, (swc_id, target_graph) in enumerate(target_graphs):
            # Initializations
            progress_bar(cnt + 1, len(self.target_graphs))
            pred_graph = self.pred_graphs[swc_id]

            # Run dfs
            r = gutils.sample_leaf(target_graph)
            dfs_edges = list(nx.dfs_edges(target_graph, source=r))
            while len(dfs_edges) > 0:
                # Visit edge
                (i, j) = dfs_edges.pop(0)
                label_i = pred_graph.nodes[i]["pred_id"]
                label_j = pred_graph.nodes[j]["pred_id"]
                if is_split(label_i, label_j):
                    pred_graph = gutils.remove_edge(pred_graph, i, j)
                elif label_j == 0:
                    dfs_edges, pred_graph = self.split_search(
                        target_graph, pred_graph, dfs_edges, i, j
                    )

            # Update predicted graph
            pred_graph = gutils.delete_nodes(pred_graph, 0)
            pred_graph = gutils.store_labels(pred_graph)
            self.pred_graphs[swc_id] = pred_graph

        # Report runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

    def split_search(self, target_graph, pred_graph, dfs_edges, nb, root):
        """
        Determines whether zero-valued labels correspond to a split or
        misalignment between "target_graph" and the predicted segmentation
        mask.

        Parameters
        ----------
        target_graph : networkx.Graph
            ...
        pred_graph : networkx.Graph
            ...
        dfs_edges : list[tuple]
            List of edges to be processed for split detection.
        nb : int
            Neighbor of "root".
        root : int
            Node where possible split starts (i.e. zero-valued label).

        Returns
        -------
        dfs_edges : list[tuple].
            Updated "dfs_edges" with visited edges removed.
        pred_graph : networkx.Graph
            ...

        """
        # Search
        queue = [root]
        visited = set()
        collision_labels = set()
        collision_nodes = set()
        while len(queue) > 0:
            j = queue.pop(0)
            label_j = pred_graph.nodes[j]["pred_id"]
            visited.add(j)
            if label_j != 0:
                collision_labels.add(label_j)
            else:
                nbs = target_graph.neighbors(j)
                for k in [k for k in nbs if k not in visited]:
                    if utils.check_edge(dfs_edges, (j, k)):
                        queue.append(k)
                        dfs_edges = remove_edge(dfs_edges, (j, k))
                    elif k == nb:
                        queue.append(k)

        # Upd zero nodes
        if len(collision_labels) == 1:
            label = collision_labels.pop()
            visited = visited.difference(collision_nodes)
            pred_graph = gutils.upd_labels(pred_graph, visited, label)

        return dfs_edges, pred_graph

    def quantify_splits(self):
        """
        Counts the number of splits, number of omit edges, and percent of omit
        edges for each graph in "self.pred_graphs".

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.split_cnts = dict()
        self.omit_cnts = dict()
        self.omit_percents = dict()
        for swc_id in self.target_graphs.keys():
            n_splits = gutils.count_splits(self.pred_graphs[swc_id])
            n_pred_edges = self.pred_graphs[swc_id].number_of_edges()
            n_target_edges = self.target_graphs[swc_id].number_of_edges()

            self.split_cnts[swc_id] = n_splits
            self.omit_cnts[swc_id] = n_target_edges - n_pred_edges
            self.omit_percents[swc_id] = 1 - n_pred_edges / n_target_edges

    def detect_merges(self):
        """
        Detects merges in the predicted segmentation, then deletes node and
        edges in "self.pred_graphs" that correspond to a merge.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initilize counts
        self.merge_cnts = self.init_merge_counter()
        self.merged_cnts = self.init_merge_counter()
        self.merged_percents = self.init_merge_counter()

        # Run detection
        t0 = time()
        for cnt, swc_id_1 in enumerate(self.pred_graphs.keys()):
            progress_bar(cnt + 1, len(self.target_graphs))
            for swc_id_2 in self.pred_graphs.keys():
                # Check if identical
                if swc_id_1 == swc_id_2:
                    continue

                # Compare pred_ids contained in graph
                pred_ids_1 = self.pred_graphs[swc_id_1].graph["pred_ids"]
                pred_ids_2 = self.pred_graphs[swc_id_2].graph["pred_ids"]
                intersection = pred_ids_1.intersection(pred_ids_2)
                if len(intersection) > 0:
                    for label in intersection:
                        self.process_merge(swc_id_1, label)
                        self.process_merge(swc_id_2, label)

        # Report Runtime
        t, unit = utils.time_writer(time() - t0)
        print(f"\nRuntime: {round(t, 2)} {unit}\n")

        print("# merges:", np.sum(list(self.merge_cnts.values())) // 2)

    def init_merge_counter(self):
        return dict([(swc_id, 0) for swc_id in self.pred_graphs.keys()])

    def process_merge(self, swc_id, label):
        # Update graph
        graph = self.pred_graphs[swc_id]
        graph, merged_cnt = gutils.delete_nodes(graph, label, return_cnt=True)
        graph.graph["pred_ids"].remove(label)
        self.pred_graphs[swc_id] = graph

        # Update cnts
        self.merge_cnts[swc_id] += 1
        self.merged_cnts[swc_id] += merged_cnt

    def quantify_merges(self):
        self.merged_percent = dict()
        for swc_id in self.target_graphs.keys():
            n_edges = self.target_graphs[swc_id].number_of_edges()
            self.merged_percent[swc_id] = self.merged_cnts[swc_id] / n_edges

    def compile_results(self):
        pass


# -- utils --
def is_split(a, b):
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
        Indication of whether there is a split.

    """
    return (a != 0 and b != 0) and (a != b)


def remove_edge(dfs_edges, edge):
    """
    Checks whether "edge" is in "dfs_edges" and removes it.

    Parameters
    ----------
    dfs_edges : list or set
        List or set of edges.
    edge : tuple
        Edge.

    Returns
    -------
    edges : list or set
        Updated list or set of edges with "dfs_edges" removed if it was
        contained in "dfs_edges".

    """
    if edge in dfs_edges:
        dfs_edges.remove(edge)
    elif (edge[1], edge[0]) in dfs_edges:
        dfs_edges.remove((edge[1], edge[0]))
    return dfs_edges
