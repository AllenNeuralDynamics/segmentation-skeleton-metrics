# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import networkx as nx
import numpy as np
from more_itertools import zip_broadcast

import aind_segmentation_evaluation.seg_metrics as sm
from aind_segmentation_evaluation import nx_utils, swc_utils, utils


class MergeMetric(sm.SegmentationMetrics):
    """
    Class that evaluates the quality of a segmentation in terms of the
    number of merges.
    """
    def detect_mistakes(self):
        """
        Detects merges in the predicted segmentation.

        Parameters
        -------
        None

        Returns
        -------
        None

        """
        self.merged_edges = set()
        self.run_lengths = dict(zip_broadcast(self.get_labels(), []))
        for graph in self.graphs:
            target_label = -1
            dfs_edges = list(nx.dfs_edges(graph))
            while len(dfs_edges) > 0:
                (i, j) = dfs_edges.pop(0)
                label_i = nx_utils.get_label(self.labels, graph, i)
                label_j = nx_utils.get_label(self.labels, graph, j)
                if super().is_mistake(label_i, label_j):
                    self.site_cnt += 1
                    super().log(graph, (i, j), "merge_site-")
                    dfs_edges = self.explore_merge(graph, dfs_edges, i, label_i)
                    dfs_edges = self.explore_merge(graph, dfs_edges, j, label_j)
                elif label_i == 0:
                    dfs_edges, collisions = self.mistake_search(graph, dfs_edges, i)

        # Save Results
        super().write_results("merges")
        self.edge_cnt = len(self.merged_edges) // 2

    def mistake_search(self, graph, dfs_edges, root):
        """
        Determines whether complex mistake is a split.

        Parameters
        ----------
        graph : networkx.Graph
            Graph with potential split.
        dfs_edges : list[tuple]
            List of edges in order wrt a depth first search.
        root : int
            Edge where possible split starts.

        Returns
        -------
        list[tuple].
            Updated "dfs_edges".

        """
        # Search
        queue = [root]
        collisions = dict()
        visited = set()
        while len(queue) > 0:
            i = queue.pop(0)
            for j in nx_utils.get_nbs(graph, i):
                label_j = nx_utils.get_label(self.labels, graph, j)
                if frozenset([i, j]) in visited:
                    continue
                elif label_j != 0:
                    if label_j not in collisions.keys():
                        collisions[label_j] = j
                else:
                    queue.append(j)
                visited.add(frozenset([i, j]))
                dfs_edges = utils.remove_edge(dfs_edges, (i, j))

        # Check for split
        recorded = set()
        if len(collisions) > 1:
            for i in collisions.values():
                label_i = nx_utils.get_label(self.labels, graph, i)
                dfs_edges = self.explore_merge(graph, dfs_edges, i, label_i)
                for j in collisions.values():
                    if i != j and frozenset((i, j)) not in recorded:
                        self.site_cnt += 1
                        recorded.add(frozenset((i, j)))
                        super().log(graph, (i, j), "merge_site-")
        return dfs_edges, collisions

    def explore_merge(self, graph, dfs_edges, root, val):
        """
        Traverses "graph" from "root" to determine how many and which
        edges are merged.

        Parameters
        ----------
        graph : networkx.Graph
            Graph that represents a neuron.
        dfs_edges : list[tuple]
            List of edges in graph ordered wrt a depth first search.
        root : int
            Root node of list of edges in "dfs_edges".
        val : int
            Value of root node.

        Returns
        -------
        list[tuple]
            List of merged edges.

        """
        cur_merge = list()
        visited = set()
        queue = [(-1, root)]  # parent, child
        while len(queue) > 0:
            # Visit
            i, j = queue.pop(0)
            label_i = nx_utils.get_label(self.labels, graph, j)
            if label_i == val and i != -1:
                self.merged_edges.update(frozenset((i, j)))
                cur_merge.append((i, j))
                dfs_edges = utils.remove_edge(dfs_edges, (i, j))
            visited.add(j)

            # Populate queue
            for k in nx_utils.get_nbs(graph, j):
                condition1 = k not in visited
                condition2 = frozenset((j, k)) not in self.merged_edges
                if condition1 and condition2:
                    queue.append((j, k))

        return dfs_edges
