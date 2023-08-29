# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from random import sample

import networkx as nx

import segmentation_skeleton_metrics.seg_metrics as sm
from segmentation_skeleton_metrics import nx_utils, utils


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
        for graph in self.graphs:
            dfs_edges = list(nx.dfs_edges(graph))
            while len(dfs_edges) > 0:
                (i, j) = dfs_edges.pop(0)
                label_i = self.get_label(graph, i)
                label_j = self.get_label(graph, j)
                if super().is_mistake(label_i, label_j):
                    self.site_cnt += 1
                    super().log(graph, [(i, j)])
                    dfs_edges = self.explore_merge(
                        graph, dfs_edges, i, label_i
                    )
                    dfs_edges = self.explore_merge(
                        graph, dfs_edges, j, label_j
                    )
                elif label_i == 0:
                    dfs_edges = self.mistake_search(graph, dfs_edges, i)

        # Save Results
        super().write_results("merges")
        self.edge_cnt = len(self.merged_edges)

    def mistake_search(self, graph, dfs_edges, root):
        """
        Determines whether void region has a mistake.

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
                label_j = self.get_label(graph, j)
                if frozenset([i, j]) in visited:
                    continue
                elif label_j != 0 and label_j not in collisions.keys():
                    collisions[label_j] = j
                else:
                    queue.append(j)
                visited.add(frozenset([i, j]))
                dfs_edges = utils.remove_edge(dfs_edges, (i, j))

        # Check for split
        recorded = list()
        if len(collisions) > 1:
            root = sample(list(collisions.values()), 1)[0]
            for i in collisions.values():
                if i != root:
                    self.site_cnt += 1
                    recorded.append((root, i))

        if len(collisions) > 1:
            super().log(graph, recorded)
        return dfs_edges

    def explore_merge(self, graph, dfs_edges, root, root_label):
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
        visited = set()
        queue = [(-1, root)]  # parent, child
        while len(queue) > 0:
            # Visit
            i, j = queue.pop(0)
            label_j = self.get_label(graph, j)
            if label_j == root_label and i != -1:
                self.merged_edges.update(frozenset((i, j)))
                dfs_edges = utils.remove_edge(dfs_edges, (i, j))
            visited.add(j)

            # Populate queue
            for k in nx_utils.get_nbs(graph, j):
                condition1 = k not in visited
                condition2 = frozenset((j, k)) not in self.merged_edges
                if condition1 and condition2:
                    queue.append((j, k))

        return dfs_edges
