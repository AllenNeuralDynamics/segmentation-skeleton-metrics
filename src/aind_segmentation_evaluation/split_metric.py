# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os
import networkx as nx
import aind_segmentation_evaluation.seg_metrics as sm
from aind_segmentation_evaluation import nx_utils, swc_utils, utils
from random import sample

class SplitMetric(sm.SegmentationMetrics):
    """
    Class that detects splits in a predicted segmentation

    """

    def detect_mistakes(self):
        """
        Detects splits in the predicted segmentation.

        Parameters
        -------
        None

        Returns
        -------
        None

        """
        for graph in self.graphs:
            root = nx_utils.sample_node(graph)
            dfs_edges = list(nx.dfs_edges(graph, source=root))
            while len(dfs_edges) > 0:
                (i, j) = dfs_edges.pop(0)
                label_i = nx_utils.get_label(self.labels, graph, i)
                label_j = nx_utils.get_label(self.labels, graph, j)
                if super().is_mistake(label_i, label_j):
                    self.site_cnt += 1
                    super().log(graph, [(i, j)], "split_site-")
                elif label_i == 0:
                    dfs_edges = self.mistake_search(graph, dfs_edges, i)
        super().write_results("splits")

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
            super().log(graph, list(recorded), "split-")
        self.edge_cnt += len(visited)
        return dfs_edges
