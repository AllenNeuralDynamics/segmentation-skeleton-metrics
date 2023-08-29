# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import networkx as nx
import random
import segmentation_skeleton_metrics.seg_metrics as sm
from segmentation_skeleton_metrics import nx_utils, utils


class SplitMetric(sm.SegmentationMetrics):
    """
    Class that detects splits in a predicted segmentation

    """

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
        for graph in self.graphs:
            root = nx_utils.sample_node(graph)
            dfs_edges = list(nx.dfs_edges(graph, source=root))
            while len(dfs_edges) > 0:
                (i, j) = dfs_edges.pop(0)
                label_i = self.get_label(graph, i)
                label_j = self.get_label(graph, j)
                if super().is_mistake(label_i, label_j):
                    self.site_cnt += 1
                    super().log(graph, [(i, j)])
                elif label_j == 0:
                    dfs_edges = self.mistake_search(graph, dfs_edges, i, j)
        print("# Splits:", self.site_cnt)
        print("% Omit:", self.edge_cnt / self.count_edges())
        super().write_results("splits")

    def mistake_search(self, graph, dfs_edges, nb, root):
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
        collisions = dict()
        while len(queue) > 0:
            i = queue.pop(0)
            label_i = self.get_label(graph, i)
            visited.add(i)
            if label_i != 0:
                collisions[label_i] = i
            else:
                nbs =  nx_utils.get_nbs(graph, i)
                for j in [j for j in nbs if j not in visited]:
                    if utils.check_edge(dfs_edges, (i, j)):
                        queue.append(j)
                        dfs_edges = utils.remove_edge(dfs_edges, (i, j))
                    elif j == nb:
                        queue.append(j)

        self.edge_cnt += len(visited)

        # Check for split
        recorded = list()
        if len(collisions) > 1:
            k = random.sample(list(collisions.values()), 1)[0]
            for i in collisions.values():
                if i != k:
                    self.site_cnt += 1
                    recorded.append((k, i))
            super().log(graph, list(recorded))
        
        return dfs_edges
