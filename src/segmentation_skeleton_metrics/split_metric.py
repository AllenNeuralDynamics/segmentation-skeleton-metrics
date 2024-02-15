# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import networkx as nx
import random
import segmentation_skeleton_metrics.seg_metrics as sm
from segmentation_skeleton_metrics import graph_utils as gutils, utils
from toolbox.utils import progress_bar


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
        for t, graph in enumerate(self.graphs):
            # Initializations
            progress_bar(t, len(self.graphs))
            pred_graph = graph.copy()
            pred_graph.graph.update({"pred_ids": set()})

            # Run dfs
            root = gutils.sample_node(graph)
            dfs_edges = list(nx.dfs_edges(graph, source=root))
            while len(dfs_edges) > 0:
                # Visit edge
                (i, j) = dfs_edges.pop(0)
                label_i, label_j = self.get_labels(graph, i, j)
                if super().is_mistake(label_i, label_j):
                    self.site_cnt += 1
                    pred_graph = upd_pred_edge(pred_graph, i, j)
                    super().log(graph, [(i, j)])
                elif label_j == 0:
                    dfs_edges, pred_graph = self.mistake_search(
                        graph, pred_graph, dfs_edges, i, j
                    )

                # Add label_j to pred_graph
                pred_graph = upd_pred_node(pred_graph, j, label_j)
                if i == root:
                    pred_graph = upd_pred_node(pred_graph, i, label_i)

            self.pred_graphs.append(pred_graph)

        print("# Splits:", self.site_cnt)
        print("% Omit:", self.edge_cnt / self.count_edges())
        super().write_results("splits")

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
        collisions = dict()
        while len(queue) > 0:
            j = queue.pop(0)
            label_j = self.get_label(graph, j)
            pred_graph = upd_pred_node(pred_graph, j, label_j)
            visited.add(j)
            if label_j != 0:
                collisions[label_j] = j
            else:
                for k in [k for k in graph.neighbors(j) if k not in visited]:
                    if utils.check_edge(dfs_edges, (j, k)):
                        queue.append(k)
                        dfs_edges = utils.remove_edge(dfs_edges, (j, k))
                    elif k == nb:
                        queue.append(k)

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

        return dfs_edges, pred_graph


# -- utils --
def upd_pred_edge(pred_graph, i, j):
    pred_graph.remove_edges_from([(i, j)])
    return pred_graph


def upd_pred_node(pred_graph, i, label):
    pred_graph.graph["pred_ids"].update(set([label]))
    pred_graph.nodes[i].update({"pred_id": label})
    return pred_graph
