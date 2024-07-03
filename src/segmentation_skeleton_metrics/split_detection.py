"""
Created on Wed March 11 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Detects splits in a predicted segmentation by comparing the ground truth
skeletons (i.e. target_graphs) to the predicted segmentation label mask.

"""

import networkx as nx

from segmentation_skeleton_metrics import graph_utils as gutils
from segmentation_skeleton_metrics import utils


def run(target_graph, labeled_graph):
    """
    Detected splits in a predicted segmentation.

    Parameters
    ----------
    target_graph : networkx.Graph
        Graph built from a ground truth swc file.
    labeled_graph : networkx.Graph
        Labeled graph built from a ground truth swc file, where each node has
        an attribute called 'label'.

    Returns
    -------
    labeled_graph : networkx.Graph
        Labeled graph with omit and split edges removed.

    """
    r = gutils.sample_leaf(target_graph)
    dfs_edges = list(nx.dfs_edges(target_graph, source=r))
    while len(dfs_edges) > 0:
        # Visit edge
        (i, j) = dfs_edges.pop(0)
        label_i = labeled_graph.nodes[i]["label"]
        label_j = labeled_graph.nodes[j]["label"]
        if is_split(label_i, label_j):
            dfs_edges, labeled_graph = is_nonzero_misalignment(
                target_graph, labeled_graph, dfs_edges, i, j
            )
        elif label_j == 0:
            dfs_edges, labeled_graph = is_zero_misalignment(
                target_graph, labeled_graph, dfs_edges, i, j
            )
    return labeled_graph


def is_zero_misalignment(target_graph, labeled_graph, dfs_edges, nb, root):
    """
    Determines whether zero-valued labels correspond to a split or
    misalignment between "target_graph" and the predicted segmentation
    mask.

    Parameters
    ----------
    target_graph : networkx.Graph
        Graph built from a ground truth swc file.
    labeled_graph : networkx.Graph
        Labeled graph built from a ground truth swc file, where each node has
        an attribute called 'label'.
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
        labeled_graph : networkx.Graph
            Ground truth graph with nodes labeled with respect to
            corresponding voxel in predicted segmentation.

    """
    # Search
    collision_labels = set()
    queue = [root]
    visited = set()
    while len(queue) > 0:
        j = queue.pop(0)
        label_j = labeled_graph.nodes[j]["label"]
        visited.add(j)
        if label_j > 0:
            collision_labels.add(label_j)
        else:
            # Add nbs to queue
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
        labeled_graph = gutils.upd_labels(labeled_graph, visited, label)

    return dfs_edges, labeled_graph


def is_nonzero_misalignment(target_graph, labeled_graph, dfs_edges, nb, root):
    """
    Determines whether nonzero-valued labels correspond to a split or
    misalignment between "target_graph" and the predicted segmentation
    mask.

    Parameters
    ----------
    target_graph : networkx.Graph
        Graph built from a ground truth swc file.
    labeled_graph : networkx.Graph
        Labeled graph built from a ground truth swc file, where each node has
        an attribute called 'label'.
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
        labeled_graph : networkx.Graph
            Ground truth graph with nodes labeled with respect to
            corresponding voxel in predicted segmentation.

    """
    # Initialize
    origin_label = labeled_graph.nodes[nb]["label"]
    hit_label = labeled_graph.nodes[root]["label"]

    # Search
    queue = [(nb, root)]
    visited = set([nb])
    while len(queue) > 0:
        parent, j = queue.pop(0)
        label_j = labeled_graph.nodes[j]["label"]
        visited.add(j)
        if label_j == origin_label and len(queue) == 0:
            # misalignment
            labeled_graph = gutils.upd_labels(
                labeled_graph, visited, origin_label
            )
            return dfs_edges, labeled_graph
        elif label_j == hit_label:
            # continue search
            nbs = list(target_graph.neighbors(j))
            for k in [k for k in nbs if k not in visited]:
                queue.append((j, k))
                dfs_edges = remove_edge(dfs_edges, (j, k))
        else:
            # left hit label
            dfs_edges.insert(0, (parent, j))
            labeled_graph = gutils.remove_edge(labeled_graph, nb, root)
            return dfs_edges, labeled_graph

    # End of search
    labeled_graph = gutils.remove_edge(labeled_graph, nb, root)
    return dfs_edges, labeled_graph


# -- utils --
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
    return (a > 0 and b > 0) and (a != b)
