"""
Created on Wed March 11 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Detects splits in a predicted segmentation by comparing the ground truth
skeletons (i.e. graphs) to the predicted segmentation label mask.

"""

from collections import deque

import networkx as nx


def run(process_id, graph):
    """
    Adjusts misalignments between ground truth graph and segmentation mask.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a ground truth neuron.

    Returns
    -------
    graph : networkx.Graph
        Labeled graph with omit and split edges removed.

    """
    # Initializations
    split_cnt = 0
    source = get_leaf(graph)
    dfs_edges = deque(list(nx.dfs_edges(graph, source=source)))
    visited_edges = set()

    # Main
    while len(dfs_edges) > 0:
        # Check whether to visit edge
        i, j = dfs_edges.popleft()
        if frozenset({i, j}) in visited_edges:
            continue

        # Visit edge
        label_i = int(graph.labels[i])
        label_j = int(graph.labels[j])
        if is_split(label_i, label_j):
            graph.remove_edge(i, j)
            split_cnt += 1
        elif label_j == 0:
            check_misalignment(graph, visited_edges, i, j)
        visited_edges.add(frozenset({i, j}))

    # Finish
    split_percent = split_cnt / graph.graph["n_edges"]
    graph.remove_nodes_with_label(0)
    return process_id, graph, split_percent


def check_misalignment(graph, visited_edges, nb, root):
    """
    Determines whether zero-valued label correspond to a split or misalignment
    between the graph and segmentation mask.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that represents a ground truth neuron.
    visited_edges : list[tuple]
        List of edges in "graph" that have been visited.
    nb : int
        Neighbor of "root".
    root : int
        Node where possible split starts (i.e. zero-valued label).

    Returns
    -------
    dfs_edges : list[tuple].
        Updated "dfs_edges" with visited edges removed.
    graph : networkx.Graph
        Ground truth graph with nodes labeled with respect to corresponding
        voxel in predicted segmentation.

    """
    # Search
    label_collisions = set()
    queue = deque([root])
    visited = set()
    while len(queue) > 0:
        # Visit node
        j = queue.popleft()
        label_j = int(graph.labels[j])
        if label_j != 0:
            label_collisions.add(label_j)
        visited.add(j)

        # Update queue
        if label_j == 0:
            for k in graph.neighbors(j):
                if k not in visited:
                    if frozenset({j, k}) not in visited_edges or k == nb:
                        queue.append(k)
                        visited_edges.add(frozenset({j, k}))

    # Upd zero nodes
    if len(label_collisions) == 1:
        label = label_collisions.pop()
        graph.upd_labels(visited, label)


# -- Helpers --
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


def get_leaf(graph):
    """
    Gets a leaf node from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be sampled from.

    Returns
    -------
    int
        Leaf node of "graph"

    """
    for i in graph.nodes:
        if graph.degree[i] == 1:
            return i
