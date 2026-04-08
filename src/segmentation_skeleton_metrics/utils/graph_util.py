"""
Created on Wed April 8 10:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for graph helper routines.

"""

from collections import deque

from segmentation_skeleton_metrics.utils import util


def combine_graphs(graphs, label_handler):
    """
    Combines graphs with the same label.

    Parameters
    ----------
    graph : Dict[str, FragmentGraph]
        Graphs to be updated.

    Returns
    -------
    new_graphs : Dict[str, FragmentGraph]
        Updated graphs.
    """
    new_graphs = dict()
    for key, graph in graphs.items():
        label = util.get_segment_id(key)
        class_id = label_handler.get(label)
        if class_id not in new_graphs:
            new_graphs[class_id] = graph
        else:
            new_graphs[class_id].add_graph(graph, set_kdtree=False)
    set_kdtrees(graphs)
    return new_graphs


def compute_segmented_run_length(graph, results, name):
    """
    Computes the run length of a graph that was segmented.

    Parameters
    ----------
    graph : LabeledGraph
        Graph to be evaluated.
    results : pandas.DataFrame
        Data frame containing skeleton metrics

    Returns
    -------
    float
        Run length of a graph that was segmented.
    """
    omit_rl = graph.run_length * results["% Omit Edges"][name] / 100
    split_rl = graph.run_length * results["% Split Edges"][name] / 100
    return graph.run_length - omit_rl - split_rl


def search_branching_node(graph, kdtree, root, radius=40):
    """
    Searches for a branching node within distance "radius" from the given
    root node.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    kdtree : scipy.spatial.KDTree
        KDTree containing physical coordinates from a ground truth tracing.
    root : int
        Root of search.
    radius : float, optional
        Distance to search from root. Default is 40.

    Returns
    -------
    root : int
        Root node or closest branching node within distance "radius".
    """
    queue = deque([(root, 0)])
    visited = {root}
    while queue:
        # Visit node
        i, d_i = queue.popleft()
        xyz_i = graph.node_xyz(i)
        if graph.degree[i] > 2:
            dist, _ = kdtree.query(xyz_i)
            if dist < 16:
                return i

        # Update queue
        for j in graph.neighbors(i):
            d_j = d_i + graph.physical_dist(i, j)
            if j not in visited and d_j < radius:
                queue.append((j, d_j))
                visited.add(j)
    return root


def flip_coordinates(graphs):
    """
    Flips the X and Z coordinates for a collections of graphs.

    Parameters
    ----------
    graph : Dict[str, FragmentGraph]
        Graphs to be updated.
    """
    for key, graph in graphs.items():
        graphs[key].node_voxel[:, [0, 2]] = graph.node_voxel[:, [2, 0]]


def set_kdtrees(graphs):
    """
    Sets "kdtree" attribute for a collection of graphs.

    Parameters
    ----------
    graph : Dict[str, FragmentGraph]
        Graphs to be updated.
    """
    for key in graphs:
        graphs[key].set_kdtree()
