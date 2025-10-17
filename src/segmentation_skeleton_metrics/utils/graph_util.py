"""
Created on Wed Aug 15 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for building a custom graph object called a SkeletonGraph and helper
routines for working with graph.

"""

import networkx as nx


# --- Helpers ---
def count_splits(graph):
    """
    Counts the number of split mistakes in the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be evaluated.

    Returns
    -------
    int
        Number of split mistakes in the given graph.
    """
    return max(nx.number_connected_components(graph) - 1, 0)


def get_leafs(graph):
    """
    Gets all leafs nodes in the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    List[int]
        Leaf nodes in the given graph.
    """
    return [node for node in graph.nodes if graph.degree[node] == 1]


def write_graph(graph, zip_writer):
    """
    Writes a graph to ZIP archive containing SWC files if it has not already
    been written.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be written to an SWC file.
    zip_writer : zipfile.ZipFile
        An open ZipFile handle in write or append mode where the SWC file will
        be written.
    """
    if graph.filename not in zip_writer.namelist():
        graph.to_zipped_swc(zip_writer)
