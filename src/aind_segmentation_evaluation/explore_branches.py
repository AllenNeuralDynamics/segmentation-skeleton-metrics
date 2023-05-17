"""
Created on Mon March 6 19:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import networkx as nx


def prune_spurious_paths(graph, min_branch_length=5):
    """
    Prunes short branches.

    Parameters
    ----------
    graph : networkx.graph
        Graph to be pruned.
    min_branch_length : int
        Upper bound on short branch length to be pruned.

    Returns
    -------
    graph : networkx.graph
        Graph with short branches pruned.

    """
    leaf_nodes = [i for i in graph.nodes if graph.degree[i] == 1]
    for leaf in leaf_nodes:
        # Traverse branch from leaf
        queue = [leaf]
        visited = set()
        hit_junction = False
        while len(queue) > 0:
            node = queue.pop(0)
            nbs = list(graph.neighbors(node))
            if len(nbs) > 2:
                hit_junction = True
                break
            else:
                visited.add(node)
                nb = [nb for nb in nbs if nb not in visited]
                queue.extend(nb)

        # Check length of branch
        if hit_junction and len(visited) <= min_branch_length:
            graph.remove_nodes_from(visited)
    return graph


def detect_short_connectors(graph, min_connector_length):
    """ "
    Detects shorts paths between branches (i.e. paths that connect branches).

    Parameters
    ----------
    graph : netowrkx.graph
        Graph to be inspected.
    min_connector_length : int
        Upper bound on short paths that connect branches.

    Returns
    -------
    remove_edges : list[tuple]
        List of edges to be removed.
    remove_nodes : list[int]
        List of nodes to be removed.

    """
    leaf_nodes = [i for i in graph.nodes if graph.degree[i] == 1]
    dfs_edges = list(nx.dfs_edges(graph, leaf_nodes[0]))
    remove_nodes = []
    remove_edges = []
    flag_junction = False
    path_length = 0
    for (i, j) in dfs_edges:
        # Check for junction
        if graph.degree[i] > 2:
            flag_junction = True
            path_length = 1
            cur_branch = [(i, j)]
        elif flag_junction:
            path_length += 1
            cur_branch.append((i, j))

        # Check whether to reset
        if graph.degree[j] == 1:
            flag_junction = False
            cur_branch = list()
        elif graph.degree[j] > 2 and flag_junction:
            if path_length < min_connector_length:
                remove_edges.extend(cur_branch)
                remove_nodes.extend(graph.neighbors(cur_branch[0][0]))
                remove_nodes.extend(graph.neighbors(j))
                cur_branch = list()
    return remove_edges, remove_nodes


def prune_short_connectors(list_of_graphs, min_connector_length=10):
    """
    Prunes short connecting paths on graph in "list_of_graphs".

    Parameters
    ----------
    list_of_graphs : list[networkx.graph]
        List of graphs such that short connecting paths will be pruned on
        each graph.
    min_connector_length : int
        Upper bound on short paths that connect branches.

    Returns
    -------
    upd : list[networkx.graph]
        List of graphs with short connecting paths pruned.

    """
    upd = []
    for graph in list_of_graphs:
        pruned_graph = prune_spurious_paths(graph)
        if pruned_graph.number_of_nodes() > 3:
            remove_edges, remove_nodes = detect_short_connectors(
                pruned_graph, min_connector_length
            )
            graph.remove_edges_from(remove_edges)
            graph.remove_nodes_from(remove_nodes)
            for g in nx.connected_components(graph):
                subgraph = graph.subgraph(g).copy()
                if subgraph.number_of_nodes() > 10:
                    upd.append(subgraph)
    return upd


def break_crossovers(list_of_graphs, depth=10):
    """
    Breaks crossovers for each graph contained in "list_of_graphs".

    Parameters
    ----------
    list_of_graphs : list[networkx.graph]
        List of graphs such that crossovers will be broken on each graph.
    depth : int
        Maximum depth of dfs performed to detect crossovers.

    Returns
    -------
    upd : list[networkx.graph]
        List of graphs with crossovers broken.

    """
    upd = []
    for i, graph in enumerate(list_of_graphs):
        pruned_graph = prune_spurious_paths(graph, min_branch_length=depth + 1)
        prune_nodes = detect_crossovers(pruned_graph, depth)
        if len(prune_nodes) > 0:
            graph.remove_nodes_from(prune_nodes)
            for g in nx.connected_components(graph):
                subgraph = graph.subgraph(g).copy()
                if subgraph.number_of_nodes() > 10:
                    upd.append(subgraph)
        else:
            upd.append(graph)
    return upd


def detect_crossovers(graph, depth):
    """
    Detects crossovers in "graph".

    Parameters
    ----------
    graph : networkx.graph
        Graph to be inspected.
    depth : int
        Maximum depth of dfs performed to detect crossovers.

    Returns
    -------
    prune_nodes : list[int]
        Nodes that are part of a crossover and should be pruned.

    """
    cnt = 0
    prune_nodes = []
    junctions = [i for i in graph.nodes() if graph.degree(i) > 2]
    for j in junctions:
        # Explore node
        upd = False
        tree, leafs = count_branches(graph, j, depth)
        num_leafs = len(leafs)

        # Detect crossover
        if num_leafs > 3:
            cnt += 1
            upd = True
            for d in range(1, depth):
                tree_d, leafs_d = count_branches(graph, j, d)
                if len(leafs_d) == num_leafs:
                    prune_nodes.extend(tree_d.nodes())
                    upd = False
                    break
            if upd:
                prune_nodes.extend(tree.nodes())
    return prune_nodes


def count_branches(graph, source, depth):
    """
    Counts the number of branches emanating from "source" by running a
    bounded dfs.

    Parameters
    ----------
    graph : networkx.graph
        Graph that contains "source".
    source : int
        Node that is contained in "graph".
    depth : int
        Maximum depth of dfs.

    Returns
    -------
    tree : networkx.dfs_tree
        Tree-structured graph rooted at "source".
    leafs : list[int]
        List of leaf nodes in "tree".

    """
    tree = nx.dfs_tree(graph, source=source, depth_limit=depth)
    leafs = [i for i in tree.nodes() if tree.degree(i) == 1]
    return tree, leafs
