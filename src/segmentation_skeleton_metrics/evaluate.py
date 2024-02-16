"""
Created on Wed Jan 10 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import numpy as np

from segmentation_skeleton_metrics import utils
from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric


def run_evaluation(
    swc_paths,
    labels,
    anisotropy=[1.0, 1.0, 1.0],
    valid_ids=None,
):
    """
    Evaluates a predicted segmentation in terms of the number of splits
    and merges.

    Parameters
    ----------
    swc_paths : str
        List of paths to swc files generated from the target segmentation.
    labels : np.array or ts.TensorStore
        Label mask of predicted segmentation.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factor for (x, y, z) which is
        applied to swc files.
    valid_ids : set
        ...

    Returns
    -------
    stats : dict
        Dictionary of (stat, value) pairs that were computed during the
        evaluation.

    """
    # Split evaluation
    skeleton_metric = SkeletonMetric(
        swc_paths,
        labels,
        anisotropy=anisotropy,
        valid_ids=valid_ids,
    )
    skeleton_metric.detect_mistakes()

    # Compute stats
    stats = dict()
    target_graphs = split_evaluator.graphs
    stats.update(compute_stats(split_evaluator, target_graphs, "split"))
    stats["edge_accuracy"] = compute_edge_accuracy(
        split_evaluator, merge_evaluator, target_graphs
    )

    utils.write_json(os.path.join(log_dir, "stats.json"), stats)
    return stats


def compute_stats(evaluator, list_of_graphs, x):
    """
    Computes various statistics that provide an evaluation of the
    predicted segmentation mask.

    Parameters
    ----------
    evaluator: SplitMetric
        SegmentationMetric type object in which "detect_mistakes"
        has been run.
    list_of_graphs : list[networkx.Graph]
        List of graphs where each graph corresponds to a neuron.

    Returns
    -------
    stats : dict
        Dictionary where the keys are the names of statistics that were
        computed and the corresponding values are the numerical values were
        computed.

    """
    site_cnt = evaluator.site_cnt
    edge_cnt = evaluator.edge_cnt
    total_edges = count_edges(list_of_graphs)
    stats = {
        x + "_cnt": site_cnt,
        x + "_edge_cnt": edge_cnt,
        x + "_edge_ratio": edge_cnt / total_edges,
    }
    return stats


def compute_edge_accuracy(eval1, list_of_graphs):
    """
    Computes the percentage of correctly reconstructed edges from
    "list_of_graphs".

    Parameters
    ----------
    eval1 : SplitMetric
        Type of SegmentationMetric.
    list_of_graphs : list[networkx.Graph]
        List of graphs where each graph corresponds to a neuron.

    Returns
    -------
    edge_accuracy : float
        Percentage of correctly reconstructed edges from
        "list_of_graphs".

    """
    e1 = eval1.edge_cnt
    e2 = eval2.edge_cnt
    total_edges = count_edges(list_of_graphs)
    return 1 - (e1 + e2) / total_edges


def count_edges(list_of_graphs):
    """
    Counts number of edges in 'list_of_graphs'.

    Parameters
    ----------
    graph : list[networkx.Graph]
        List of graphs where each graph corresponds to a neuron.

    Returns
    -------
    cnt : int
        Number of edges in 'list_of_graphs'.

    """
    return np.sum([graph.number_of_edges() for graph in list_of_graphs])
