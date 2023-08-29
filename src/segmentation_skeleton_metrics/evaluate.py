"""
Created on Wed Jan 10 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

import numpy as np

from segmentation_skeleton_metrics import utils
from segmentation_skeleton_metrics.merge_metric import MergeMetric
from segmentation_skeleton_metrics.split_metric import SplitMetric


def run_evaluation(
    target_swc_dir,
    target_labels,
    pred_swc_dir,
    pred_labels,
    anisotropy=[1.0, 1.0, 1.0],
    filetype=None,
    log_dir=None,
    swc_log=False,
    txt_log=False,
):
    """
    Evaluates a predicted segmentation in terms of the number of splits
    and merges.

    Parameters
    ----------
    target_swc_dir : str
        Path to directory of swc files of the target segmentation.
    target_labels : np.array, ts.TensorStore, dict, or str
        Target segmentation mask or path to it.
    pred_swc_dir : str
        Path to directory of swc files of the predicted
        segmentation.
    pred_labels : np.array, ts.TensorStore, dict, or str
        Predicted segmentation mask or path to it.
    anisotropy : list[float], optional
        Image to real-world coordinates scaling factor for (x, y, z) which is
        applied to swc files.
    filetype : str, optional
        File type of target_labels and pred_labels if path is provided.
        The default is None.
    log_dir : str, optional
        Directory where logged information (i.e. swc log and txt log)
        is saved.
    swc_log : bool, optional
        Indicates whether to store swc files that indicates where mistakes are
        located.
    txt_log : bool, optional
        Indicates whether to store a txt files that contains (x, y, z)
        coordinates of where mistakes are located.

    Returns
    -------
    stats : dict
        Dictionary of (stat, value) pairs that were computed during the
        evaluation.

    """
    if log_dir is not None:
        utils.rmdir(log_dir)

    # Split evaluation
    split_evaluator = SplitMetric(
        target_swc_dir,
        pred_labels,
        anisotropy=anisotropy,
        filetype=filetype,
        prefix="split-",
        log_dir=log_dir,
        swc_log=swc_log,
        txt_log=txt_log,
    )
    split_evaluator.detect_mistakes()

    # Merge evaluation
    merge_evaluator = MergeMetric(
        pred_swc_dir,
        target_labels,
        anisotropy=anisotropy,
        filetype=filetype,
        prefix="merge-",
        log_dir=log_dir,
        swc_log=swc_log,
        txt_log=txt_log,
    )
    merge_evaluator.detect_mistakes()

    # Compute stats
    stats = dict()
    target_graphs = split_evaluator.graphs
    stats.update(compute_stats(split_evaluator, target_graphs, "split"))
    stats.update(compute_stats(merge_evaluator, target_graphs, "merge"))
    stats["num_mistakes"] = split_evaluator.site_cnt + merge_evaluator.site_cnt
    stats["wgt_mistakes"] = (
        split_evaluator.site_cnt + 3 * merge_evaluator.site_cnt
    )
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
    evaluator: MergeMetric or SplitMetric
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


def compute_edge_accuracy(eval1, eval2, list_of_graphs):
    """
    Computes the percentage of correctly reconstructed edges from
    "list_of_graphs".

    Parameters
    ----------
    eval1 : MergeMetric or SplitMetric
        Type of SegmentationMetric.
    eval2 : MergeMetric or SplitMetric
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
