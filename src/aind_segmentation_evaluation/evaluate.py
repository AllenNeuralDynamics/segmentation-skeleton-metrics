"""
Created on Wed Jan 10 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import numpy as np
import aind_segmentation_evaluation.utils as utils
from aind_segmentation_evaluation.merge_metric import MergeMetric
from aind_segmentation_evaluation.split_metric import SplitMetric


def run_evaluation(
    target_graphs,
    target_labels,
    pred_graphs,
    pred_labels,
    clip=16,
    filetype=None,
    output=None,
    output_dir=None,
    scaling_factors=[1.10, 1.10, 1.10],
):
    """
    Evaluates a predicted segmentation in terms of the number of splits
    and merges.

    Parameters
    ----------
    target_graphs : list[networkx.Graph] or str
        List of graphs corresponding to target segmentation or path to
        directory of swc files.
    target_labels : np.array, n5, or str
        Target segmentation mask or path to target segmentation.
    pred_graphs : list[networkx.Graph] or str
        List of graphs corresponding to the predicted segmentation mask or path
        to directory of swc files.
    pred_labels : np.array, n5, or str
        Predicted segmentation mask or path to predicted segmentation.
    filetype : str, optional
        File type of target_labels and pred_labels. Supported file types
        include tif and n5. The default is None.
    output : str, optional
        Type of output, supported options include 'swc' and 'tif'.
        The default is None.
    output_dir : str, optional
        Path to directory that outputs are written to.
        The default is None.
    scaling_factors : list[float], optional
        Scaling factor from image to real-world coordinates.
        The default is None.

    Returns
    -------
    stats : dict
        Dictionary where the keys are the names of statistics that were
        computed and the corresponding values are the numerical values were
        computed.

    """
    # Split evaluation
    split_evaluator = SplitMetric(
        target_graphs,
        target_labels,
        pred_graphs,
        pred_labels,
        filetype=filetype,
        output=output,
        output_dir=output_dir,
        scaling_factors=scaling_factors,
    )
    split_evaluator.detect_mistakes()

    # Merge evaluation
    merge_evaluator = MergeMetric(
        target_graphs,
        target_labels,
        pred_graphs,
        pred_labels,
        filetype=filetype,
        output=output,
        output_dir=output_dir,
        scaling_factors=scaling_factors,
    )
    merge_evaluator.detect_mistakes()

    # Account for boundaries
    clipped_mask = utils.clip(split_evaluator.site_mask, clip)
    split_evaluator.interior_site_cnt = int(np.sum(clipped_mask > 0))
    merge_evaluator = rm_spurious_sites(merge_evaluator, pred_labels, clip)

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
        split_evaluator,
        merge_evaluator,
        target_graphs,
    )
    stats["erl"], stats["normalized_erl"] = compute_erl(
        split_evaluator, merge_evaluator
    )
    return stats


def rm_spurious_sites(evaluator, pred_labels, clip):
    """
    Removes merge sites that are very close. This is a common issue that causes
    merges to be over-counted.

    Parameters
    ----------
    evaluator : MergeMetric
        SegmentationMetric type object in which "detect_mistakes"
        has been run.

    """
    merge_sites = pred_labels * evaluator.site_mask
    for i in [i for i in np.unique(merge_sites) if i != 0]:
        site_mask_i = (merge_sites == i).astype(int)
        if np.sum(site_mask_i) > 1:
            x, y, z = np.where(site_mask_i)
            for i in range(len(x)):
                for j in range(len(x)):
                    nonzero_i = evaluator.site_mask[x[i], y[i], z[i]] > 0
                    nonzero_j = evaluator.site_mask[x[j], y[j], z[j]] > 0
                    if i < j and nonzero_i and nonzero_j:
                        dists = [abs(val[i] - val[j]) for val in [x, y, z]]
                        if np.sum(dists) <= 20:
                            evaluator.site_mask[x[i], y[i], z[i]] = 0

    clipped_mask = utils.clip(evaluator.site_mask, clip)
    evaluator.site_cnt = int(np.sum(evaluator.site_mask > 0))
    evaluator.interior_site_cnt = int(np.sum(clipped_mask))
    return evaluator


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
        x + "_inside_cnt": evaluator.interior_site_cnt // 2,
        x + "_edge_cnt": edge_cnt,
        x + "_ratio": site_cnt / total_edges,
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
    e1 = 0 if eval1 is None else eval1.edge_cnt
    e2 = 0 if eval2 is None else eval2.edge_cnt
    total_edges = count_edges(list_of_graphs)
    return 1 - (e1 + e2) / total_edges


def compute_erl(split_evaluator, merge_evaluator):
    """
    Computes the expected run length (ERL) of a predicted segmentation

    Paramters
    ---------
    split_evaluator : SplitMetric
        Type of SegmentationMetric.
    merge_evaluator : MergeMetric
        Type of SegmentationMetric.

    Returns
    -------
    erl : float
        Expected run length
    normalized_erl : float
        Normalized expected run length

    """
    path_lengths = dict()
    target_labels = merge_evaluator.labels
    for graph in split_evaluator.graphs:
        for i in graph.nodes:
            val_i = utils.get_value(target_labels, graph, i)
            if val_i != 0:
                path_lengths[val_i] = graph.number_of_edges()
                break

    # compute erl_i
    run_lengths = merge_evaluator.run_lengths
    erls = dict()
    for i in list(run_lengths.keys()):
        lens = np.array(run_lengths[i])
        if i in path_lengths.keys():
            val = np.sum(lens**2) / path_lengths[i]
            erls[i] = val if len(lens) > 0 else 0

    # compute erl
    erl = 0
    total_path_length = np.sum(list(path_lengths.values()))
    for i in list(erls.keys()):
        w_i = path_lengths[i] / total_path_length
        erl += w_i * erls[i]
    normalized_erl = erl / total_path_length
    return erl, normalized_erl


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
