"""
Created on Wed Jan 10 12:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from skimage.io import imread
from skimage.metrics import adapted_rand_error, variation_of_information

from aind_segmentation_evaluation.merge_metric import MergeMetric
from aind_segmentation_evaluation.split_metric import SplitMetric


def graph_based_eval(
    shape,
    eval_merges=True,
    eval_splits=True,
    target_graphs=None,
    target_graphs_dir=None,
    target_volume=None,
    path_to_target_volume=None,
    pred_graphs=None,
    pred_graphs_dir=None,
    pred_volume=None,
    path_to_pred_volume=None,
    output=None,
    output_dir=None,
):
    """
    Evaluates a predicted segmentation in terms of the number of splits
    and merges. At least one of {target_graphs, target_graphs_dir,
    target_volume, path_to_target_volume} and one of {pred_volume,
    path_to_pred_volume, pred_graphs, pred_graphs_dir} must be
    provided. The recommended inputs are

        (1) "target_graphs_dir" or "target_graphs"
        (2) "path_to_pred_volume" or "pred_volume"
        (3) "pred_graphs_dir" or "pred_graphs"

    Parameters
    ----------
    shape : tuple
        The xyz- dimensions of volume.
    eval_merges : bool, optional
        Boolean that indicates whether to run merge detection.
        The default is True.
    eval_splits : bool, optional
        Boolean that indicates whether to run split detection.
        The default is True.
    target_graphs : list[networkx.Graph], optional
        List of graphs corresponding to target segmentation.
        The default is None.
    target_graph_dir : str, optional
        Path to directory containing swc files of target segmentation.
        The default is None.
    target_volume : np.array, optional
        Target segmentation mask.
        The default is None.
    path_to_target_volume : str, optional
        Path to target segmentation mask (i.e. tif file).
        The default is None.
    pred_graphs : list[nx.Graph], optional
        List of graphs corresponding to the predicted segmentation mask.
        The default is None.
    pred_graph_dir : str, optional
        Path to directory with swc files of predicted segmentation mask.
        The default is None.
    pred_volume : np.array, optional
        Predicted segmentation mask.
        The default is None
    path_to_pred_volume : str, optional
        Path to predicted segmentation mask (i.e. tif file).
        The default is None.
    output : str, optional
        Type of output, supported options include 'swc' and 'tif'.
        The default is None.
    output_dir : str, optional
        Path to directory that outputs are written to.
        The default is None.

    Returns
    -------
    stats : dict
        Dictionary where the keys are the names of statistics that were
        computed and the corresponding values are the numerical values were
        computed.

    """
    # Initializations
    stats = dict()
    target_graphs = None
    split_evaluator = None
    merge_evaluator = None

    # Split evaluation
    if eval_splits:
        split_evaluator = SplitMetric(
            shape,
            target_graphs=target_graphs,
            target_graphs_dir=target_graphs_dir,
            target_volume=target_volume,
            path_to_target_volume=path_to_target_volume,
            pred_graphs=pred_graphs,
            pred_graphs_dir=pred_graphs_dir,
            pred_volume=pred_volume,
            path_to_pred_volume=path_to_pred_volume,
            output=output,
            output_dir=output_dir,
        )

        split_evaluator.detect_mistakes()
        target_graphs = split_evaluator.graphs
        stats.update(compute_stats(split_evaluator, target_graphs, "split"))

    # Merge evaluation
    if eval_merges:
        merge_evaluator = MergeMetric(
            shape,
            target_volume=target_volume,
            path_to_target_volume=path_to_target_volume,
            target_graphs=target_graphs,
            target_graphs_dir=target_graphs_dir,
            pred_graphs=pred_graphs,
            pred_graphs_dir=pred_graphs_dir,
            pred_volume=pred_volume,
            path_to_pred_volume=path_to_pred_volume,
            output=output,
            output_dir=output_dir,
        )

        merge_evaluator.detect_mistakes()
        stats.update(compute_stats(merge_evaluator, target_graphs, "merge"))

    # Compute additional stats
    if eval_splits or eval_merges:
        stats["edge_accuracy"] = compute_edge_accuracy(
            split_evaluator,
            merge_evaluator,
            target_graphs,
        )

    return stats


def voxel_based_eval(
    target_volume=None,
    path_to_target_volume=None,
    pred_volume=None,
    path_to_pred_volume=None,
):
    """
    Evaluates a predicted segmentation mask in terms of the number of
    splits and merges. At least one of {target_volume, path_to_target_volume}
    and one of {pred_volume, path_to_pred_volume} must be provided.

    target_volume : np.array, optional
        Target segmentation mask.
        The default is None.
    path_to_target_volume : str, optional
        Path to target segmentation mask (i.e. tif file).
        The default is None.
    pred_volume : np.array, optional
        Predicted segmentation mask.
        The default is None.
    path_to_pred_volume : str, optional
        Path to predicted segmentation mask (i.e. tif file).
        The default is None.

    Returns
    -------
    stats : dict
        Dictionary where the keys are the names of statistics that were
        computed and the corresponding values are the numerical values were
        computed.

    """

    # Upload data
    if target_volume is None:
        target_volume = imread(path_to_target_volume)

    if pred_volume is None:
        pred_volume = imread(path_to_pred_volume)

    # Run evaluation
    rand, _, _ = adapted_rand_error(target_volume, pred_volume)
    voi_splits, voi_merges = variation_of_information(
        target_volume, pred_volume
    )

    # Compile stats
    stats = {
        "rand": rand,
        "voi_splits": voi_splits,
        "voi_merges": voi_merges,
        "voi": voi_splits + voi_merges,
    }

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
    float : float
        percentage of correctly reconstructed edges from
        "list_of_graphs".

    """
    e1 = 0 if eval1 is None else eval1.edge_cnt
    e2 = 0 if eval2 is None else eval2.edge_cnt
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
    cnt = 0
    for graph in list_of_graphs:
        cnt += graph.number_of_edges()
    return cnt
