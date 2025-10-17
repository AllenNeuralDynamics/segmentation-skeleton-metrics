"""
Created on Thu Oct 16 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.data_handling.graph_loading import (
    DataLoader, LabelHandler
)


def evaluate(
    gt_pointer,
    label_mask,
    output_dir,
    anisotropy=(1.0, 1.0, 1.0),
    connections_path=None,
    fragments_pointer=None,
    save_merges=False,
    save_fragments=False,
    use_anisotropy=False,
    valid_labels=None,
    verbose=True
):
    """
    ...

    Parameters
    ----------
    gt_pointer : str
        Pointer to ground truth SWC files, see "swc_util.Reader" for
        documentation. These SWC files are assumed to be stored in voxel
        coordinates.
    label_mask : ImageReader
        Predicted segmentation.
    anisotropy : Tuple[float], optional
        ...
    connections_path : str, optional
        Path to a txt file containing pairs of segment IDs that represents
        fragments that were merged. Default is None.
    fragments_pointer : str, optional
        Pointer to SWC files corresponding to "label_mask", see
        "swc_util.Reader" for documentation. Notes: (1) "anisotropy" is
        applied to these SWC files and (2) these SWC files are required
        for counting merges. Default is None.
    """
    # Load data
    label_handler = LabelHandler(connections_path, valid_labels)
    dataloader = DataLoader(
        label_handler,
        anisotropy=anisotropy,
        use_anisotropy=use_anisotropy,
        verbose=verbose
    )    
    gt_graphs = dataloader.load_groundtruth(gt_pointer, label_mask)
    fragment_graphs = dataloader.load_fragments(fragments_pointer, gt_graphs)

    # Evaluator
    skeleton_metric = SkeletonMetric(
        gt_graphs,
        fragment_graphs,
        label_handler,
        anisotropy=anisotropy,
        output_dir=output_dir,
        save_merges=save_merges,
        save_fragments=save_fragments,
    )
    skeleton_metric.run()

    # Compute metrics

    # Report results


# --- Evaluator ---
class Evaluator:
    pass
