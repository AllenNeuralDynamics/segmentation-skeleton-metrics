"""
Created on Thu Oct 16 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.data_handling.graph_loading import DataLoader


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
    # Load data
    dataloader = DataLoader(
        anisotropy=anisotropy,
        connections_path=connections_path,
        use_anisotropy=use_anisotropy,
        valid_labels=valid_labels,
        verbose=verbose
    )
    gt_graphs = dataloader.load_groundtruth(gt_pointer, label_mask)
    fragments_graph = dataloader.load_fragments(fragments_pointer, gt_graphs)

    # Evaluator
    skeleton_metric = SkeletonMetric(
        gt_pointer,
        label_mask,
        anisotropy=anisotropy,
        connections_path=connections_path,
        fragments_pointer=fragments_pointer,
        output_dir=output_dir,
        save_merges=save_merges,
        save_fragments=save_fragments,
        use_anisotropy=use_anisotropy,
    )
    skeleton_metric.run()

    # Compute metrics

    # Report results


# --- Evaluator ---
class Evaluator:
    pass
