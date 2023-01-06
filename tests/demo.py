"""
Created on Tue Jan 3 15:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from aind_segmentation_evaluation.merge_metric import MergeMetric
from aind_segmentation_evaluation.split_metric import SplitMetric

if __name__ == "__main__":

    # Initializations
    output_dir = "C:\\Users\\anna.grim\\Downloads"
    path_to_target_volume = (
        "C:\\Users\\anna.grim\\Downloads\\eval\\tests\\target_volume.tif"
    )
    path_to_pred_volume = (
        "C:\\Users\\anna.grim\\Downloads\\eval\\tests\\pred_volume.tif"
    )

    target_graphs_dir = (
        "C:\\Users\\anna.grim\\Downloads\\eval\\tests\\target_graphs"
    )
    pred_graphs_dir = (
        "C:\\Users\\anna.grim\\Downloads\\eval\\tests\\pred_graphs"
    )

    # Run evaluation ~ exact
    split_evaluator = SplitMetric(
        (148, 226, 282),
        target_graphs_dir=target_graphs_dir,
        path_to_pred_volume=path_to_pred_volume,
        output="tif",
        output_dir=output_dir,
    )
    merge_evaluator = MergeMetric(
        (148, 226, 282),
        target_graphs_dir=target_graphs_dir,
        path_to_pred_volume=path_to_pred_volume,
        output="tif",
        output_dir=output_dir,
    )
    split_evaluator.detect_mistakes()
    merge_evaluator.detect_mistakes()
    print("   Number of splits:", split_evaluator.split_cnt)
    print("   Number of split edges:", split_evaluator.split_edge_cnt)
    print("   Number of merges:", merge_evaluator.merge_cnt)
    print("   Number of merged edges:", merge_evaluator.merge_edge_cnt)
