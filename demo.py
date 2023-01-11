"""
Created on Tue Jan 3 15:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

from aind_segmentation_evaluation.merge_metric import MergeMetric
from aind_segmentation_evaluation.split_metric import SplitMetric

if __name__ == "__main__":

    # Initializations
    shape = (148, 226, 282)
    output_dir = "./resources"
    path_to_pred_volume = os.path.join(output_dir, 'pred_volume.tif')
    target_graphs_dir = os.path.join(output_dir, 'target_graphs')

    # Run evaluation ~ exact
    split_evaluator = SplitMetric(
        shape,
        target_graphs_dir=target_graphs_dir,
        path_to_pred_volume=path_to_pred_volume,
        output="tif",
        output_dir=output_dir,
    )
    merge_evaluator = MergeMetric(
        shape,
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
