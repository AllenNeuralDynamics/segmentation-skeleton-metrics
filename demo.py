"""
Created on Tue Jan 3 15:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

from aind_segmentation_evaluation.run_evaluation import (
    graph_based_eval,
    voxel_based_eval,
)

if __name__ == "__main__":

    # Initializations
    shape = (148, 226, 282)
    data_dir = "./resources"
    path_to_pred_volume = os.path.join(data_dir, "pred_volume.tif")
    path_to_target_volume = os.path.join(data_dir, "target_volume.tif")
    target_graphs_dir = os.path.join(data_dir, "target_graphs")

    # Evaluation
    graph_stats = graph_based_eval(
        shape,
        target_graphs_dir=target_graphs_dir,
        path_to_pred_volume=path_to_pred_volume,
        output="tif",
        output_dir=data_dir,
    )

    voxel_stats = voxel_based_eval(
        path_to_target_volume=path_to_target_volume,
        path_to_pred_volume=path_to_pred_volume,
    )

    # Write out results
    print("Graph-based evaluation...")
    for key in graph_stats.keys():
        print("   " + key + ":", graph_stats[key])
    print("")

    print("Voxel-based evaluation...")
    for key in voxel_stats.keys():
        print("   " + key + ":", voxel_stats[key])
