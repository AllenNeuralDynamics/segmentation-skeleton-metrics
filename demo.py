"""
Created on Tue Jan 3 15:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

from aind_segmentation_evaluation.evaluate import run_evaluation


if __name__ == "__main__":

    # Initializations
    shape = (148, 226, 282)
    data_dir = "./resources"
    path_to_pred_volume = os.path.join(data_dir, "pred_volume.tif")
    path_to_target_volume = os.path.join(data_dir, "target_volume.tif")
    target_graphs_dir = os.path.join(data_dir, "target_graphs")

    # Evaluation
    stats = run_evaluation(
        shape,
        target_graphs_dir=target_graphs_dir,
        path_to_pred_volume=path_to_pred_volume,
        output="tif",
        output_dir=data_dir,
    )

    # Write out results
    print("Graph-based evaluation...")
    for key in stats.keys():
        print("   " + key + ":", stats[key])
