"""
Created on Tue Jan 3 15:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import os

from aind_segmentation_evaluation.evaluate import run_evaluation
from aind_segmentation_evaluation.graph_routines import volume_to_graph
from tifffile import imread


if __name__ == "__main__":

    # Initializations
    data_dir = "./resources"
    target_graphs_dir = os.path.join(data_dir, "target_graphs")
    path_to_target_labels = os.path.join(data_dir, "target_labels.tif")
    pred_labels = imread(os.path.join(data_dir, "pred_labels.tif"))
    pred_graphs = volume_to_graph(pred_labels)

    # Evaluation
    stats = run_evaluation(
        target_graphs_dir,
        path_to_target_labels,
        pred_graphs,
        pred_labels,
        output="tif",
        output_dir=data_dir,
        filetype="tif",
    )

    # Write out results
    print("Graph-based evaluation...")
    for key in stats.keys():
        print("   " + key + ":", stats[key])
