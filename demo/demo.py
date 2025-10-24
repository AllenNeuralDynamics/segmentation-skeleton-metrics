"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that runs of demo of using this library to compute skeleton metrics.

"""

from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.utils.img_util import TiffReader


def main():
    """
    Evaluates the accuracy of a predicted segmentation by comparing it to a
    set of ground truth skeletons, then reports and saves various performance
    metrics.
    """
    # Initializations
    pred_labels = TiffReader(pred_labels_path, swap_axes=False)
    evaluate(
        groundtruth_pointer,
        pred_labels,
        output_dir,
        fragments_pointer=fragments_pointer,
    )


if __name__ == "__main__":
    # Initializations
    output_dir = "./"
    pred_labels_path = "./data/pred_labels.tif"
    fragments_pointer = "./data/pred_swcs.zip"
    groundtruth_pointer = "./data/target_swcs.zip"

    # Run
    main()
