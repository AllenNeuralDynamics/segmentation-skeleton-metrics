"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that runs of demo of using this library to compute skeleton metrics.

"""

from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.utils.img_util import TiffImage


def main():
    """
    Evaluates the accuracy of a predicted segmentation by comparing it to a
    set of ground truth skeletons, then reports and saves various performance
    metrics.
    """
    # Initializations
    segmentation = TiffImage(segmentation_path, swap_axes=False)
    evaluate(
        groundtruth_path,
        segmentation,
        output_dir,
        fragments_path=fragments_path,
    )


if __name__ == "__main__":
    # Initializations
    output_dir = "./"
    segmentation_path = "./data/pred_segmentation.tif"
    fragments_path = "./data/pred_swcs.zip"
    groundtruth_path = "./data/target_swcs.zip"

    # Run
    main()
