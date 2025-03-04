"""
Created on Wed Dec 21 19:00:00 2022

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that runs of demo of using this library to compute skeleton metrics.

"""

from segmentation_skeleton_metrics.utils import util
from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from segmentation_skeleton_metrics.utils.img_util import TiffReader


def evaluate():
    """
    Evaluates the accuracy of a predicted segmentation by comparing it to a
    set of ground truth skeletons, then reports and saves various performance
    metrics.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # Initializations
    path = f"{output_dir}/results.xls"
    pred_labels = TiffReader(pred_labels_path)
    skeleton_metric = SkeletonMetric(
        groundtruth_pointer,
        pred_labels,
        fragments_pointer=fragments_pointer,
        output_dir=output_dir,
    )
    full_results, avg_results = skeleton_metric.run(path)


if __name__ == "__main__":
    # Initializations
    output_dir = "./"
    pred_labels_path = "./pred_labels.tif"
    fragments_pointer = "./pred_swcs.zip"
    groundtruth_pointer = "./target_swcs.zip"

    # Run
    evaluate()
