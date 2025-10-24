"""
Created on Thu Oct 16 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

from zipfile import ZipFile

import numpy as np
import os
import pandas as pd

from segmentation_skeleton_metrics.skeleton_metrics import (
    MergeCountMetric,
    MergeRateMetric,
    MergedEdgePercentMetric,
    OmitEdgePercentMetric,
    SplitEdgePercentMetric,
    SplitCountMetric,
    SplitRateMetric,
    EdgeAccuracyMetric,
    ERLMetric,
    NormalizedERLMetric
)
from segmentation_skeleton_metrics.data_handling.graph_loading import (
    DataLoader, LabelHandler
)
from segmentation_skeleton_metrics.utils import util


def evaluate(
    gt_pointer,
    label_mask,
    output_dir,
    anisotropy=(1.0, 1.0, 1.0),
    connections_path=None,
    fragments_pointer=None,
    results_filename="results",
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

    # Run evaluation
    evaluator = Evaluator(output_dir, results_filename, verbose)
    evaluator.run(gt_graphs, fragment_graphs)

    # Optional saves
    if save_merges:
        evaluator.save_merge_results()

    if save_fragments and fragment_graphs:
        evaluator.save_fragments(gt_graphs, fragment_graphs)


# --- Evaluator ---
class Evaluator:

    def __init__(self, output_dir, results_filename, verbose=True):
        # Instance attributes
        self.output_dir = output_dir
        self.results_filename = results_filename
        self.verbose = verbose

        # Set core metrics
        self.metrics = {
            "# Splits": SplitCountMetric(verbose=verbose),
            "# Merges": MergeCountMetric(verbose=verbose),
            "% Split Edges": SplitEdgePercentMetric(verbose=verbose),
            "% Omit Edges": OmitEdgePercentMetric(verbose=verbose),
            "% Merged Edges": MergedEdgePercentMetric(verbose=verbose),
            "ERL": ERLMetric(verbose=verbose)
        }

        # Set derived metrics
        self.derived_metrics = {
            "Normalized ERL": NormalizedERLMetric(verbose=verbose),
            "Edge Accuracy": EdgeAccuracyMetric(verbose=verbose),
            "Split Rate": SplitRateMetric(verbose=verbose),
            "Merge Rate": MergeRateMetric(verbose=verbose),
        }

    # --- Core Routines ---
    def run(self, gt_graphs, fragment_graphs=None):
        # Printout step
        if self.verbose:
            print("\n(3) Compute Metrics")

        # Compute core metrics
        results = self.init_results(gt_graphs)
        for name, metric in self.metrics.items():
            if name == "# Merges" and fragment_graphs:
                results[name] = metric.compute(gt_graphs, fragment_graphs)
            elif name != "# Merges":
                results.update(metric.compute(gt_graphs))

        # Compute derived metrics
        for name, metric in self.derived_metrics.items():
            if name == "Merge Rate" and fragment_graphs:
                results[name] = metric.compute(gt_graphs, results)
            elif name != "Merge Rate":
                results[name] = metric.compute(gt_graphs, results)

        # Save report
        path = f"{self.output_dir}/{self.results_filename}.csv"
        results.to_csv(path, index=True)
        self.report_summary(results)

    def init_results(self, gt_graphs):
        # Create dataframe
        cols = (
            ["SWC Run Length"] +
            list(self.metrics.keys()) +
            list(self.derived_metrics.keys())
        )
        index = list(gt_graphs.keys())
        index.sort()
        results = pd.DataFrame(np.nan, index=index, columns=cols)

        # Populate SWC Run Length column
        for key, graph in gt_graphs.items():
            results.loc[key, "SWC Run Length"] = graph.run_length
        return results

    def report_summary(self, results):
        pass

    # --- Writers ---
    def save_fragments(self):
        pass

    def save_merge_results(self, gt_graphs, fragment_graphs, output_dir):
        # Initialize a writer
        zip_path = os.path.join(output_dir, "merged_fragments.zip")
        util.rm_file(zip_path)
        zip_writer = ZipFile(zip_path, "a")

        # Save SWC files
        self.save_merge_sites(zip_writer)
        self.save_skeletons_with_merge(gt_graphs, fragment_graphs, zip_writer)
        zip_writer.close()

        # Save CSV file
        path = os.path.join(output_dir, "merge_sites.csv")
        self.merge_sites.to_csv(path, index=True)

    def save_merge_sites(self, zip_writer):
        merge_sites = self.metrics["# Merges"].merge_sites
        for i in range(len(merge_sites)):
            filename = merge_sites.index[i]
            xyz = merge_sites["World"].iloc[i]
            util.to_zipped_point(zip_writer, filename, xyz)

    def save_skeletons_with_merge(
        self, gt_graphs, fragment_graphs, zip_writer
    ):
        # Save ground truth skeletons
        for key in self.merge_sites["GroundTruth_ID"].unique():
            gt_graphs[key].to_zipped_swc(zip_writer)

        # Save fragments
        for key in self.fragments_with_merge:
            fragment_graphs[key].to_zipped_swc(zip_writer)
