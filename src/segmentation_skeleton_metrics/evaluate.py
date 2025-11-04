"""
Created on Thu Oct 16 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that provides a wrapper for loading data and calling the SkeletMetric
subclasses.

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
    segmentation,
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
    Loads data, calls an evaluator object that computes skeleton-based
    segmentation, and saves the results.

    Parameters
    ----------
    gt_pointer : str
        Pointer to ground truth SWC files, see "swc_util.Reader" for
        documentation. These SWC files are assumed to be stored in voxel
        coordinates.
    segmentation : ImageReader
        Predicted segmentation.
    anisotropy : Tuple[float], optional
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
    connections_path : str, optional
        Path to a txt file containing pairs of segment IDs that represents
        fragments that were merged. Default is None.
    fragments_pointer : str, optional
        Pointer to SWC files corresponding to "segmentation", see
        "swc_util.Reader" for documentation. Notes: (1) "anisotropy" is
        applied to these SWC files and (2) these SWC files are required
        for counting merges. Default is None.
    results_filename : str, optional
        Name of file that skeleton metric results are written to. Default is
        "results".
    save_merges : bool, optional
        Indication of whether to save merge sites and fragments with a merge
        mistake. Default is False.
    save_fragments : bool, optional
        Indication of whether to save fragments that intersect with each
        ground truth skeleton. Default is False.
    use_anisotropy : bool, optional
        Indication of whether to apply the anisotropy to the coordinates from
        the fragment SWC files. Default is False.
    valid_labels : Set[int], optional
        Labels that are allowed to be assigned. Default is None.
    verbose : bool, optional
        Indication of whether to display progress bars and printout results.
        Default is True.
    """
    # Load data
    label_handler = LabelHandler(connections_path, valid_labels)
    dataloader = DataLoader(
        label_handler,
        anisotropy=anisotropy,
        use_anisotropy=use_anisotropy,
        verbose=verbose
    )
    gt_graphs = dataloader.load_groundtruth(gt_pointer, segmentation)
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
    """
    A class that evaluates neuron reconstruction quality by computing a set of
    skeleton-based metrics.

    Attributes
    ----------
    output_dir : str
        Directory where evaluation results will be saved.
    results_filename : str
        Filename (without extension) for the CSV report.
    verbose : bool
        Indication of whether to display progress bars and printout results.
    metrics : dict
        Core evaluation metrics mapping metric names to metric objects:
        - "# Splits": SplitCountMetric
        - "# Merges": MergeCountMetric
        - "% Split Edges": SplitEdgePercentMetric
        - "% Omit Edges": OmitEdgePercentMetric
        - "% Merged Edges": MergedEdgePercentMetric
        - "ERL": ERLMetric
    derived_metrics : dict
        Derived metrics computed from core metrics:
        - "Normalized ERL": NormalizedERLMetric
        - "Edge Accuracy": EdgeAccuracyMetric
        - "Split Rate": SplitRateMetric
        - "Merge Rate": MergeRateMetric
    """

    def __init__(self, output_dir, results_filename, verbose=True):
        """
        Instantiates an Evaluator object.

        Parameters
        ----------
        output_dir : str
            Directory where evaluation results will be saved.
        results_filename : str
            Filename (without extension) for the CSV report.
        verbose : bool, optional
            Indication of whether to display progress bars and printout
            results. Default is True.
        """
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
        """
        Computes evaluation metrics for neuron reconstructions and saves a CSV
        report.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.
        fragment_graphs : Dict[str, FragmentsGraph], optional
            Graphs built from skeletons obtained from a segmentation. This
            parameter is required to compute the metric "# Merges". Default
            is None.
        """
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
        """
        Initializes a data frame that results from skeleton metrics will be
        stored in.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs to be evaluated.

        Returns
        -------
        results : pandas.DataFrame
            Data frame that results from skeleton metrics will be stored in.
        """
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
        """
        Generates and saves a summary report of evaluation results.

        Parameters
        ----------
        results : pandas.DataFrame
            DataFrame containing evaluation results for individual SWCs.
        """
        # Averaged results
        filename = f"{self.results_filename}-overview.txt"
        path = os.path.join(self.output_dir, filename)
        util.update_txt(path, "\nAverage Results...", self.verbose)
        for column in results.columns:
            if column != "SWC Run Length" and column != "SWC Name":
                avg = util.compute_weighted_avg(results, column)
                util.update_txt(path, f"  {column}: {avg:.4f}", self.verbose)

        # Total results
        n_splits = results["# Splits"].sum()
        util.update_txt(path, "\nTotal Results...", self.verbose)
        util.update_txt(path, f"  # Splits: {n_splits}", self.verbose)
        if "# Merges" in results.columns:
            n_merges = results["# Merges"].sum()
            util.update_txt(path, f"  # Merges: {n_merges}", self.verbose)

    # --- Writers ---
    def save_fragments(self, gt_graphs, fragment_graphs):
        """
        Saves ground-truth graphs and their intersecting fragment graphs to
        zipped SWC files.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs built from ground truth SWC files.
        fragment_graphs : Dict[str, FragmentsGraph]
            Graphs built from skeletons obtained from a segmentation.
        """
        # Initializations
        fragments_dir = os.path.join(self.output_dir, "fragments")
        util.mkdir(fragments_dir, delete=True)

        # Main
        for key, graph in gt_graphs.items():
            # Create zip writer
            zip_path = os.path.join(fragments_dir, f"{graph.name}.zip")
            zip_writer = ZipFile(zip_path, "a")

            # Save skeletons
            graph.to_zipped_swc(zip_writer)
            self.save_intersecting_fragments(
                graph, fragment_graphs, zip_writer
            )

    @staticmethod
    def save_intersecting_fragments(gt_graph, fragment_graphs, zip_writer):
        """
        Saves SWC files for all fragment graphs whose label intersects with
        the given ground-truth graph.

        Parameters
        ----------
        gt_graph : LabeledGraph
            Graphs built from ground truth SWC files.
        fragment_graphs : Dict[str, FragmentGraph]
            Graphs built from skeletons obtained from a segmentation.
        zip_writer : zipfile.ZipFile
            Open ZIP file handle used to write fragments.
        """
        intersecting_labels = gt_graph.get_node_labels()
        for key, graph in fragment_graphs.items():
            if graph.label in intersecting_labels:
                graph.to_zipped_swc(zip_writer)

    def save_merge_results(self, gt_graphs, fragment_graphs, output_dir):
        """
        Saves all detected merge results, including skeletons, merge sites,
        and metadata.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs built from ground truth SWC files.
        fragment_graphs : Dict[str, FragmentsGraph]
            Graphs built from skeletons obtained from a segmentation.
        output_dir : str
            Directory that results are written to.
        """
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
        """
        Saves merge site coordinates into a ZIP archive.

        Parameters
        ----------
        zip_writer : zipfile.ZipFile
            Open ZIP file handle used to write merge site data.
        """
        merge_sites = self.metrics["# Merges"].merge_sites
        for i in range(len(merge_sites)):
            filename = merge_sites.index[i]
            xyz = merge_sites["World"].iloc[i]
            util.to_zipped_point(zip_writer, filename, xyz)

    def save_skeletons_with_merge(
        self, gt_graphs, fragment_graphs, zip_writer
    ):
        """
        Saves ground truth and fragment skeletons containing merge sites into
        a ZIP archive.

        Parameters
        ----------
        gt_graphs : Dict[str, LabeledGraph]
            Graphs built from ground truth SWC files.
        fragment_graphs : Dict[str, FragmentsGraph]
            Graphs built from skeletons obtained from a predicted segmentation.
        zip_writer : zipfile.ZipFile
            Open ZIP file handle used to write SWC data.
        """
        # Save ground truth skeletons
        for key in self.merge_sites["GroundTruth_ID"].unique():
            gt_graphs[key].to_zipped_swc(zip_writer)

        # Save fragments
        for key in self.fragments_with_merge:
            fragment_graphs[key].to_zipped_swc(zip_writer)
