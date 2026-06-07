"""
Created on Mon June 7 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing ground truth graphs and their intersecting fragments.

"""

from scipy.ndimage import binary_dilation

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os


def save_mips(graph_list, output_dir, filename, dilation=16):
    """
    Generates and saves maximum-intensity projections (MIPs) for a collection
    of graphs.

    Parameters
    ----------
    graph_list : List[SkeletonGraph]
       Graphs to rasterize.
    output_dir : str
        Directory in which to save the output image.
    filename : str
        Base filename used for the saved image and figure title.
    dilation : int, optional
        Dilation radius applied during graph rasterization. Default is 16.
    """
    _, shape = _get_combined_bbox(graph_list)
    mip_xy, mip_xz, mip_yz = _rasterize_graphs(graph_list, dilation)
    _plot_and_save_mips(mip_xy, mip_xz, mip_yz, output_dir, filename)


def _rasterize_graphs(graph_list, dilation):
    colors = _get_colors()
    struct2d = np.ones((dilation,) * 2, dtype=bool)
    min_voxel, shape = _get_combined_bbox(graph_list)

    mip_xy = np.ones((shape[1], shape[2], 3), dtype=float)
    mip_xz = np.ones((shape[0], shape[2], 3), dtype=float)
    mip_yz = np.ones((shape[0], shape[1], 3), dtype=float)

    cc_idx = 0
    for graph in graph_list:
        shifted_voxels = graph.node_voxel - min_voxel
        for cc_nodes in nx.connected_components(graph):
            color = np.array(colors[cc_idx % len(colors)])
            cc_voxels = shifted_voxels[list(cc_nodes)]
            z, y, x = cc_voxels[:, 0], cc_voxels[:, 1], cc_voxels[:, 2]
            _paint_projections(
                mip_xy, mip_xz, mip_yz, z, y, x, color, struct2d, dilation
            )
            cc_idx += 1
    return mip_xy, mip_xz, mip_yz


def _paint_projections(
    mip_xy, mip_xz, mip_yz, z, y, x, color, struct2d, dilation
):
    for (a, b), mip in [
        ((y, x), mip_xy),
        ((z, x), mip_xz),
        ((z, y), mip_yz),
    ]:
        local, slc = _make_dilated_local(a, b, mip.shape, struct2d, dilation)
        mip[slc][local] = color


def _make_dilated_local(a, b, mip_shape, struct2d, pad):
    a_min, b_min = a.min(), b.min()
    a_max, b_max = a.max(), b.max()

    # Build padded local array and dilate
    local = np.zeros(
        (a_max - a_min + 1 + 2 * pad, b_max - b_min + 1 + 2 * pad), dtype=bool
    )
    local[a - a_min + pad, b - b_min + pad] = True
    local = binary_dilation(local, structure=struct2d)

    # Compute MIP slice bounds clamped to MIP shape
    a_start = max(a_min - pad, 0)
    b_start = max(b_min - pad, 0)
    a_end = min(a_max + pad + 1, mip_shape[0])
    b_end = min(b_max + pad + 1, mip_shape[1])

    # Trim local to match clamped slice
    local_a0 = a_start - (a_min - pad)
    local_b0 = b_start - (b_min - pad)
    local_trimmed = local[
        local_a0: local_a0 + (a_end - a_start),
        local_b0: local_b0 + (b_end - b_start),
    ]

    return local_trimmed, np.s_[a_start:a_end, b_start:b_end]


def _get_combined_bbox(graph_list):
    all_voxels = np.vstack([graph.node_voxel for graph in graph_list])
    min_voxel = all_voxels.min(axis=0)
    shape = tuple((all_voxels.max(axis=0) - min_voxel) + 1)
    return min_voxel, shape


def _plot_and_save_mips(mip_xy, mip_xz, mip_yz, output_dir, filename):
    """
    Plots and saves orthogonal maximum-intensity projections (MIPs).

    Parameters
    ----------
    mip_xy : ndarray
        Maximum-intensity projection in the XY plane.
    mip_xz : ndarray
        Maximum-intensity projection in the XZ plane.
    mip_yz : ndarray
        Maximum-intensity projection in the YZ plane.
    output_dir : str
        Directory in which to save the output image.
    filename : str
        Base filename used for the figure title and output file.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mip_xy)
    axes[0].set_title("XY")
    axes[1].imshow(mip_xz)
    axes[1].set_title("XZ")
    axes[2].imshow(mip_yz)
    axes[2].set_title("YZ")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(filename)
    plt.tight_layout()

    path = os.path.join(output_dir, f"{filename}_mips.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _get_colors():
    """
    Gets a predefined palette of RGB colors.

    Returns
    -------
    List[Tuple[float]]
        RGB values in the range [0, 1].
    """
    # Create hex color list
    colors = list()
    colors.extend(["#1F78B4", "#66B2D9", "#2C5496", "#6495ED"])  # Blues
    colors.extend(["#FF7F0E", "#FFA651", "#D65F1A", "#F39238"])  # Oranges
    colors.extend(["#2CA02C", "#70C270", "#1A6B1A", "#4FAE61"])  # Greens
    colors.extend(["#D62728", "#FF6B6B", "#9E0000", "#F05945"])  # Reds
    colors.extend(["#9467BD", "#C5A6E4", "#5F3594", "#AE7DD4"])  # Purples
    colors.extend(["#8C564B", "#C49184", "#A67263"])  # Browns
    colors.extend(["#E377C2", "#B23D93", "#F095D2"])  # Pinks
    colors.extend(["#159EA5", "#63D1D1", "#0C6B7A", "#33B3B2"])  # Teals
    colors.extend(["#BCBD22", "#E7E76C", "#899400", "#D1D84C"])  # Yellows
    colors.extend(["#668C5E", "#435D64", "#87A784", "#577B79"])  # Olive

    # Convert to RGB
    return [
        tuple(int(h[i: i + 2], 16) / 255 for i in (1, 3, 5)) for h in colors
    ]
