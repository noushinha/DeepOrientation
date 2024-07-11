import os.path
import numpy as np
from matplotlib.lines import Line2D
from utils.utility_functions import *
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from config import *
import pandas as pd
from glob import glob
import yaml


def smooth_curve(points, factor=0.7):
    """ This function smooths the fluctuation of a plot by
        calculating the moving average of the points based on factor
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_slides(mol_3d_struct):
    rows = 2
    cols = 2
    fig, ax = plt.subplots(rows, cols, figsize=(20, 20))

    slice_indx = int(mol_3d_struct.shape[0]/2) - 10
    for r in range(0, rows):
        for c in range(0, cols):
            ax[r][c].imshow(gaussian_filter(mol_3d_struct[slice_indx, :, :], sigma=1), cmap="gray")
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
            ax[r][c].set_xticklabels([])
            ax[r][c].set_yticklabels([])
            slice_indx += 5

    plt.subplots_adjust(wspace=-0.05, hspace=0.05)


def midpoints(x):
    sl = ()
    for _ in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


def plot_vol(rot_matrx, mode, job_id, res_dir):
    mid = int(config_dict['patch_size'] / 2)
    plot_title = "Ground Truth"
    if mode != "gt":
        c = "r"
        plot_title = "Prediction"
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    c1, c2, c3 = 0, 0, 0
    ax.quiver(c1, c2, c3, rot_matrx[0][0], rot_matrx[1][0], rot_matrx[2][0], color="k", length=10)
    ax.quiver(c1, c2, c3, rot_matrx[0][1], rot_matrx[1][1], rot_matrx[2][1], color="g", length=10)
    ax.quiver(c1, c2, c3, rot_matrx[0][2], rot_matrx[1][2], rot_matrx[2][2], color="b", length=10)
    ax.set_xlim([-mid, mid])
    ax.set_ylim([-mid, mid])
    ax.set_zlim([-mid, mid])
    ax.set_xlabel("X")
    ax.set_xlabel("Y")
    ax.set_xlabel("Z")
    ax.grid(False)
    ax.set_title(plot_title)
    filename = f"vector_field_plot_{job_id}_{mode}"
    # write_mrc(vol, os.path.join(res_dir, f"{filename}.mrc"))
    filename_jpg = os.path.join(res_dir,  f"{filename}.jpg")
    plt.savefig(filename_jpg, format="jpg", dpi=150, bbox_inches='tight', transparent=True)


def plot_euler_vectors(mol_3d_struct, gt_rot, pd_rot, job_id, res_dir):
    plot_vol(gt_rot, "gt", job_id, res_dir)
    plot_vol(pd_rot, "prd", job_id, res_dir)

