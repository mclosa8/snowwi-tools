"""
Utility library for SNOWWI processing.

Author: Marc Closa Tarres (MCT)

Changelog:
    - v0: Initial version - Nov 13, 2024 - MCT
    - v0.1: Added to_wavelength function - Feb 26, 2025 - MCT
"""
import numpy as np
import pandas as pd

import re

from scipy.constants import speed_of_light

from matplotlib.colors import LinearSegmentedColormap


cmyc_cmap = LinearSegmentedColormap.from_list('cmyc', ['cyan', 'magenta', 'yellow', 'cyan'])

def atoi(text): return int(text) if text.isdigit() else text


def natural_keys(text): return [atoi(c) for c in re.split(r'(\d+)', text)]


def to_wavelength(f): return speed_of_light / f


def kn_to_mps(kn):
    return kn * 0.514444


def make_chirp_dict(f0, f_l, f_h, tp, chirp_type, fs):
    return {
        'f0': f0,
        'f_l': f_l,
        'f_h': f_h,
        'tp': tp,
        'chirp_type': chirp_type,
        'fs': fs
    }


def read_spreadsheet(flightline_xl, sheet_name, maxcols=11):
    with pd.ExcelFile(flightline_xl) as xls:
        df = pd.read_excel(xls, sheet_name, skiprows=1,
                           usecols=np.arange(maxcols))
    print(df.keys())
    return df


def set_rcParams(plt, tex=False):
    # Set the RC params
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['lines.linewidth'] = .7
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["grid.linestyle"] = '--'
    plt.rcParams["grid.linewidth"]
    if tex:
        plt.rcParams['text.usetex'] = True


def average_n_rows(matrix, n):
    # Number of rows in the matrix
    N = matrix.shape[0]

    # Number of full groups of n rows
    num_full_groups = N // n

    # Number of remaining rows
    remainder = N % n

    print(N, num_full_groups, remainder, n)

    # If there are no remaining rows, we can simply reshape and take the mean
    if remainder == 0:
        return matrix.reshape(num_full_groups, n, -1).mean(axis=1)

    # If there are remaining rows, we need to handle them separately
    else:
        # Average each full group of n rows
        averaged_matrix = matrix[:num_full_groups *
                                 n].reshape(num_full_groups, n, -1).mean(axis=1)

        # Average the remaining rows and append to the result
        averaged_matrix = np.vstack(
            (averaged_matrix, matrix[-remainder:].mean(axis=0)))

    return averaged_matrix


def normalize(data):
    return (data/data.max())


def vertical_colorbar(ax, im, label=""):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """
    Adds a vertical colorbar next to an imshow plot and makes it match the plot height.

    Parameters:
        ax (matplotlib.axes.Axes): The axis where the image is plotted.
        im (matplotlib.image.AxesImage): The result from ax.imshow(...).
        label (str): Label for the colorbar (optional).
    
    Returns:
        cbar (matplotlib.colorbar.Colorbar): The created colorbar object.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label)
    return cbar


def wrap_to_half(data, value):
    return ((data + value/2) % value) - value/2


def wrap_to_full(data, value):
    return data % value


def dms_to_dd(deg, min, sec, dir):
    dd = deg + min / 60 + sec / 3600
    if dir.lower() in ['w', 's']:
        dd *= -1
    return dd