"""
Utility library for SNOWWI processing.

Author: Marc Closa Tarres (MCT)

Changelog:
    - v0: No13, 2024 - MCT
"""
import numpy as np
import pandas as pd

import re

def atoi(text): return int(text) if text.isdigit() else text

def natural_keys(text): return [atoi(c) for c in re.split(r'(\d+)', text)]

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
        df = pd.read_excel(xls, sheet_name, skiprows=1, usecols=np.arange(maxcols))
    print(df.keys())
    return df


def set_rcParams(plt, tex=False):
    # Set the RC params
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['lines.linewidth'] = .7
    if tex:
        plt.rcParams['text.usetex'] = True