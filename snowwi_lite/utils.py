"""
Utility library for SNOWWI processing.

Author: Marc Closa Tarres (MCT)

Changelog:
    - v0: No13, 2024 - MCT
"""

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