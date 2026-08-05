#!/usr/bin/env python3
"""Plot SAR scene (vertical + horizontal) for a given band and channel."""
import argparse
from pathlib import Path

import h5py as h5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from snowwi_tools.utils import set_rcParams, vertical_colorbar
from snowwi_tools.lib.file_handling import make_if_not_a_dir

set_rcParams(plt, tex=True)


def extract_date_line(filepath: Path):
    parts = filepath.parts
    for i, part in enumerate(parts):
        if part == 'processing' and i + 2 < len(parts):
            return parts[i + 1], parts[i + 2]
    raise ValueError(f"Could not find processing/DATE/LINE in path: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Plot SAR scene.')
    parser.add_argument('filepath', help='Path to the SLC .h5 file')
    parser.add_argument('--band',     required=True, help='Band name (e.g. low, high, c)')
    parser.add_argument('--channel',  required=True, type=int, help='Channel index')
    parser.add_argument('--bare',     action='store_true',
                        help='Save bare image (no axes/colorbar), 1 px per sample')
    parser.add_argument('-r', '--resolution', nargs=2, type=float, metavar=('AZ', 'RG'),
                        default=[1.0, 1.0], help='Pixel spacing in metres: az rg (default: 1.0 1.0)')
    parser.add_argument('--rg-offset', type=float, default=0.0,
                        help='Near-range slant distance offset in metres (default: 2500.0)')
    parser.add_argument('--vmin', type=float, default=-15.0, help='Colorbar lower limit dB (default: -25)')
    parser.add_argument('--vmax', type=float, default=10.0,   help='Colorbar upper limit dB (default: 0)')
    parser.add_argument('--dpi', type=int, default=500, help='Figure DPI (default: 500)')
    parser.add_argument('--pdf', action='store_true', help='Save as PDF instead of PNG')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output directory (default: cwd)')
    parser.add_argument('--vertical', action='store_true', help='Plot vertical orientation only')
    parser.add_argument('--fontsize', type=float, default=10.0, help='Label font size (default: 10)')
    parser.add_argument('--az', nargs=2, type=int, metavar=('AZ0', 'AZ1'), default=None,
                        help='Azimuth crop: start and end pixel (default: full scene)')
    parser.add_argument('--rg', nargs=2, type=int, metavar=('RG0', 'RG1'), default=None,
                        help='Range crop: start and end pixel (default: full scene)')
    args = parser.parse_args()

    filepath = Path(args.filepath).resolve()
    date, line = extract_date_line(filepath)
    test = filepath.parents[2].name

    az_res, rg_res = args.resolution
    fmt = 'pdf' if args.pdf else 'png'

    save_path = args.output.resolve() if args.output is not None else Path.cwd()
    make_if_not_a_dir(str(save_path))

    print(f'File   : {filepath}')
    print(f'Date   : {date}  Line: {line}')
    print(f'Output : {save_path}')

    # ── Load SLC ──────────────────────────────────────────────────────────────
    print('Loading SLC ...')
    with h5.File(filepath, 'r') as f:
        d = f['slc_distributed']
        N_az, N_rg = d.shape
        print(f'  Shape: {N_az} az × {N_rg} rg')
        amp = abs(d[:]).astype(np.float32)

    Z_dB = 20 * np.log10(np.maximum(amp, 1e-12))
    del amp
    print(f'Z_dB shape: {Z_dB.shape}')

    # ── Optional crop ─────────────────────────────────────────────────────────
    az0 = max(0,    args.az[0] if args.az else 0)
    az1 = min(N_az, args.az[1] if args.az else N_az)
    rg0 = max(0,    args.rg[0] if args.rg else 0)
    rg1 = min(N_rg, args.rg[1] if args.rg else N_rg)
    Z_dB = Z_dB[az0:az1, rg0:rg1]
    print(f'Crop: az=[{az0},{az1}]  rg=[{rg0},{rg1}]  shape={Z_dB.shape}')

    # ── Axis arrays ───────────────────────────────────────────────────────────
    az_axis = (np.arange(az0, az1) - N_az / 2) * az_res
    rg_axis = args.rg_offset + np.arange(rg0, rg1) * rg_res
    az_min, az_max_ = az_axis[0], az_axis[-1]
    rg_min, rg_max_ = rg_axis[0], rg_axis[-1]

    crop_tag = (f'_az{az0}-{az1}' if args.az else '') + (f'_rg{rg0}-{rg1}' if args.rg else '')

    def save_fig(fig, tag, bbox):
        fig.savefig(save_path / f'{tag}.{fmt}', bbox_inches=bbox, dpi=args.dpi)

    if args.vertical:
        # ── Vertical plot ─────────────────────────────────────────────────────
        data = Z_dB
        if args.bare:
            h, w = data.shape
            fig = plt.figure(figsize=(w / args.dpi, h / args.dpi), dpi=args.dpi)
            ax  = fig.add_axes([0, 0, 1, 1])
            ax.imshow(data, cmap='gray', aspect='auto', origin='upper',
                      vmin=args.vmin, vmax=args.vmax, interpolation='none',
                      extent=[rg_min, rg_max_, az_max_, az_min])
            ax.axis('off')
            bbox = 0
        else:
            fig, ax = plt.subplots(figsize=(5, 14))
            im = ax.imshow(data, cmap='gray', aspect='equal', origin='upper',
                           vmin=args.vmin, vmax=args.vmax,
                           extent=[rg_min, rg_max_, az_max_, az_min])
            ax.set_xlabel('Slant range (m)', fontsize=args.fontsize)
            ax.set_ylabel('Along track (m)', fontsize=args.fontsize)
            ax.tick_params(labelsize=args.fontsize)
            vertical_colorbar(ax, im, label='$\\sigma^0$ (dB)')
            fig.tight_layout()
            bbox = 'tight'
        tag = f'{date}_{line}_vert_{args.band}_{args.channel}{crop_tag}' + ('_bare' if args.bare else '')
        print('Vertical plot saved')
    else:
        # ── Horizontal plot ────────────────────────────────────────────────────
        data = Z_dB.T[:, ::-1]
        if args.bare:
            h, w = data.shape
            fig = plt.figure(figsize=(w / args.dpi, h / args.dpi), dpi=args.dpi)
            ax  = fig.add_axes([0, 0, 1, 1])
            ax.imshow(data, cmap='gray', aspect='auto', origin='upper',
                      vmin=args.vmin, vmax=args.vmax, interpolation='none',
                      extent=[az_min, az_max_, rg_max_, rg_min])
            ax.axis('off')
            bbox = 0
        else:
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(data, cmap='gray', aspect='equal', origin='upper',
                           vmin=args.vmin, vmax=args.vmax,
                           extent=[az_min, az_max_, rg_max_, rg_min])
            ax.set_xlabel('Along track (m)', fontsize=args.fontsize)
            ax.set_ylabel('Slant range (m)', fontsize=args.fontsize)
            ax.tick_params(labelsize=args.fontsize)
            vertical_colorbar(ax, im, label='$\\sigma^0$ (dB)')
            fig.tight_layout()
            bbox = 'tight'
        tag = f'{date}_{line}_horiz_{args.band}_{args.channel}{crop_tag}' + ('_bare' if args.bare else '')
        print('Horizontal plot saved')

    save_fig(fig, tag, bbox)
    plt.close(fig)


if __name__ == '__main__':
    main()
