#!/usr/bion/env python3

import numpy as np
import matplotlib.pyplot as plt

from gangler.lib.antenna import AntennaPattern

from snowwi_tools.utils import set_rcParams

import argparse
import json

set_rcParams(plt)

def parse_args():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "filename", help="Patterns to plot.",
        nargs='+',
    )
    arg_parser.add_argument(
        "--save-fig", help="Flag to save figure as png.",
        action='store_true', default=False
    )
    arg_parser.add_argument('-o', '--output-name', default='out.png')
    arg_parser.add_argument(
        "--show-fig", help="Flag to show figure.",
        action='store_true', default=False
    )

    return arg_parser.parse_args()


def main():
    args = parse_args()

    figure, axis = plt.subplots()
    for name in args.filename:
        print(name)
        pattern = AntennaPattern.from_file(name)
        angle_extent = np.arcsin(pattern.extent)*1.2
        angles = np.linspace(-angle_extent, angle_extent, pattern.gain.size*4)
        gain = pattern(angles)
        _3db_bw = np.degrees(angles[np.where(gain >= 0.5)[0][0]])
        axis.plot(np.rad2deg(angles), 10*np.log10(gain), label=f"{name} - 3 dB BW = {abs(2*_3db_bw):.2f} deg.")
    plt.axhline(-3, ls='--', lw=0.7, c='k')
    plt.grid(ls='--', lw=0.7)
    plt.legend(fontsize=8, loc='lower center')
    plt.xlabel('degrees')
    plt.ylabel('dB')
    plt.ylim(-40, 2)
    if args.save_fig:
        plt.savefig(args.output_name, dpi=250)
    if args.show_fig:
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")
