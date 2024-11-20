#!/usr/bin/env python3

"""
    Generates antenna pattern for SNOWWI antennas.

    Author: Marc Closa Tarres
    Date: 2024-11-19
    Version: 0.1
    
    Changelog:
        - v 0.1: Initial version (MCT)
"""

import matplotlib.pyplot as plt
import numpy as np

import argparse


def parse_args():
    """Manages script arguments from command line when running the script."""

    arg_parser = argparse.ArgumentParser(add_help=True)

    arg_parser.add_argument('--azimuth', '-a', required=True, type=float,
                            help="Antenna size in the azimuth direction, in m.")
    arg_parser.add_argument('--elevation', '-e', required=True, type=float,
                            help="Antenna size in the elevation direction, in m.")
    arg_parser.add_argument('--frequency', '-f', required=True, type=float,
                            help="Antenna frequency for the pattern creation.")
    arg_parser.add_argument('--output', '-o', default='ant_pattern.json',
                            help='Filename suffix to save the pattern. Convention as follows: {azimuth, elevation}_<filename>.json')

    args = arg_parser.parse_args()

    return args


def main():
    args = parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")
