#!/usr/bin/env python3
import argparse
import configobj
import os

import numpy as np

from pprint import PrettyPrinter
pp = PrettyPrinter()


def parse_args():
    
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('corner_location',
                            help="Corner reflector location in the SLC imagery (az, rg) pixels.")
    arg_parser.add_argument('config_file',
                            help="Configuration file for azimuth compression.")
    arg_parser.add_argument('--border', '-b',
                            help="Define border, in m, for smaller map based on CRs.",
                            default=200,
                            type=float)
    arg_parser.add_argument('--update', '-u',
                            help="Update config with a small corner-centered map",
                            action='store_true',
                            default=False)
    
    args = arg_parser.parse_args()
    args.corner_location = os.path.abspath(args.corner_location)
    args.config_file = os.path.abspath(args.config_file)

    return arg_parser.parse_args()


def main():
    
    args = parse_args()
    pp.pprint(args)

    config_ac = configobj.ConfigObj(args.config_file)
    # 2D array where rows are CRs and columns are az, rg
    crs = np.atleast_2d(np.loadtxt(args.corner_location, dtype=int, delimiter=' '))

    min_az, max_az = crs[:, 0].min(), crs[:, 0].max()

    # Calculate num samples in border according to output resolution
    d_az = config_ac['Output Map'].as_float('Azimuth Spacing (m)')

    # Border in m to samples
    border_az = int(np.ceil(args.border / d_az))

    # In samples
    az_start = min_az - border_az
    az_stop  = max_az + border_az

    # This is in m
    old_az_start = config_ac['Output Map'].as_float("First Azimuth Sample (m)")
    new_az_start = old_az_start + az_start * d_az
    config_ac['Output Map']['First Azimuth Sample (m)'] = new_az_start

    # This is in samples 
    az_samples = az_stop - az_start
    config_ac['Output Map']['Number of Azimuth Samples'] = az_samples

    print("Old corner indices:")
    print(crs)
    crs[:, 0] = crs[:, 0] - az_start
    print("New corner indices:")
    print(crs)

    print(f"First azimuth sample (m): {new_az_start}")
    print(f"Number of azimuth samples: {az_samples}")

    np.savetxt('target_indices.txt', crs, fmt='%.0f')

    if args.update:
        print("Updating config file...")
        config_ac.write()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script terinated.")