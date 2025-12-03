#!/usr/bin/env python3

"""
    Processes multiple SNOWWI scenes using offset range.

    Author: Marc Closa Tarres
    Date: 2025-11-21
    Version: v0

    Changelog:
        - v0: Nov 21, 2025 - Initial Version - MCT
"""

import configparser
import glob
import os
import re
import subprocess
import sys

import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentTypeError
from pprint import PrettyPrinter
from time import sleep, time

from snowwi_tools.utils import natural_keys

pp = PrettyPrinter()

def parse_args():
    
    arg_parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
    )

    arg_parser.add_argument('working_directory',
                            help='Working directory for I/O.')
    arg_parser.add_argument('--yaw', '-y', required=True, nargs='+', type=float,
                            help="Range and step of yaw offsets for iteration space and step, in degrees. E.g., iterate from 0 to 3 degrees in steps of 0.5 degrees: -y 0.0 3.0 0.5. If step not specified, defaults to 1 degree. If only one number specified, uses that fixed offset.")
    arg_parser.add_argument('--pitch', '-p', required=True, nargs='+', type=float,
                            help="Range and step of pitch offsets for iteration space and step, in degrees. E.g., iterate from 0 to 3 degrees in steps of 0.5 degrees: -p 0.0 3.0 0.5. If step not specified, defaults to 1 degree. If only one number specified, uses that fixed offset.")
    band_choices = ['low', 'high', 'c']
    arg_parser.add_argument('--band', '-b', required=True, choices=band_choices, type=str.lower,
                            help="Band of choice.")
    mode_choices = ['snowwi', 'kasi']
    arg_parser.add_argument('--mode', '-m', choices=mode_choices, type=str.lower, default='snowwi',
                            help="Mode of operation/instrument. Default: 'snowwi'.")
    channel_choices = ['0', '1', '2', '3', 'all']
    arg_parser.add_argument('--channel', '-c', nargs="+", choices=channel_choices, default="all",
                            help="Channels to process.")
    omap_help = """\
    Defines a custom output map. Modifies azmcomp to match the desired
    output map parameters:
        Number of Azimuth Samples = 
        First Azimuth Sample (m) = 
        Azimuth Spacing (m) = 

        E.g., if -om 6000 -3000 1.0, then
        Number of Azimuth Samples = 6000
        First Azimuth Sample (m) = -3000.0
        Azimuth Spacing (m) = 1.0
    """

    arg_parser.add_argument('--output-map', '-om', nargs=3,
                            metavar=('NSAMPLES', 'FIRST_AZ', 'DAZ'),
                            help=omap_help)
    arg_parser.add_argument("--patch-size", "-ps", type=int, default=1000,
                            help="Patch size for azmcomp step. Default: 1000. Reduce if memory issues.")
    
    args = arg_parser.parse_args()

    args.working_directory = os.path.abspath(args.working_directory)
    if 'all' in args.channel:
        args.channel = ['0', '1', '2', '3']

    if args.output_map is not None:
        try:
            nsamples_str, first_az_str, daz_str = args.output_map
            nsamples = int(nsamples_str)
            first_az = int(first_az_str)
            daz = float(daz_str)
            args.output_map = (nsamples, first_az, daz)
        except ValueError:
            arg_parser.error("--output-map expects: NSAMPLES FIRST_AZ DAZ = int int float")

    return args

def calculate_steps(arg_params):

    assert (len(arg_params) >= 1) & (len(arg_params) <= 3), "Invalid parameters for iteration."
    start = arg_params[0]
    try:
        stop = arg_params[1]
    except IndexError:
        stop = start
    try:
        step = arg_params[2]
    except IndexError:
        step = 1.0
    n = int(abs(stop - start)/step + 1)
    print(start, stop, step, n)

    return np.linspace(start, stop, n), step


def update_config_field(path, key, new_value):
    try:
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#]*)(.*)$")
        out_lines = []
        found = False

        with open(path, "r") as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    # Replace only the value before any comment
                    line = f"{m.group(1)}{new_value}{m.group(3)}\n"
                    found = True
                out_lines.append(line)

        # Optionally: if the key was never found, treat as error
        if not found:
            return -1

        with open(path, "w") as f:
            f.writelines(out_lines)

        return 0

    except Exception:
        return -1


def main():
    
    args = parse_args()
    pp.pprint(args)

    # Define iteration parameters
    
    # Yaw
    y_offsets, y_step = calculate_steps(args.yaw)
    print(f"Iterating YAW offsets from {y_offsets[0]:.3f} degrees to {y_offsets[-1]:.3f} degrees every {y_step:.3f} degrees.")
    print(f"Iteration steps: {y_offsets}.")

    # Pitch
    p_offsets, p_step = calculate_steps(args.pitch)
    print(f"Iterating PITCH offsets from {p_offsets[0]:.3f} degrees to {p_offsets[-1]:.3f} degrees every {p_step:.3f} degrees.")
    print(f"Iteration steps: {p_offsets}.")

    path_to_config = os.path.join(args.working_directory, "config", args.band)
    print(path_to_config)
    cfg_preprocess_tx = os.path.join(path_to_config, f'preprocess_tx.cfg')
    print(cfg_preprocess_tx)

    pp.pprint(glob.glob(path_to_config+'/*.cfg'))

    for y_off in y_offsets:
        print(f'\n\nYAW OFFSET: {y_off:3f} deg.')
        for p_off in p_offsets:
            print(f'\n\nPITCH OFFSET: {p_off:3f} deg.')

            preprocess_dir = f'preprocess_y{y_off}_p{p_off}'
            azmcomp_dir = f'azmcomp_y{y_off}_p{p_off}'

            for ch in args.channel:
                print(f"\n\n Preparing Channel {ch}...")

                cfg_preprocess = os.path.join(path_to_config, f'preprocess_{ch}.cfg')
                print()
                print(f"Config file to modify:")
                print(f"    {cfg_preprocess}")

                ypr = f"{y_off:.3f}, {p_off:.3f}, 0.0"

                success = update_config_field(
                    cfg_preprocess,
                    "IMU to Body Angles (degrees)",
                    ypr)
                
                if success == -1:
                    print(f"Error modifying config file: {cfg_preprocess}. Skipping iteration...")
                    continue
                
                success = update_config_field(
                    cfg_preprocess_tx,
                    "IMU to Body Angles (degrees)",
                    ypr)
                
                if success == -1:
                    print(f"Error modifying config file: {cfg_preprocess_tx}. Skipping iteration...")
                    continue
                
            # For preprocess
            # preprocess.sh working_dir out_dir mode band [ch [ch]]
            
            preprocess_cmd = [
                'preprocess.sh',
                args.working_directory,
                preprocess_dir,
                args.mode,
                args.band,
                " ".join(args.channel)
            ]
            print("Running preprocessing.sh... Command to use:")
            print(preprocess_cmd)
            sleep(1)

            sys.stdout.flush()

            preprocess_p = subprocess.Popen(preprocess_cmd)
            preprocess_p.wait()
            print("\n\nPreprocessing done.")
            sys.stdout.flush()
            sleep(1)


            # Now modify the azmcomp config for the output map (if so)
            if args.output_map:
                # TODO: This is actually a bandaid to avoid iterating, bc snowwi_azmcomp_<ch>.cfg are links to the template file.
                cfg_azmcomp = os.path.join(path_to_config, "azmcomp_template.cfg")

                success = update_config_field(
                    cfg_azmcomp,
                    "Number of Azimuth Samples",
                    args.output_map[0])
                
                if success == -1:
                    print(f"Error modifying config file: {cfg_azmcomp}. Skipping iteration...")
                    continue
                
                success = update_config_field(
                    cfg_azmcomp,
                    "First Azimuth Sample (m)",
                    args.output_map[1])
                
                if success == -1:
                    print(f"Error modifying config file: {cfg_azmcomp}. Skipping iteration...")
                    continue
                
                success = update_config_field(
                    cfg_azmcomp,
                    "Azimuth Spacing (m)",
                    args.output_map[2])
                
                if success == -1:
                    print(f"Error modifying config file: {cfg_azmcomp}. Skipping iteration...")
                    continue

            # For azmcomp
            # azmcomp.sh working_dir out_dir preprocess_dir mode [band] n_patch swap_channels [ch [ch]]
            azmcomp_cmd = [
                "azmcomp.sh",
                args.working_directory,
                azmcomp_dir,
                preprocess_dir,
                args.mode,
                args.band,
                str(args.patch_size),
                "False",
                " ".join(args.channel)
            ]

            print("Running azmcomp.sh... Command to use:")
            print(azmcomp_cmd)
            sleep(1)

            sys.stdout.flush()

            azmcomp_p = subprocess.Popen(azmcomp_cmd)
            azmcomp_p.wait()
            print("\n\Azimuth compression done.")
            sys.stdout.flush()
            sleep(1)



                
       

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")
    
