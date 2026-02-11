#!/usr/bin/env python3

"""
    Batch processing script for all flightlines in the Synology drive.
    Uses pcap2dat_2026.py script with corresponding data format.

    Author: Marc Closa Tarres (MCT)
    Date: 2026-02-11
    Version: v0

    Changelog:
        - v0: Feb 11, 2026 - Initial version - MCT
"""

import glob
import os
import subprocess
import sys

from argparse import ArgumentParser

from snowwi_tools.utils import natural_keys

from time import sleep

from pprint import PrettyPrinter
pp = PrettyPrinter()


def parse_args():

    arg_parser = ArgumentParser()

    arg_parser.add_argument('base_directory',
                            help='Path to the day to process.')
    arg_parser.add_argument('--use-arduino-time', '-at',
                            default=False,
                            action='store_true',
                            help='Uses the arduino time from /base-directory/arduino_dump.log')
    arg_parser.add_argument('--use_gps_time', '-ut',
                            default=False,
                            action='store_true',
                            help="Uses GPS time embedded in data stream to derive pulse timestamp.")
    arg_parser.add_argument('--save-to', '-st',
                            help='Path to output directory. Default: /base_directory')
    arg_parser.add_argument('--sec-of-data', '-sd',
                            default='all',
                            help="Total time of radar data to process. Will be converted into number of PCAP files. If 'all', process all the files with data. Default: all.")
    arg_parser.add_argument('--exclude-dst-drive', '-edd',
                            help='Excludes destination_drive with corresponding number.',
                            type=int,
                            default=None)
    arg_parser.add_argument('--reverse', '-r',
                            default=False,
                            action='store_true',
                            help='Reverses list of flightlines to parse. Will parse most recent first.')
    arg_parser.add_argument('--parse-first', '-pf',
                            type=str,
                            nargs='+',
                            default=None,
                            help="First flightline or flightlines to parse. Will place them in the front of the parsing queue.")

    args = arg_parser.parse_args()
    args.base_directory = os.path.normpath(args.base_directory)
    print(args.parse_first)
    args.parse_first = [fl.split("_")[-1] for fl in args.parse_first]
    print(args.parse_first)

    return args


def move_to_front(paths, number):
    number = str(number)

    for i, p in enumerate(paths):
        parts = p.split('/')
        for part in parts:
            if part.endswith(f"_{number}"):
                return [paths[i]] + paths[:i] + paths[i+1:]
    
    return paths


def main():

    args = parse_args()
    print(args)
    tmp_path = os.path.join(os.path.abspath(args.base_directory), '**/*.pcap*')
    all_pcaps = glob.glob(
        tmp_path,
        recursive=True
    )
    all_pcaps.sort(key=natural_keys)
    print(len(all_pcaps))

    pcap_paths = [
        os.path.dirname(item) for item in all_pcaps if 'stream' in item
    ]
    pcap_paths = set(pcap_paths)
    print(f"Directories with .pcap files: {len(pcap_paths)}")


    if args.use_arduino_time:
        all_arduinos = glob.glob(
            os.path.join(args.base_directory, '**/ard*.log*'),
            recursive=True
        )
        ards_paths = [
            os.path.dirname(item) for item in all_arduinos if ('destination' in item) and ('stream' in item)
        ]
        ards_paths = set(ards_paths)
        print(f"Directories with arduino files: {len(ards_paths)}")

        final_paths = list(pcap_paths.intersection(ards_paths))
        final_paths.sort(key=natural_keys)

        print(f"Directories with arduino AND .pcap files: {len(final_paths)}")
    else:
        final_paths = list(pcap_paths)

    final_paths.sort(key=natural_keys)

    if args.reverse:
        final_paths = final_paths[::-1]

    if args.parse_first:
        for line in args.parse_first[::-1]:
            final_paths = move_to_front(final_paths, line)

    print(final_paths[0])
    print(final_paths[-1])

    for path in final_paths:
        cmd = [
            'pcap2dat_2026.py',
            path,
            '-st', args.save_to,
            '-sd', args.sec_of_data,
        ]
        if args.use_arduino_time:
            cmd.append('-at')
        if args.use_gps_time:
            cmd.append('-ut')

        print(f"Command to use:")
        print(f"    {cmd}")

        process = subprocess.Popen(cmd)
        process.wait()
        print(f"\n\nDid I wait?\n\n\n\n\n")
        sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")
