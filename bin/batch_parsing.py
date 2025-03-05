#!/usr/bin/env python3

"""
    Batch processing script for all flightlines in the Synology drive.

    Author: Marc Closa Tarres (MCT)
    Date: 2025-02-21
    Version: v0

    Changelog:
        - v0: Feb 21, 2025 - Initial version - MCT
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
                            default='True',
                            action='store_true',
                            help='Uses the arduino time from /base-directory/arduino_dump.log')
    arg_parser.add_argument('--save-to', '-st',
                            help='Path to output directory. Default: /base_directory')
    arg_parser.add_argument('--sec-of-data', '-sd',
                            default='all',
                            help="Total time of radar data to process. Will be converted into number of PCAP files. If 'all', process all the files with data. Default: all.")
    arg_parser.add_argument('--exclude-dst-drive', '-edd',
                            help='Excludes destination_drive with corresponding number.',
                            type=int,
                            default=None)

    args = arg_parser.parse_args()
    args.base_directory = os.path.normpath(args.base_directory)

    return args


def main():

    args = parse_args()
    print(args)
    tmp_path = os.path.join(args.base_directory, '**/*.pcap*')
    print(tmp_path)
    all_pcaps = glob.glob(
        tmp_path,
        recursive=True
    )
    all_pcaps.sort(key=natural_keys)
    print(len(all_pcaps))

    pcap_paths = [
        os.path.dirname(item) for item in all_pcaps if ('destination' in item) and ('stream' in item)
    ]
    pcap_paths = set(pcap_paths)
    print(f"Directories with .pcap files: {len(pcap_paths)}")

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
    print(final_paths[0])
    print(final_paths[-1])

    if args.exclude_dst_drive != None:
        dd_excl = f"destination_drive_{args.exclude_dst_drive}"
        print(f"Excluding {dd_excl}...")
        final_paths = [
            item for item in final_paths if dd_excl not in item
        ]
        print(f"Directories to parse after excluding: {len(final_paths)}")
        print(final_paths[0])
        print(final_paths[-1])

    for path in final_paths:
        cmd = [
            'pcap_2_raw.py',
            path,
            '-st', args.save_to,
            '-sd', args.sec_of_data,
        ]
        if args.use_arduino_time:
            cmd.append('-at')

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
