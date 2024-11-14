"""
    Real aperture field processing for SNOWWI. 

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 14, 2024 - MCT
"""

import numpy as np
import matplotlib.pyplot as plt

import argparse
import glob
import os
import time

from multiprocessing import Pool

from pprint import PrettyPrinter
pp = PrettyPrinter()

from snowwi.file_handling import read_and_reshape, list_files_from_dir, combine_results, read_and_compress_local
from snowwi.params import get_band_params_4x2, get_band_params
from snowwi.ra.radar_processing import filter_and_compress
from snowwi.utils import make_chirp_dict

from scipy.constants import speed_of_light

# Some hardcoded params I don't like...

# Data handling stuff
N = 64000
HEADER_LENGTH = 48
SKIP_SAMPLES = 2450
SKIP_SAMPLES = 0
SWATH_START = 19750

# DSP stuff
PRF = 1e3
PRT = PRF ** -1
SAMP_FREQ = 983.064e6
SAMP_TIME = SAMP_FREQ ** -1
PULSE_LENGTH = 11e-6

# Plotting stuff
CLIMS = {
    0: [90, 130],
    2: [60, 90],
    1: [90, 130],
    3: [60, 90],
}

def parse_args():
    
    arg_parser = argparse.ArgumentParser(add_help=True)

    arg_parser.add_argument('path', help='/path/to/data/to/process')
    arg_parser.add_argument('flightline', help='Fligtline ID (YYYYMMDDThhmmss)')
    arg_parser.add_argument('--channel', '-c',help='DAQ system channel',
                            type=int, nargs='+', required=True)
    arg_parser.add_argument('--band', '-b',
                            help='SNOWWI band: low, high, c, all',
                            default='all',
                            nargs='+')
    arg_parser.add_argument('--num-files', '-nf',
                            help='Number of files to process. If number is larger than number of files collected, process all. If not specified, process 100 files.',
                            default=100, type=int)
    arg_parser.add_argument('--process-from', '-pf',
                            help='First file to process. If not specified, start from first file',
                            default=0)
    arg_parser.add_argument('--samples-per-pulse', '-n0',
                            help='Number of radar samples per pulse.',
                            type=int, default=100000)
    arg_parser.add_argument('--ettus', '-ettus', action='store_true',
                            help='If flag used, will range compress data using Ettus frequency plan.',
                            default=False)
    
    print("\n\nDEFAULT NUMBER OF SAMPLES PER PULSE SET TO NEW NUMBER OF SAMPLES\n\n")

    args = arg_parser.parse_args()

    # Deal with args that need special treatment
    path_to_flightline = os.path.join(args.path, args.flightline)
    args.band = [b.lower() for b in args.band]

    # Assertions before continuing
    path_assert_msg = "Invalid path to data."
    assert os.path.isdir(args.path), path_assert_msg

    fl_assert_msg = "Invalid flightline."
    assert os.path.isdir(path_to_flightline), fl_assert_msg

    band_list = ['high', 'low', 'c', 'all']
    band_assert_msg = "Invalid band. Please choose from: low, high, c, all"
    assert set(args.band).issubset(set(band_list)), band_assert_msg

    ch_list = [0, 1, 2, 3]
    channel_assert_msg = "Invalid channel. Please choose from: 0, 1, 2, 3"
    assert set(args.channel).issubset(set(ch_list)), channel_assert_msg

    return arg_parser.parse_args(), path_to_flightline


def main():
    
    args, path_to_flightline = parse_args()
    print(args)
    print()

    print(f"Path to chosen flightline: {path_to_flightline}")
    print()

    if args.band == 'all':
        bands_to_process = ['low', 'high', 'c']
    else:
        bands_to_process = args.band
    print(f"Bands to process: {bands_to_process}")

    if args.channel == 'all':
        channels_to_process = [0, 1, 2, 3]
    else:
        channels_to_process = args.channel
    print(f"Channels to process: {channels_to_process}")
    print()

    time.sleep(2)

   # I need to think a little better how to deal with that

    for i, chan in enumerate(channels_to_process):
        print(f"Processing channel {chan}...\n")

        path_to_channel = os.path.join(path_to_flightline, f'chan{chan}')
        print(f"Path to channel to process: {path_to_channel}")
        
        filelist, ts_from_file = list_files_from_dir(
            path_to_channel,
            args.process_from,
            args.process_from + args.num_files)
        print('\n')
        
        pp.pprint(f"Files to process: {filelist}")
        print('\n')

        """
            results = p.starmap(
                read_and_compress_local,
                    [(file, N0, header_length,
                    skip_samples, N0,
                    chirp_dict, cfg['Range Compression']['Data window'],
                    filter)
                    for file in files_to_process])
        """
        # We read all three bands within the channel
        with Pool(processes=os.cpu_count()) as p:
            tmp_results = p.starmap(
                read_and_reshape,
                    [(file, N, HEADER_LENGTH,
                      SKIP_SAMPLES, N)
                     for file in filelist]
            )

        raw_data, ts_from_file, headers = combine_results(tmp_results)
        del(tmp_results)
        print(raw_data.shape)

        for j, band in enumerate(bands_to_process):
            print(f"\n\nProcessing band {band}...\n")

            if args.ettus:
                print('Using Ettus baseband frequency.\n')
                band_params = get_band_params(band, chan)
            else:
                print('Using 4x2 baseband frequency.\n')
                band_params = get_band_params_4x2(band, chan)

            pp.pprint(f'Band parameters: {band_params}')
            print()

            chirp_dict = make_chirp_dict(
                band_params['f0'],
                band_params['f_l'],
                band_params['f_h'],
                PULSE_LENGTH,
                band_params['chirp_type'],
                SAMP_FREQ
            )

            pp.pprint(f'\Reference chirp parameters: {chirp_dict}')

            # Split in chunks to paralellize BPF
            chunks = np.array_split(raw_data, os.cpu_count())
            print(len(chunks), chunks[0].shape, chunks[-1].shape)

            with Pool(processes=os.cpu_count()) as p:
                tmp_results = p.starmap(
                    filter_and_compress,
                    [(chunk, band_params, chirp_dict) for chunk in chunks]
                )

            # Reconstruct the RC matrix
            samp_ref = tmp_results[0].shape[1]
            sh_assert_msg = 'Invalid number of samples. Error in reshaping.'
            assert all([ch.shape[1] == samp_ref for ch in tmp_results]), sh_assert_msg
            
            rc = np.vstack([tmp for tmp in tmp_results])
            print(rc.shape)

            # This could be condensed into a single function but as of now
            # it'll stay like this...

            flight_time = [
                0,
                rc.shape[0] * PRT
            ]

            max_samp = np.argmax(abs(rc[3]))
            print(max_samp)

            sr = np.array([
                -max_samp * SAMP_TIME,
                (rc.shape[1] - max_samp) * SAMP_TIME
            ]) * speed_of_light / 2

            im_ratio = rc.shape[0]/rc.shape[1]
            print(im_ratio)

            plt.figure()
            plt.imshow(20*np.log10(abs(rc)), cmap='gray', origin='upper',
                       vmin=CLIMS[chan][0], vmax=CLIMS[chan][1],
                       extent=[sr[0], sr[1], flight_time[-1], flight_time[0]],
                       aspect=1/(5*im_ratio))
            plt.title(f"RC image - Band: {band} - Channel: {chan} - ({args.process_from}, {args.process_from + args.num_files})")
            plt.colorbar()
            plt.xlabel('Slant range (m)')
            plt.ylabel('Flight time (s)')
            plt.draw()
            plt.pause(0.001)
            time.sleep(1)

    plt.show()





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")