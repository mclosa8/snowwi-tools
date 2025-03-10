#!/usr/bin/env python3

"""
    Real aperture field processing for SNOWWI. 

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0.1

    Changelog:
        - v0: Nov 14, 2024 - MCT
        - v0.1: March 7, 2025 - Added magnitude multilooking and output to binary - MCT
"""
import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import speed_of_light

from snowwi_tools.params import get_band_params_4x2, get_band_params_ettus
from snowwi_tools.utils import make_chirp_dict, kn_to_mps, set_rcParams

from snowwi_tools.lib.file_handling import read_and_reshape
from snowwi_tools.lib.file_handling import list_files_from_dir
from snowwi_tools.lib.file_handling import combine_results
from snowwi_tools.lib.file_handling import make_if_not_a_dir
from snowwi_tools.lib.file_handling import to_binary
from snowwi_tools.lib.file_handling import write_ann_file

from snowwi_tools.ra.data_handling import average_submatrix

from snowwi_tools.ra.radar_processing import filter_and_compress
from snowwi_tools.ra.radar_processing import range_loss_correct

import argparse
import glob
import os
import time

from multiprocessing import Pool

from pprint import PrettyPrinter
pp = PrettyPrinter()

set_rcParams(plt)

# Some hardcoded params I don't like...

# Data handling stuff
N = 64000
N = 100000
HEADER_LENGTH = 4
SKIP_SAMPLES = 2450
SKIP_SAMPLES = 0
SWATH_START = 19750

# DSP stuff
PRF = 1e3
PRT = PRF ** -1
SAMP_FREQ = 983.064e6
SAMP_FREQ = 1.2288e9
SAMP_TIME = SAMP_FREQ ** -1
PULSE_LENGTH = 11e-6
# PULSE_LENGTH = 20e-6

# Plotting stuff # Even channels Co-pol, Odd channels X-pol
CLIMS = {
    0: [75, 120],
    1: [60, 90],
    2: [75, 120],
    3: [60, 90],
}
CLIMS = {
    0: [75, 120],
    1: [75, 120],
    2: [75, 120],
    3: [75, 120],
}
CLIMS = {
    0: [75, 110],
    1: [60, 110],
    2: [75, 110],
    3: [60, 110],
}


def parse_args():

    arg_parser = argparse.ArgumentParser(add_help=True)

    arg_parser.add_argument('path', help='/path/to/data/to/process')
    arg_parser.add_argument(
        'flightline', help='Fligtline ID (YYYYMMDDThhmmss)')
    arg_parser.add_argument('--channel', '-c', help='DAQ system channel',
                            type=int, nargs='+', required=True)
    arg_parser.add_argument('--band', '-b',
                            help='SNOWWI band: low, high, c, all',
                            default='all',
                            nargs='+', required=True)
    arg_parser.add_argument('--pulse-length', '-pl', help='Chirp length in s',
                            default=16.5e-6, type=float)
    arg_parser.add_argument('--num-files', '-nf',
                            help='Number of files to process. If number is larger than number of files collected, process all. If not specified, process 100 files.',
                            default=100, type=int)
    arg_parser.add_argument('--process-from', '-pf',
                            help='First file to process. If not specified, start from first file',
                            default=0, type=int)
    arg_parser.add_argument('--samples-per-pulse', '-n0',
                            help='Number of data samples per pulse. Default: 100_000',
                            type=int,
                            default=100000)
    arg_parser.add_argument('--header-samples', '-hs',
                            help="Sets the number of header samples per pulse. Default: 4.",
                            default=4,
                            type=int)
    arg_parser.add_argument('--save-to', '-st',
                            help="Output path. Default: cwd/output/flightline",
                            default=os.getcwd())
    arg_parser.add_argument('--save_fig', '-sf',
                            help="Saves figure to output path if set. Default: False",
                            action='store_true',
                            default=False)
    arg_parser.add_argument("--save_binary", "-sb",
                            help="Saves binary AND annotation file to output path. Default: False",
                            action='store_true',
                            default=False)
    arg_parser.add_argument('--no-plot', '-np',
                            help="If flag active, does not show plot.",
                            action='store_true',
                            default=False)
    arg_parser.add_argument('--ettus', '-ettus', action='store_true',
                            help='If flag used, will range compress data using Ettus frequency plan.',
                            default=False)
    arg_parser.add_argument('--interferometry', '-itf',
                            action='store_true', default=False)
    arg_parser.add_argument('--multilook', '-ml',
                            help="Number of looks for output resolution (azimuth_looks, range_looks). Will multilook magnitude. #TODO: Implement complex multilooking.",
                            nargs=2,
                            type=int)
    arg_parser.add_argument("--multilook-to-res", '-mltr',
                            help="Define the output resolution, and subsequently calculates the number of looks in (azimuth_looks, range_looks). Will multilook magnitude. #TODO: Implement complex multilooking.",
                            nargs=2,
                            type=float)
    arg_parser.add_argument("--platform-velocity", '-vel',
                            help="Sets the platform velocity in knots for azimuth resolution calculations. Will be automatically converted to m/s. Default: 100 kn.",
                            default=100,
                            type=float)
    arg_parser.add_argument("--calibration-factor", '-cal',
                            help="Sets calibration factor, in dB. Will be automatically converted to linear scale. Will be applied as (SLC)^2 * <cal_factor>.",
                            default=0,
                            type=float)
    arg_parser.add_argument('--colorlimits', '-clim',
                            help=f"Overwrites the default colorlimits, in log units. Default: {CLIMS}.",
                            nargs=2)
    arg_parser.add_argument('--range-correction', '-rcorr',
                            help='Corrects for range losses, at the expense of increasing noise floor. Default: False.',
                            action='store_true',
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

    save_to_msg = "Invalid output path."
    assert os.path.isdir(args.save_to), save_to_msg
    args.save_to = os.path.join(
        os.path.abspath(args.save_to),
        "outputs",
        f"{args.flightline}"
    )
    args.calibration_factor = 10**(args.calibration_factor / 10)
    print(args.calibration_factor)
    args.platform_velocity = kn_to_mps(args.platform_velocity)

    return args, path_to_flightline


def main():

    args, path_to_flightline = parse_args()
    print(args)
    print()

    print(f"Path to chosen flightline: {path_to_flightline}")
    print()

    if 'all' in args.band:
        bands_to_process = ['low', 'high', 'c']
    else:
        bands_to_process = args.band
    print(f"Bands to process: {bands_to_process}")
    time.sleep(2)

    if 'all' in args.channel:
        channels_to_process = [0, 1, 2, 3]
    else:
        channels_to_process = args.channel
    print(f"Channels to process: {channels_to_process}")
    print()

    if args.interferometry:
        if 0 in args.channel or 2 in args/{args.flightline}.channel:
            channels_to_process = [0, 2]
        elif 1 in args.channel or 3 in args.channel:
            channels_to_process = [1, 3]
        print(
            f'Doing interferometry. Processing channels {channels_to_process}')
        time.sleep(2)

    if args.save_to:
        make_if_not_a_dir(args.save_to)

    print(f"Pulse length used: {args.pulse_length}")

    if (args.save_fig):
        print(f"Saving figure to: {args.save_to}")
    else:
        print('Not saving figure.')

    if (args.save_binary):
        print(f"Saving binary and annotation file to: {args.save_to}")
    else:
        print('Not saving binary.')
    print()

    print(args.no_plot)

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

        daq_params = get_band_params_4x2('daq')

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
                [(file, daq_params['data_samps'], daq_params['header_samps'],
                  SKIP_SAMPLES)
                 for file in filelist]
            )

        raw_data, ts_from_file, headers = combine_results(tmp_results)
        del (tmp_results)
        print(raw_data.shape)

        for j, band in enumerate(bands_to_process):
            print(f"\n\nProcessing band {band}...\n")

            if args.ettus:
                print('Using Ettus baseband frequency.\n')
                band_params = get_band_params_ettus(band)
            else:
                print('Using 4x2 baseband frequency.\n')
                band_params = get_band_params_4x2(band)

            pp.pprint(f'Band parameters: {band_params}')
            print()

            chirp_dict = make_chirp_dict(
                band_params['f0'],
                band_params['f_l'],
                band_params['f_h'],
                args.pulse_length,
                band_params['chirp_type'],
                daq_params['fs']
            )

            pp.pprint(f'\Reference chirp parameters: {chirp_dict}')

            az_res = args.platform_velocity / daq_params['prf']
            rg_res = speed_of_light / daq_params['fs'] / 2
            original_resolution = (az_res, rg_res)
            print(f"Original data resolution (rg, az): {original_resolution}")
            final_resolution = original_resolution

            if args.multilook_to_res:
                print(f"Multilooking to resolution: {args.multilook_to_res}")
                ml_assert_msg = f"Invalid resolution. Final resolution must be larger than original resolution: {original_resolution}"
                assert (args.multilook_to_res[0] > az_res) or (
                    args.multilook_to_res[1] > rg_res), ml_assert_msg
                az_looks = int(np.round(args.multilook_to_res[0] / az_res))
                rg_looks = int(np.round(args.multilook_to_res[1] / rg_res))
                args.multilook = (az_looks, rg_looks)

            if args.multilook:
                print(f"Number of looks (az, rg): {args.multilook}")
                final_resolution = (
                    az_res * args.multilook[0],  # Azimuth
                    rg_res * args.multilook[1]   # Range
                )
                print(
                    f"Output resolution (rg, az): ({final_resolution[0]:.2f}, {final_resolution[1]:.2f}) m.")

            # Split in chunks to paralellize BPF
            chunks = np.array_split(raw_data, os.cpu_count())
            print(len(chunks), chunks[0].shape, chunks[-1].shape)
            # TODO - think about how to do the multilook on the go, elegantly
            with Pool(processes=os.cpu_count()) as p:
                tmp_results = p.starmap(
                    filter_and_compress,
                    [(chunk, band_params, chirp_dict, args.multilook)
                     for chunk in chunks]
                )

            # Reconstruct the RC matrix
            samp_ref = tmp_results[0].shape[1]
            sh_assert_msg = 'Invalid number of samples. Error in reshaping.'
            assert all(
                [ch.shape[1] == samp_ref for ch in tmp_results]), sh_assert_msg

            rc = np.vstack([tmp for tmp in tmp_results])
            print(rc.shape)

            if args.interferometry and chan == channels_to_process[0]:
                ref = rc + 0
                print('Reference image stored.')
            else:
                print('No reference image stored.')

            # This could be condensed into a single function but as of now
            # it'll stay like this...

            at = [
                0,
                rc.shape[0] * final_resolution[0]
            ]
            print(f"AT: {at}")

            # max_samp = np.argmax(abs(rc[3]))
            # print(max_samp)

            sr = np.array([
                0,
                rc.shape[1]
            ]) * final_resolution[1]
            print(f"SR: {sr}")

            im_ratio = rc.shape[0]/rc.shape[1]
            print(im_ratio)

            if args.range_correction:
                print("Correcting for range loss...")
                rc = range_loss_correct(rc)
                print("Correction completed.")

            if args.colorlimits:
                vmin = args.colorlimits[0]
                vmax = args.colorlimits[1]
            else:
                vmin = CLIMS[chan][0]
                vmax = CLIMS[chan][1]

            print(f"Applying calibration factor: {args.calibration_factor}")
            as_ratio = abs((sr[1] - sr[0]) /
                           (at[-1] - at[0]))
            fig, ax = plt.subplots(figsize=(16, 9))
            im = ax.imshow(10*np.log10((abs(rc)**2) * args.calibration_factor), cmap='gray', origin='upper',
                           vmin=vmin, vmax=vmax,
                           extent=[sr[0], sr[-1],
                                   at[-1], at[0]],
                           aspect='equal')
            plt.title(
                f"RC image - Band: {band} - Channel: {chan} - ({args.process_from}, {args.process_from + args.num_files})")
            plt.xlabel('Slant range (m)')
            plt.ylabel('Along track (m)')

            # Adjust colorbar to plot extent
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            # Adjust size and padding
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax, label='(dB)')

            # plt.draw()
            # plt.pause(0.001)
            figname = f"{args.flightline}_{band}_{chan}_{args.process_from}_{args.num_files}_rc{args.range_correction}"
            if (args.save_fig):
                plt.savefig(
                    os.path.join(
                        args.save_to,
                        figname + ".png"),
                    dpi=500
                )
            if (args.no_plot):
                print("Clearing fig...")
                plt.close()
            plt.close()
            time.sleep(1)

            if args.save_binary:
                output = os.path.join(
                    args.save_to,
                    figname
                )
                to_binary(output, rc**2 * args.calibration_factor)
                write_ann_file(
                    output,
                    band,
                    chan,
                    band_params,
                    daq_params,
                    final_resolution,
                    rc.shape
                )

    if args.interferometry:
        interferogram = ref * np.conjugate(rc)

        plt.figure()
        plt.imshow(np.angle(interferogram), cmap='jet',
                   extent=[sr[0], sr[1], at[-1], at[0]],
                   aspect=.125e3)
        plt.xlabel('Slant range (m)')
        plt.ylabel('Flight time (s)')
        if (args.save_fig):
            plt.savefig(
                f"{args.save_fig}/{args.flightline}/{args.flightline}_{band}_{channels_to_process}_{args.process_from}_{args.num_files}_interf.png", dpi=500)


if __name__ == "__main__":
    try:
        now = time.time()
        main()
        print(f"Elapsed: {time.time() - now} s.")
    except KeyboardInterrupt:
        print("Script interrupted.")
