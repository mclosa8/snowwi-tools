#!/usr/bin/env python3

"""
    Calculates the doppler model and compares it to the actual Doppler signature.

    Author: Marc Closa Tarres (MCT)
    Date: 2025/02/25
    Version: v0

    Changelog:
        - v0: Initial version - Feb 25, 2025 - MCT
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
import os
import sys
import time

from snowwi_tools.doppler.doppler import generate_doppler_model
from snowwi_tools.doppler.doppler import pulse_pair_doppler

from snowwi_tools.lib.file_handling import combine_results
from snowwi_tools.lib.file_handling import make_if_not_a_dir
from snowwi_tools.lib.file_handling import read_and_reshape
from snowwi_tools.lib.novatel import read_novatel, get_attitude_dictionary
from snowwi_tools.lib.time import timestamp_from_header_4x2, timestamp_to_week_seconds

from snowwi_tools.params import get_band_params_4x2

from snowwi_tools.ra.radar_processing import filter_and_compress

from snowwi_tools.utils import make_chirp_dict
from snowwi_tools.utils import natural_keys
from snowwi_tools.utils import set_rcParams
from snowwi_tools.utils import to_wavelength

from argparse import ArgumentParser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
from pprint import PrettyPrinter
pp = PrettyPrinter()

set_rcParams(plt, True)

# TODO - implement doppler centroids, instead of PP


def two_imshow(data1, data2, axis_labels, titles, suptitle, figname, save_to, format, vlims):
    ################################ PLOT SECTION ####################################
    plt.figure()
    ax = plt.subplot(121)
    im = ax.imshow(data1, aspect='auto',  # extent=[gamma_deg[0], gamma_deg[-1], time_array[0], time_array[-1]],
                   cmap='jet')  # , vmin=7*vmin, vmax=7*vmax)
    plt.ylabel(axis_labels[1])
    plt.xlabel(axis_labels[0])
    plt.title(titles[0])
    divider = make_axes_locatable(ax)
    # Adjust size and padding
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, label='$f_{\eta}$ (Hz)')

    ax = plt.subplot(122)
    im = ax.imshow(data2,  aspect='auto',  # extent=[gamma_deg[0], gamma_deg[-1], time_array[0], time_array[-1]],
                   cmap='jet', vmin=vlims[1][0], vmax=vlims[1][1])
    plt.ylabel(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.title(titles[1])
    divider = make_axes_locatable(ax)
    # Adjust size and padding
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, label='$f_{\eta}$ (Hz)')

    plt.suptitle(suptitle)

    plt.tight_layout()
    # figname = f'doppler_model_{args.flightline}_{band}_{channel}_{args.process_from_to[0]:.2f}{args.process_from_to[-1]:.2f}'
    plt.savefig(os.path.join(
        save_to,
        f"{figname}.{format}"),
        dpi=300,
        bbox_inches='tight')
    plt.close()

##################################################################################


def parse_args():

    arg_parser = ArgumentParser()

    arg_parser.add_argument('path_to_novatel',
                            help='Path to NovAtel data.')
    arg_parser.add_argument('path_to_data',
                            help='Path to radar data.')
    arg_parser.add_argument('flightline',
                            help='Flightline to process.')
    arg_parser.add_argument('--channel', '-c',
                            help="Channel to process. If 'all', process all channels. Default: 0",
                            default='0',
                            type=str)
    arg_parser.add_argument('--band', '-b',
                            help="Band to process. If 'all', process all bands. Default: 'all'",
                            default='all',
                            nargs='+')
    arg_parser.add_argument('--save-to', '-st',
                            help="Output path. Will follow /output/path/date/flightline/band/channel. Default: cwd().",
                            default=os.getcwd())
    arg_parser.add_argument('--save-format', '-sf',
                            help="Output format. Default: png.",
                            default='png')
    arg_parser.add_argument('--daq', '-daq',
                            help="DAQ system used in data collection campaign. Options: 'ettus', '4x2'. Default: '4x2'.",
                            default='4x2')
    arg_parser.add_argument('--get-seconds', '-gs',
                            help="Displays the flilghtline length in seconds.",
                            action="store_true")
    arg_parser.add_argument('--process-from-to', '-pft',
                            help='Specify processing limits, in seconds.',
                            nargs=2,
                            type=float)
    arg_parser.add_argument('--yaw-offset', '-yo',
                            help="Introduces artificial offset to yaw field in degrees to Doppler model. CW rotation = positive yaw. Default: no offset (0 deg).",
                            default=0,
                            type=float)
    arg_parser.add_argument('--pitch-offset', '-po',
                            help="Introduces artificial offset to pitch field in degrees to Doppler model. CW rotation = positive yaw. Default: no offset (0 deg).",
                            default=0,
                            type=float)
    arg_parser.add_argument('--yaw-sweep', '-ys',
                            help="If desired, specify angle interval in degrees for iterations, in degrees. If single argument, interval will be ± argument.",
                            nargs='+',
                            type=float)
    arg_parser.add_argument('--pitch-sweep', '-ps',
                            help="If desired, specify angle interval in degrees for iterations, in degrees. If single argument, interval will be ± argument.",
                            nargs='+',
                            type=float)
    arg_parser.add_argument('--sweep-step', '-ss',
                            help="Sweep step of Doppler model iterations. Default: 0.1 deg.",
                            default=0.5,
                            type=float)
    arg_parser.add_argument('--pulse-length', '-cl',
                            help="Define pulse length for pulse compression. Default: 16.5 us.",
                            default=16.5e-6)
    arg_parser.add_argument('--time-offset', '-to',
                            help="Artificial offset to GPS timestamps from radar data, in seconds. Accounts for timezone changes. Default: 0 s.",
                            default=0,
                            type=float)
    arg_parser.add_argument('--nominal-height', '-nh',
                            help="Platform's nominal height for Doppler calculations. Default: 3000 m.",
                            default=3000,
                            type=float)
    arg_parser.add_argument('--ground-range', '-gr',
                            help="Slant range for Doppler calculations. Default: [0, 15e3] m.",
                            nargs=2,
                            default=[0, 15e3],
                            type=float)

    args = arg_parser.parse_args()
    print(args)

    # Check if arguments are valid
    args.path_to_data = os.path.abspath(args.path_to_data)
    args.path_to_novatel = os.path.abspath(args.path_to_novatel)

    if not os.path.exists(args.flightline):
        print("Invalid flightline path.")
        sys.exit(1)

    chan_list = ['0', '1', '2', '3', 'all']
    args.channel = args.channel.lower()
    if args.channel.lower() not in chan_list:
        print(f"Invalid channel. Channels channels: {chan_list}")
        sys.exit(1)
    if args.channel.lower() == 'all':
        args.channel = [0, 1, 2, 3]

    args.band = [item.lower() for item in args.band]
    band_list = ['high', 'low', 'c', 'all']
    band_assert_msg = "Invalid band. Please choose from: low, high, c, all"
    assert set(args.band).issubset(set(band_list)), band_assert_msg
    if args.band == 'all':
        args.band = ['low', 'high', 'c']

    args.save_to = os.path.abspath(args.save_to)
    if not os.path.exists(args.save_to):
        print("Invalid path. Base path should exist.")
        sys.exit(1)

    daq_list = ['ettus', '4x2']
    args.daq = args.daq.lower()
    if not args.daq in daq_list:
        print(f"Invalid DAQ system. Available options: {daq_list}")
        sys.exit(1)

    if 'yaw_range' in args and len(args.yaw_range) == 1:
        args.yaw_range = [-args.yaw_range, args.yaw_range]
        print(f"Setting yaw range to {args.yaw_range}...")
    if 'pitch_range' in args and len(args.pitch_range) == 1:
        args.pitch_range = [-args.pitch_range, args.pitch_range]
        print(f"Setting yaw range to {args.pitch_range}...")

    args.yaw_offset = np.deg2rad(args.yaw_offset)
    args.pitch_offset = np.deg2rad(args.pitch_offset)

    return args


def main():

    args = parse_args()
    print(args)
    print("Path to NovAtel files:")
    print(f"    {args.path_to_novatel}\n")
    print("Path to Radar data:")
    print(f"    {args.path_to_data}\n")

    date = int(os.path.basename(args.path_to_data))
    print(f"Date to process: {date}")

    if args.get_seconds:
        num_files = len(glob.glob(
            os.path.join(args.path_to_data,
                         f'**/{args.flightline}/chan0/**/*.dat'),
            recursive=True))
        print(
            f"Flightline {args.flightline} is {num_files*0.5:.2f} sec long ({num_files} files).")
        sys.exit(0)

    novatel_files = glob.glob(
        os.path.join(args.path_to_novatel, f"**/*{date}*.txt"),
        recursive=True
    )
    pp.pprint(novatel_files)

    tmp_df_list = []

    for nv_file in novatel_files:
        tmp_df_list.append(
            read_novatel(nv_file)
        )

    novatel_df = pd.concat(tmp_df_list)
    print(f"GPS Week: {novatel_df['Week'].iloc[0]}")
    print(
        f"    from {novatel_df['GPSSeconds'].iloc[0]} to {novatel_df['GPSSeconds'].iloc[-1]} s.")

    # This is sketchy. Will calculate the Doppler Model on our first iteration.
    doppler_models = {}
    time_limits = []  # Same, this will be [start_time, stop_time]

    for channel in args.channel:
        print(f"Processing channel {channel}...\n")
        path_to_process = os.path.join(
            args.path_to_data, args.flightline, f"chan{channel}", "*.dat"
        )

        filelist = glob.glob(path_to_process)
        filelist.sort(key=natural_keys)

        # Check if indices are valid
        num_files = len(filelist)
        num_seconds = 0.5 * num_files
        if args.get_seconds:
            print(
                f"Flightline {args.flightline} is {num_seconds:.2f} sec long ({num_files} files).")
            sys.exit(0)

        if args.process_from_to:
            start_idx = int(2 * args.process_from_to[0])
            stop_idx = int(2 * args.process_from_to[-1])
            if start_idx > num_files:
                print(
                    f"Invalid start time. Time interval is [0, {num_seconds}] s.")
                sys.exit(1)
            if stop_idx > num_files:
                print(
                    f"Invalid stop time. Time interval is [0, {num_seconds}] s.")
                print(f"Setting stop time to {num_seconds} s.")
                stop_idx = None
            filelist = filelist[start_idx:stop_idx]

        print("Processing from:")
        print(f"    {filelist[0]}")
        print("to: ")
        print(f"    {filelist[-1]}\n")

        # Read raw data in channel
        daq_params = get_band_params_4x2('daq')
        with Pool(processes=os.cpu_count()) as p:
            tmp_results = p.starmap(
                read_and_reshape,
                [(file,
                  daq_params['data_samps'],
                  daq_params['header_samps'])
                 for file in filelist]
            )

        raw_data, timestamps, headers = combine_results(tmp_results)
        gps_timestamps = timestamp_from_header_4x2(
            headers, args.time_offset)
        del (tmp_results)
        print(f"Raw data shape: {raw_data.shape}")
        print(f"GPS timestamps shape: {gps_timestamps.shape}")
        time_limits.append(gps_timestamps[0])
        time_limits.append(gps_timestamps[-1])
        print(time_limits)

        # Build generic Doppler model from NovAtel
        gps_start = timestamp_to_week_seconds(time_limits[0])
        gps_stop = timestamp_to_week_seconds(time_limits[-1])
        assert gps_start['Week'] == gps_stop['Week'], print(
            "Different GPS week.")
        fl_info = {
            'Date (local)': date,
            'Week': gps_start['Week'],
            'Start': gps_start["GPSSeconds"],
            'Stop': gps_stop["GPSSeconds"]
        }
        pp.pprint(fl_info)
        attitude_dict = get_attitude_dictionary(
            novatel_df,
            fl_info
        )

        # Calculate the oversample factor
        diff_data = np.mean(np.diff(gps_timestamps))
        diff_novatel = np.mean(np.diff(attitude_dict['time']))
        print(diff_data, diff_novatel)

        os_factor = int(diff_novatel // diff_data)
        print(
            f"The radar data is oversampled by a factor of {os_factor} w.r.t. the NovAtel data.")

        save_to = os.path.join(args.save_to,
                               f"{date}",
                               args.flightline,
                               'imgs')
        make_if_not_a_dir(save_to)

        plt.figure()
        plt.plot(attitude_dict['time'], attitude_dict['ypr'][0][0])
        plt.title(f'Yaw - {args.flightline}')
        plt.xlabel("GPS time (s)")
        plt.ylabel("Yaw (deg.)")
        plt.grid(linestyle='--', linewidth=.7)
        plt.savefig(os.path.join(
            save_to,
            f"yaw_{args.flightline}_{args.process_from_to[0]}_{args.process_from_to[-1]}.{args.save_format}"),
            dpi=300,
            bbox_inches='tight')

        ground_range = np.linspace(
            args.ground_range[0], args.ground_range[1], raw_data.shape[1])
        # Look angle is not equispaced!!!!
        look_angle = np.arctan(ground_range / args.nominal_height)
        look_angle = np.rad2deg(look_angle)
        print(look_angle[0], look_angle[-1])
        if not doppler_models:
            print("Generating Doppler models for chosen data and bands...")
            for band in args.band:
                save_to = os.path.join(
                    os.path.abspath(args.save_to),
                    f"{date}",
                    args.flightline,
                    'imgs',
                    band,
                    f"chan{args.channel}"
                )
                print(f"Output directory:")
                print(f"    {save_to}\n")
                make_if_not_a_dir(save_to)
                print(f"Generating Doppler model for {band}...")
                band_params = get_band_params_4x2(band)
                doppler_model, doppler_model_al = generate_doppler_model(
                    attitude_dict['ypr'][0][0],
                    attitude_dict['ypr'][0][1],
                    look_angle,
                    attitude_dict['mag_vels'],
                    to_wavelength(band_params['f0']),
                    daq_params['prf']
                )
                doppler_models = {band: doppler_model_al}
                print(f"Doppler model shape: {doppler_model.shape}.")

                # In a future, experiment with plotting inside a function
                figname = f'doppler_model_{args.flightline}_{band}_{channel}_{args.process_from_to[0]:.2f}{args.process_from_to[-1]:.2f}'
                two_imshow(
                    doppler_model,
                    doppler_model_al,
                    ("Rg. samples", "Az. samples"),  # (x-label, y-label)
                    ("True Doppler", "Aliased Doppler"),
                    "Doppler model. Davidson \& Cumming, 1997",
                    figname,
                    save_to,
                    args.save_format,
                    ((None, None), (-daq_params['prf']/2, daq_params['prf']/2))
                )

        for band in args.band:
            print(f"Processing band: {band}...\n")
            band_params = get_band_params_4x2(band)

            save_to = os.path.join(
                os.path.abspath(args.save_to),
                f"{date}",
                args.flightline,
                'imgs',
                band,
                f"chan{args.channel}"
            )
            print(f"Saving to:")
            print(f"    {save_to}")

            chirp_dict = make_chirp_dict(
                band_params['f0'],
                band_params['f_l'],
                band_params['f_h'],
                args.pulse_length,
                band_params['chirp_type'],
                daq_params['fs']
            )

            pp.pprint(chirp_dict)

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
            assert all(
                [ch.shape[1] == samp_ref for ch in tmp_results]), sh_assert_msg

            rc = np.vstack([tmp for tmp in tmp_results])
            print(rc.shape)

            fig, ax = plt.subplots()
            im = ax.imshow(20*np.log10(abs(rc)), cmap='gray', aspect='auto',
                           vmin=80, vmax=120)
            divider = make_axes_locatable(ax)
            # Adjust size and padding
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax, label='(dB) - Uncal.')
            figname = f'rc_{args.flightline}_{band}_{channel}_{args.process_from_to[0]:.2f}{args.process_from_to[-1]:.2f}'
            plt.savefig(os.path.join(
                save_to,
                f"{figname}.{args.save_format}"),
                dpi=300,
                bbox_inches='tight')

            # We could plot RC
            pp_phase = pulse_pair_doppler(
                rc,
                daq_params['prf'],
                os_factor
            )

            fig, ax = plt.subplots()
            ax.imshow(pp_phase, cmap='jet', aspect='auto')
            # Adjust size and padding
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax, label='(dB) - Uncal.')
            figname = f'pp_doppler_{args.flightline}_{band}_{channel}_{args.process_from_to[0]:.2f}{args.process_from_to[-1]:.2f}'
            plt.savefig(os.path.join(
                save_to,
                f"{figname}.{args.save_format}"),
                dpi=300,
                bbox_inches='tight')

            if args.yaw_sweep or args.pitch_sweep:
                print(args.yaw_sweep, args.pitch_sweep)
                y_num = int(
                    abs(args.yaw_sweep[1] -
                        args.yaw_sweep[0]) / args.sweep_step) + 1
                yaw_interval = np.linspace(
                    args.yaw_sweep[0], args.yaw_sweep[1], num=y_num)
                p_num = int(
                    abs(args.pitch_sweep[1] -
                        args.pitch_sweep[0]) / args.sweep_step) + 1
                pitch_interval = np.linspace(
                    args.pitch_sweep[0], args.pitch_sweep[1], num=p_num
                )
                print(
                    f"Yaw sweep: [{yaw_interval[0]}, {yaw_interval[-1]}]. Step: {args.sweep_step}. Length: {len(yaw_interval)}")
                print(
                    f"Pitch sweep: [{pitch_interval[0]}, {pitch_interval[-1]}]. Step: {args.sweep_step}. Length: {len(pitch_interval)}")

                for y in yaw_interval:
                    for p in pitch_interval:
                        _, doppler_model_al = generate_doppler_model(
                            attitude_dict['ypr'][0][0],
                            attitude_dict['ypr'][0][1],
                            look_angle,
                            attitude_dict['mag_vels'],
                            to_wavelength(band_params['f0']),
                            daq_params['prf'],
                            y_offset=y,
                            p_offset=p
                        )

                        figname = f'pp_vs_model_sweep_{args.flightline}_{band}_{channel}_{args.process_from_to[0]:.2f}{args.process_from_to[-1]:.2f}_({y:.3f},{p:.3f})'
                        two_imshow(
                            pp_phase,
                            doppler_model_al,
                            ("Rg. samples", "Az. samples"),
                            ("Pulse-Pair Doppler", "Doppler Model"),
                            f"Attitude offsets (Y, P) deg.: ({y:.3f}, {p:.3f}) deg.",
                            figname,
                            save_to,
                            args.save_format,
                            ((-daq_params['prf']/2, daq_params['prf']/2),
                             (-daq_params['prf']/2, daq_params['prf']/2))
                        )


if __name__ == "__main__":
    now = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")
    print(f"Elapsed (s): {time.time() - now}")
