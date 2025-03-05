#!/usr/bin/env python3

"""
    Plots a summary of attitude from each flightline.

    Author: Marc Closa Tarres (MCT)
    Date: 2025-02-19
    Version: v0

    Changelog:
        - v0: Initial version. - Feb 19, 2025 - MCT

"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import glob
import os
import sys

from argparse import ArgumentParser

from pprint import PrettyPrinter

from snowwi_tools.doppler.doppler import calculate_doppler_from_fl

from snowwi_tools.lib.file_handling import make_if_not_a_dir

from snowwi_tools.lib.novatel import get_attitude_dictionary
from snowwi_tools.lib.novatel import read_excel_database
from snowwi_tools.lib.novatel import read_novatel
from snowwi_tools.lib.novatel import retrieve_date_from_excel_database
from snowwi_tools.lib.novatel import retrieve_flightline

from snowwi_tools.utils import natural_keys
from snowwi_tools.utils import set_rcParams

from snowwi_tools.params import get_band_params_4x2

set_rcParams(plt, tex=True)

def parse_args():
    
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        'novatel_path',
        help='Path to NovAtel files.'
    )
    arg_parser.add_argument(
        'excel_database',
        help='Excel database with flightline info.'
    )
    arg_parser.add_argument(
        'campaign',
        help='Campaign specifier from Excel database.'
    )
    arg_parser.add_argument(
        '-st',
        '--save-to',
        help='Path to save images.',
        default=f"{os.getcwd()}/imgs",
    )
    arg_parser.add_argument('--look-angle', '-la',
                            help='Set the look (depression) angle for the antennas. Default: 35 deg.',
                            default=35,
                            type=float)

    args = arg_parser.parse_args()

    return args


def main():
    
    pp = PrettyPrinter()
    
    args = parse_args()
    print(args)

    print(f"\nPath to NovAtel files: {args.novatel_path}\n")
    print(f"\nFlightline database: {args.excel_database}\n")
    
    print(f'\nCampaign chosen: {args.campaign}\n')

    novatel_list = glob.glob(os.path.join(args.novatel_path, f'*.txt'))
    novatel_list.sort(key=natural_keys)

    print('Flights to process:')
    pp.pprint(novatel_list)

    print(f"Saving outputs to: {args.save_to}")
    make_if_not_a_dir(args.save_to)

    # Read excel database and retrieve flight days
    flightlines_db = read_excel_database(args.excel_database, args.campaign)

    dates = flightlines_db['Date (local)'].tolist()
    print(dates)

    filtered_dates = set([date for date in dates if (type(date)==int)])
    print(filtered_dates)

    flight_dict = {}

    kul_params = get_band_params_4x2('low')
    kuh_params = get_band_params_4x2('low')
    c_params = get_band_params_4x2('c')
    prf = get_band_params_4x2('prf')
    dop_nyq = prf/2

    for date in filtered_dates:
        novatel_list = glob.glob(os.path.join(args.novatel_path, f'*{date}*.txt'))
        print(novatel_list)

        tmp_df_list = []

        for nv_file in novatel_list:
            tmp_df_list.append(
                read_novatel(nv_file)
            )
        
        novatel_df = pd.concat(tmp_df_list)
        print(novatel_df.shape)

        flight_dict[date] = {}

        flightlines_day = retrieve_date_from_excel_database(flightlines_db, date)

        for flightline_info in flightlines_day.iterrows(): # flightline_info is a tuple!!!!!!
            fl_info = flightline_info[1]
            
            attitude = get_attitude_dictionary(novatel_df, fl_info)
            """
            For keys 'xyz' and 'ypr', values are tuples of the form (values [3xN], means [3x1])
            """

            # Dictionary will follow the structure:
            """
                flight_dict[date][fl_id] = {
                    'yaw': (mean, 1_std),
                    'pitch': (mean, 1_std),
                    'roll': (mean, 1_std),
                    'centroid': (dopp_centroids @ look_angle (35 deg), mean, 1_std)
                }
            """

            f_dopp_kul = calculate_doppler_from_fl(
                attitude,
                args.look_angle,
                kul_params['f0']
                )
            f_dopp_kuh = calculate_doppler_from_fl(
                attitude,
                args.look_angle,
                kuh_params['f0']
                )
            f_dopp_c = calculate_doppler_from_fl(
                attitude,
                args.look_angle,
                c_params['f0']
                )

            flight_dict[date][fl_info['FlightLog ID']] = {
                'yaw': (np.mean(attitude['ypr'][0][0]), np.std(attitude['ypr'][0][0])),
                'pitch': (np.mean(attitude['ypr'][0][1]), np.std(attitude['ypr'][0][1])),
                'roll': (np.mean(attitude['ypr'][0][2]), np.std(attitude['ypr'][0][2])),
                'centroid': {
                    'low': (f_dopp_kul, np.mean(f_dopp_kul), np.std(f_dopp_kul)),
                    'high': (f_dopp_kuh, np.mean(f_dopp_kuh), np.std(f_dopp_kuh)),
                    'c': (f_dopp_c, np.mean(f_dopp_c), np.std(f_dopp_c))
                }
            }

    # Here we should have all the necessary info to plot the different params
    # TODO - Make it look nicer... 
    # Initialize subplots
    fig1, (ax11, ax12, ax13) = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True)
    fig2, (ax21, ax22, ax23) = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True)

    # Define colors for each day
    days = list(flight_dict.keys())
    num_days = len(days)
    colors = {day: cm.tab10(i / 10) for i, day in enumerate(days)}

    x_pos = []
    x_labels = []

    x_pos_ctr = 0

    ax21.set_ylim(-4*prf, 4*prf)
    ax22.set_ylim(-4*prf, 4*prf)
    ax23.set_ylim(-4*prf, 4*prf)

    ax11.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax12.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax13.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax21.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax22.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax23.axhline(0, color='gray', linestyle='--', alpha=0.7)

    ax21.axhspan(dop_nyq, ax21.get_ylim()[1], color='r', alpha=0.1)
    ax21.axhspan(-dop_nyq, -ax21.get_ylim()[1], color='r', alpha=0.1)
    ax22.axhspan(dop_nyq, ax21.get_ylim()[1], color='r', alpha=0.1)
    ax22.axhspan(-dop_nyq, -ax21.get_ylim()[1], color='r', alpha=0.1)
    ax23.axhspan(dop_nyq, ax21.get_ylim()[1], color='r', alpha=0.1)
    ax23.axhspan(-dop_nyq, -ax21.get_ylim()[1], color='r', alpha=0.1)

    for day, sub_dict in flight_dict.items():
        print(day)
        year = int(day/10000)
        for k, v in sub_dict.items():
            print(k)

            x_pos.append(x_pos_ctr)
            x_labels.append(k)

            ax11.errorbar(x_pos_ctr, v['yaw'][0], yerr=v['yaw'][1], fmt='.', elinewidth=1.2,
                         color=colors[day], label=day if x_pos_ctr == 0 else "")
            ax12.errorbar(x_pos_ctr, v['pitch'][0], yerr=v['pitch'][1], fmt='.',elinewidth=1.2,
                         color=colors[day], label=day if x_pos_ctr == 0 else "")
            ax13.errorbar(x_pos_ctr, v['roll'][0], yerr=v['roll'][1], fmt='o',elinewidth=1.2,
                         color=colors[day], label=day if x_pos_ctr == 0 else "")

            ctds = v['centroid']
            ax21.errorbar(x_pos_ctr, ctds['low'][1], yerr=ctds['low'][2], fmt='.',elinewidth=1.2,
                         color=colors[day], label=day if x_pos_ctr == 0 else "")
            ax22.errorbar(x_pos_ctr, ctds['high'][1], yerr=ctds['high'][2], fmt='.',elinewidth=1.2,
                         color=colors[day], label=day if x_pos_ctr == 0 else "")
            ax23.errorbar(x_pos_ctr, ctds['c'][1], yerr=ctds['c'][2], fmt='.',elinewidth=1.2,
                         color=colors[day], label=day if x_pos_ctr == 0 else "")
            
            x_pos_ctr += 1

    # Formatting stuff
    ax11.grid(linestyle='--', linewidth=.3, alpha=.5)
    ax12.grid(linestyle='--', linewidth=.3, alpha=.5)
    ax13.grid(linestyle='--', linewidth=.3, alpha=.5)
    ax21.grid(linestyle='--', linewidth=.3, alpha=.5)
    ax22.grid(linestyle='--', linewidth=.3, alpha=.5)
    ax23.grid(linestyle='--', linewidth=.3, alpha=.5)

    ax11.set_ylabel('Yaw (deg)')
    ax12.set_ylabel('Pitch (deg)')
    ax13.set_ylabel('Roll (deg)')
    fig1.suptitle(f'Attitude - {args.campaign} {year} campaign.', fontsize=12)
    ax13.set_xticks(x_pos, x_labels, rotation=90)

    ax21.set_ylabel('Ku-Low (Hz)')
    ax22.set_ylabel('Ku-High (Hz)')
    ax23.set_ylabel('C-Band (Hz)')
    fig2.suptitle(f'Doppler centroids (Look angle: {args.look_angle:.2f} deg.) - {args.campaign} {year} campaign.', fontsize=12)
    ax23.set_xticks(x_pos, x_labels, rotation=90)

    handles = [plt.Line2D([0], [0], marker='.', color=colors[day], linestyle='', markersize=8) for day in days]
    labels = days
    fig1.legend(handles, labels, loc="upper right", title="Days")
    fig2.legend(handles, labels, loc="upper right", title="Days")

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f'{args.save_to}/{args.campaign}{year}_attitude.png', dpi=200)
    fig2.savefig(f'{args.save_to}/{args.campaign}{year}_{args.look_angle:.0f}_doppler.png', dpi=200)
        


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")



