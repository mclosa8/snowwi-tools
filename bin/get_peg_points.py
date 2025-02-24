#!/usr/bin/env python3

"""
    Generates PEG points from NovAtel data and spreadsheet.

    Author: Marc Closa Tarres
    Date: 2024-11-19
    Version: 0.1

    Changelog:
        - v0.1: Initial version (MCT)
"""

from snowwi_tools.utils import read_spreadsheet, natural_keys
from snowwi_tools.novatel import read_novatel, get_llh, get_cog
import numpy as np
import pandas as pd

import argparse
import glob

import pprint
pp = pprint.PrettyPrinter(indent=4)


def parse_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('novatel_path', type=str,
                            help="NovAtel data file.")
    arg_parser.add_argument('spreadsheet_file', type=str,
                            help="Spreadsheet file with flightlines.")
    arg_parser.add_argument(
        'campaign', type=str, help="Campaign month. 2024: January, March. 2025: January, February.")
    arg_parser.add_argument('--output', '-o', type=str, default='snowwi_peg.txt',
                            help="Output txt file with PEG points.")

    args = arg_parser.parse_args()

    return args


def main():

    args = parse_args()

    # List novatel files
    novatel_files = sorted(
        glob.glob(args.novatel_path + '/*.txt'), key=natural_keys)
    pp.pprint(novatel_files)

    # Read spreadsheet
    excel_df = read_spreadsheet(args.spreadsheet_file, args.campaign)

    # Iterate over flightlines in spreadsheet and save names to dict
    flightline_dict = {}
    for idx, row in excel_df.iterrows():
        fl_id = row['FlightLog ID']
        if fl_id not in flightline_dict:
            flightline_dict[fl_id] = []
    # Deal with nan key
    flightline_dict.pop(np.nan)

    # Here we have initialized the flightline_dict with the flightline IDs.
    print(flightline_dict)

    all_dates = {}

    for novatel_file in novatel_files:
        print(f'\n\nProcessing {novatel_file}...')

        date_candidates = novatel_file.split('/')[-1].split('.')[0].split('_')
        date = [date for date in date_candidates if date.isdigit()] # We do this because the naming of NovAtel files is inconsistent.
        date = int(date[0])
        all_dates[date] = []
        if (date == 20240328):
            band = novatel_file.split(
                '/')[-1].split('_')[2].split('.')[0].upper()
        else:
            band = False
        print(band)

        # Iterate over flightlines in spreadsheet and save names to dict
        fl_date = flightline_dict.copy()

        fl_info = excel_df.loc[(
            excel_df['Date (local)'] == date)].reset_index(drop=True)
        print(fl_info)
        if band:
            mask = [(band in tmp) and (tmp != 'APPROACH')
                    for tmp in fl_info['Complete ID']]
            fl_info = fl_info.loc[mask].reset_index(drop=True)
            print(fl_info)

        all_dates[date] = fl_date

        # Retrieve flightlines and peg points
        print('Reading NovAtel file...')
        novatel_df = read_novatel(novatel_file)
        print(novatel_df.keys())

        novatel_df['GPSSeconds'] = pd.to_numeric(
            novatel_df['GPSSeconds'], errors='coerce')
        novatel_df['Week'] = pd.to_numeric(novatel_df['Week'], errors='coerce')

        for idx, row in fl_info.iterrows():
            fl_id = row['FlightLog ID']

            print(f'\nProcessing flightline {fl_id}...')

            flightline = novatel_df.loc[(novatel_df['GPSSeconds'] >= row['Start']) & (
                novatel_df['GPSSeconds'] <= row['Stop'])]
            print(flightline.shape)

            # Retrieve ECEF coordinates
            llh = get_llh(flightline)
            print(llh.shape)
            cog_heading = get_cog(flightline)
            print(cog_heading.shape)

            # And we column_stack llh and cog so the final tuple is (lat, lon, hei, cog_heading)
            llh = np.vstack((llh, cog_heading))
            print(llh.shape)

            # Calculate PEG point
            peg = np.mean(llh, axis=1)
            print(peg.shape)
            print(f'PEG point (llhh): {peg}')
            print(not any(np.isnan(peg)))
            if not any(np.isnan(peg)):
                print('Appending...')
                fl_date[fl_id].append(peg)
            # print(fl_date)

        all_dates[date] = fl_date
        pp.pprint(all_dates)

    # Now we iterate over all the dates and we average all the PEGs
    for date, date_dict in all_dates.items():

        # Now we iterate over the flightlines and average the PEGs
        for fl_id, value in date_dict.items():
            print(value)
            pegs = np.array(value)
            print(pegs)
            means = np.mean(pegs, axis=0)
            print(means)
            flightline_dict[fl_id].append(means)

    pp.pprint(flightline_dict)

    # Now, one last time we iterate over all the flightlines and we average the PEGs
    for fl_id in flightline_dict.keys():
        flightline_dict[fl_id] = np.mean(flightline_dict[fl_id], axis=0)

    pp.pprint(flightline_dict)

    sorted_keys = list(flightline_dict.keys())
    sorted_keys.sort(key=natural_keys)

    # Finally, we write the dict to file
    with open(f'pegs_{args.campaign.lower()}.txt', 'w') as f:
        f.write(
            f'Flightline_ID    PEG_LAT(deg)    PEG_LON(deg)    PEG_H-ELL(m)    PEG-Heading(deg)\n')
        for key in sorted_keys:
            value = flightline_dict[key]
            f.write(
                f'{key}    {value[0]}    {value[1]}    {value[2]}    {value[3]}\n')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Script interrupted.')
