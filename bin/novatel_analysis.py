#!/usr/bin/env python3

"""
    Flightline analysis from NovAtel files.

    Author: Marc Closa Tarres (MCT)
    Date: 2025-02-13
    Version: v0

    Changelog:
        - v0: Feb 13, 2025
        - v.1: Feb 09, 2026 - Changed color lines to gradient for ease of visualization. LLM used for line gradient skeleton.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
import os
import utm
import sys

from argparse import ArgumentParser
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pprint import PrettyPrinter

from snowwi_tools.lib import file_handling
from snowwi_tools.lib.novatel import read_novatel
from snowwi_tools.lib.novatel import get_attitude_dictionary
from snowwi_tools.lib.novatel import read_excel_database_and_get_date
from snowwi_tools.lib.novatel import ecef_2_tcn

from snowwi_tools.utils import natural_keys
from snowwi_tools.utils import set_rcParams

# Set rcParams
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
        'date',
        help="Specify date to process. Format: YYYYMMDD. If multiple dates to process, will prompt to specify. Introduce 'display' to list flightlines available.",
    )
    arg_parser.add_argument(
        'flightline',
        help="Specify flightline index / name. Introduce 'display' to list flightlines available. If 'all', process all flightlines from date.",
        nargs='+'
    )
    arg_parser.add_argument(
        '--add-start-time',
        help='Advances the start time of the chosen flightline in seconds.',
        default=0,
        type=int
    )
    arg_parser.add_argument(
        '--add-end-time',
        help='Delays the end time of the chosen flightline in seconds.',
        default=0,
        type=int
    )
    arg_parser.add_argument(
        '-st',
        '--save-to',
        help='Path to save images.',
        default=f"{os.getcwd()}/imgs",
    )
    
    args = arg_parser.parse_args()
    
    if args.date.isdigit():
        args.date = int(args.date)

    print(args)
        
    return args


def main():
    
    pp = PrettyPrinter()
    
    args = parse_args()
    print(args)

    print(f"\nPath to NovAtel files: {args.novatel_path}\n")
    print(f"\nFlightline database: {args.excel_database}\n")
    
    print(f'\nCampaign chosen: {args.campaign}\n')
   
    print(f"Processing NovAtel files from {args.date}")

    # List available flightlines from specified date
    if 'display' in str(args.date):
        novatel_list = glob.glob(args.novatel_path+f'/*.txt')
        novatel_list.sort(key=natural_keys)
        for i, file in enumerate(novatel_list):
            print(f'[{i}] - {file}')
        sys.exit(0)
    else:
        novatel_list = glob.glob(args.novatel_path+f'/*{args.date}*.txt')
        novatel_list.sort(key=natural_keys)
        for i, file in enumerate(novatel_list):
            print(f'[{i}] - {file}')
        if len(novatel_list) > 1:
            idx = int(input("Choose file to process: "))
            nv_file_to_process = novatel_list[idx]
        else:
            nv_file_to_process = novatel_list[0]
        print(f"Processing NovAtel file {nv_file_to_process}\n\n")

    nv_df = read_novatel(nv_file_to_process)
    print(nv_df.shape)

    flightlines = read_excel_database_and_get_date(
    args.excel_database, args.campaign, args.date)


    if 'display' in args.flightline:
        print(flightlines['FlightLog ID'])
        sys.exit(0)
    elif 'all' in args.flightline:
        fls_to_process = flightlines['FlightLog ID'].values.tolist()
        print("Processing all flightlines.\n\n")
    else:
        fls_to_process = args.flightline

    for flightline in fls_to_process:

        print(f"\nFlightline to process: {flightline}.")

        if not flightline.isdigit():
            idx = flightlines.index[(flightlines['FlightLog ID'] == flightline)][0]
            print(f"Converting {args.flightline} into index {idx}")
        else:
            idx = int(flightline)

        fl_info = flightlines.loc[idx]
        print(fl_info)
        fl_info['Start'] = fl_info['Start'] - args.add_start_time
        fl_info['Stop'] = fl_info['Stop'] + args.add_end_time
        print(fl_info)
        
        subdir_name = fl_info['Complete ID'].replace("-", f"/{idx}_")
        fl_image_path = os.path.join(f'{args.save_to}', f'{subdir_name}')
        print(f"Saving to: {fl_image_path}")
        if not os.path.exists(fl_image_path):
            print(f'{fl_image_path} does not exist. Creating it...')
            os.makedirs(fl_image_path, exist_ok=True)
        else:
            print(f"{fl_image_path} already exists.")


        fl_df = get_attitude_dictionary(nv_df, fl_info)
        if fl_df is None:
            print("Skipping...")
            continue
        

        # Full flight vs flightline - LAT LON plot
        plt.figure()
        plt.plot(nv_df['Longitude'], nv_df['Latitude'])
        plt.plot(fl_df['llh'][1], fl_df['llh'][0], 'r')
        plt.title(f"{fl_info['Complete ID']}")
        plt.ylabel('Latitude (deg)')
        plt.xlabel('Longitude (deg)')
        plt.grid(linestyle='--', linewidth=.7)
        plt.savefig(f"{fl_image_path}/{fl_info['Complete ID']}_fullflight.png", dpi=300, bbox_inches='tight')

       # TCN and ideal TCN line
        tcn_points = ecef_2_tcn(fl_df['xyz'], fl_df['time'])

        # Calculate ideal TCN
        ideal_tcn = np.vstack((
            np.linspace(tcn_points[0, 0], tcn_points[0, -1], num=len(tcn_points[0])),
            np.zeros_like(tcn_points[0]),
            np.zeros_like(tcn_points[0])
        ))



        # Big plot - 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False


        # ---- Gradient line for tcn_points ----
        xp = tcn_points[0]
        yp = tcn_points[1]
        zp = tcn_points[2]

        points = np.column_stack([xp, yp, zp])
        segments = np.stack([points[:-1], points[1:]], axis=1)

        # Color along trajectory (index-based)
        values = np.arange(len(xp) - 1)

        lc = Line3DCollection(
            segments,
            cmap="RdYlGn_r",
            linewidths=1,
            alpha=1,
        )
        lc.set_array(values)

        ax.add_collection(lc)


        # Ideal trajectory (plain line)
        ax.plot(
            ideal_tcn[0],
            ideal_tcn[1],
            ideal_tcn[2],
            color="k",
            linewidth=1,
        )

        # Start / end points
        ax.plot(xp[0], yp[0], zp[0], ".g")
        ax.plot(xp[-1], yp[-1], zp[-1], ".r")

        # Labels and title
        ax.set_xlabel("Along-track - T (m)")
        ax.set_ylabel("Cross-track - C (m)")
        ax.set_zlabel("Height - N (m)")
        plt.title(f"{fl_info['Complete ID']}")

        # IMPORTANT: 3D collections do not autoscale
        ax.set_xlim(xp.min(), xp.max())
        ax.set_ylim(yp.min(), yp.max())
        ax.set_zlim(zp.min(), zp.max())

        plt.savefig(f"{fl_image_path}/{fl_info['Complete ID']}_tcn_vs_ideal.png", dpi=300, bbox_inches='tight')



        # Tube plot
        radius = 10
        theta = np.linspace(0, 2*np.pi, num=500)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Tube
        plt.figure(figsize=(3*1.5, 4*1.5))
        plt.subplot(211)

        # Regular line
        plt.plot(x, y, 'k', label=f'Radius: {radius}m.')
        plt.plot(0, 0, '+k')

        # ---- Gradient line for tcn_points ----
        xp = tcn_points[1]
        yp = tcn_points[2]

        points = np.column_stack([xp, yp])
        segments = np.stack([points[:-1], points[1:]], axis=1)

        # Color along the path (index-based)
        values = np.arange(len(xp) - 1)

        lc = LineCollection(segments, cmap="RdYlGn_r", linewidths=1, alpha=1)
        lc.set_array(values)

        ax = plt.gca()
        ax.add_collection(lc)

        # Start / end points
        plt.plot(xp[0], yp[0], '.g')
        plt.plot(xp[-1], yp[-1], '.r')

        # Formatting
        plt.axis('equal')
        plt.grid(linestyle='--', linewidth=.7)
        plt.title(f"{fl_info['Complete ID']}")
        plt.xlabel('Cross-track - C (m)')
        plt.ylabel('Height - N (m)')

        # Important: collections don’t autoscale
        x_min = np.min([x.min(), xp.min()]) * 1.2
        x_max = np.max([x.max(), xp.max()]) * 1.2

        y_min = np.min([y.min(), yp.min()]) * 1.2
        y_max = np.max([y.max(), yp.max()]) * 1.2

        print(x_min, x_max)
        print(y_min, y_max)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Height - along track
        plt.subplot(413)
        plt.plot(ideal_tcn[0], np.ones_like(ideal_tcn[0])*radius, '--k')
        plt.plot(ideal_tcn[0], -np.ones_like(ideal_tcn[0])*radius, '--k')
        plt.plot(ideal_tcn[0], tcn_points[2])
        plt.plot(ideal_tcn[0, 0], tcn_points[2, 0], '.g')
        plt.plot(ideal_tcn[0, -1], tcn_points[2, -1], '.r')
        plt.grid(linestyle='--', linewidth=.7)
        maxval = np.max(np.abs(tcn_points[2]))
        if maxval < radius:
            maxval = radius
        plt.title('Height vs. Along-track')
        plt.xlabel('Along-track - T (m)')
        plt.ylabel('Height - N (m)')
        plt.ylim((-1.2*maxval, 1.2*maxval))

        # Cross track - along track
        plt.subplot(414)
        plt.plot(ideal_tcn[0], np.ones_like(ideal_tcn[0])*radius, '--k')
        plt.plot(ideal_tcn[0], -np.ones_like(ideal_tcn[0])*radius, '--k')
        plt.plot(ideal_tcn[0], tcn_points[1])
        plt.plot(ideal_tcn[0, 0], tcn_points[1, 0], '.g')
        plt.plot(ideal_tcn[0, -1], tcn_points[1, -1], '.r')
        plt.grid(linestyle='--', linewidth=.7)
        maxval = np.max(np.abs(tcn_points[1]))
        if maxval < radius:
            maxval = radius
        plt.ylim((-1.2*maxval, 1.2*maxval))
        plt.title('Cross- vs. Along-track')
        plt.ylabel('Cross-track - C (m)')
        plt.xlabel('Along-track - T (m)')


        plt.tight_layout()
        plt.savefig(f"{fl_image_path}/{fl_info['Complete ID']}_tube.png", dpi=300, bbox_inches='tight')



        # YPR plot
        ypr_range = 5 # NOTE: This is an arbitrary value for the YPR range.

        plt.figure()
        # Yaw
        plt.subplot(311)
        yaw_mean = np.mean(fl_df['ypr'][1][0])
        plt.plot(ideal_tcn[0], fl_df['ypr'][1][0], '--r',
                label=f"{yaw_mean:.2f} deg.")
        plt.plot(ideal_tcn[0], fl_df['ypr'][0][0])
        plt.ylim((yaw_mean - ypr_range, yaw_mean + ypr_range))
        plt.xlabel('Along-track - T (m)')
        plt.ylabel("(deg)")
        plt.grid(linestyle='--', linewidth=.7)
        plt.legend()
        plt.title('Yaw')

        # Pitch
        plt.subplot(312)
        pitch_mean = np.mean(fl_df['ypr'][1][1])
        plt.plot(ideal_tcn[0], fl_df['ypr'][1][1], '--r',
                label=f"{pitch_mean:.2f} deg.")
        plt.plot(ideal_tcn[0], fl_df['ypr'][0][1])
        plt.xlabel('Along-track - T (m)')
        plt.ylabel("(deg)")
        plt.ylim((pitch_mean - ypr_range, pitch_mean + ypr_range))
        plt.grid(linestyle='--', linewidth=.7)
        plt.legend()
        plt.title('Pitch')

        # Roll
        plt.subplot(313)
        roll_mean = np.mean(fl_df['ypr'][1][2])
        plt.plot(ideal_tcn[0], fl_df['ypr'][1][2], '--r',
                label=f"{roll_mean:.2f} deg.")
        plt.plot(ideal_tcn[0], fl_df['ypr'][0][2])
        plt.xlabel('Along-track - T (m)')
        plt.ylabel("(deg)")
        plt.grid(linestyle='--', linewidth=.7)
        plt.legend()
        plt.title('Roll')
        plt.ylim((roll_mean - ypr_range, roll_mean + ypr_range))
        plt.suptitle(f"{fl_info['Complete ID']}")
        plt.tight_layout()

        plt.savefig(f"{fl_image_path}/{fl_info['Complete ID']}_ypr.png", dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")