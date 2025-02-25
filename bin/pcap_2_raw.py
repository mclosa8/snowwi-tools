#!/usr/bin/env python3

"""
    Parses the raw PCAP files generated by host computer to raw radar data files.
    Adaptation from Joseph Maloyan's Jupyter Notebook pcap2dat.ipynb.
    Automatically picks time when 4x2 started recording.

    Author(s): Marc Closa Tarres (MCT)
               Joseph Maloyan
    Date: 2025-02-14
    Version: v0

    Changelog:
        - v0: Initial version. - Feb 14, 2025 - MCT
"""

import numpy as np
import pandas as pd

import glob
import os
import sys

from argparse import ArgumentParser
import datetime
from time import sleep

from snowwi_tools.lib.file_handling import make_if_not_a_dir
from snowwi_tools.lib.utils import natural_keys

# Some constants Joe had hardcoded
packets_per_file = 29343
packets_per_burst = 250
packets_per_new_file = 250 * 500  # Half a second of data
samples_per_packet = 8*50
header_per_packet = 96
# TODO - change this to argument - talk to Joe - Don't really need this anymore
num_files = 1580
packetsTotal = 0

# TODO - Also change this to script argument
base_directory = "/home/olaf/mission_control/in_flight_storage/destination_drive_0/stream_1733449335/raw_data.pcap"

# Global constants
GPS_EPOCH = datetime.datetime(1980, 1, 6, 0, 0, 0)

# Path constants
CWD = os.getcwd()


def parse_args():

    arg_parser = ArgumentParser()

    # TODO - Modify this to accept dynamic paths.
    arg_parser.add_argument('base_directory',
                            help="Path to the PCAP files to process.")
    arg_parser.add_argument('--use-arduino-time', '-at',
                            default='True',
                            action='store_true',
                            help='Uses the arduino time from /base-directory/arduino_dump.log')
    arg_parser.add_argument('--save-to', '-st',
                            help='Path to output directory. Default: /base_directory')
    arg_parser.add_argument('--sec-of-data', '-sd',
                            default='all',
                            help="Total time of radar data to process. Will be converted into number of PCAP files. If 'all', process all the files with data. Default: all.")

    args = arg_parser.parse_args()
    args.base_directory = os.path.normpath(args.base_directory)

    return args


def time_2_num_pcap(time):
    ms_per_pcap = 29343/250
    s_per_pcap = ms_per_pcap / 1000
    print(ms_per_pcap)
    print(s_per_pcap)
    num_files = int(np.ceil(float(time)/s_per_pcap))
    print(num_files)
    return num_files


def datetime_to_gps_timestamp(time):
    return (time - GPS_EPOCH).total_seconds()


def get_arduino_time(directory):
    arduino_file = os.path.join(directory, "arduino_data.log")
    arduino_df = pd.read_csv(arduino_file, sep='\s+', header=None)
    try:
        first_row = arduino_df.loc[arduino_df[3] == 1].iloc[0]
    except Exception:
        print("Invalid arduino file.")
        sys.exit(1)
    fr_date = first_row[9].split('/')  # Of the format [MM, DD, YYYY]
    fr_time = first_row[10].split(':')  # Of the format [hh, mm, ss]
    print(f"Date from Arduino log: {fr_date}")
    print(f"Time from Arduino log: {fr_time}")
    gps_datetime = datetime.datetime(
        int(fr_date[2]), int(fr_date[0]), int(
            fr_date[1]),  # We want this to be YYYY MM DD
        int(fr_time[0]), int(fr_time[1]), int(fr_time[2]))
    print(gps_datetime)
    arduino_gps_s = datetime_to_gps_timestamp(gps_datetime)
    print(arduino_gps_s)
    # host_unix_s = 0
    # arduino_gps_s = int(datetime.strptime("12/06/2024 01:42:22", "%m/%d/%Y %H:%M:%S").timestamp()) #1733449335
    # host_unix_ms = host_unix_s * 1000
    arduino_unix_ms = arduino_gps_s * 1000
    print(arduino_unix_ms)
    return gps_datetime, arduino_unix_ms


def pcap2numpy(pcap_file, packetsPerBurst=250, headerSize=96):
    packets = np.fromfile(pcap_file, dtype=np.uint16)
    packets = packets[12:]
    packets = packets.reshape(-1, 32*53+8).T[8:].flatten()
    full_burst_packets = packets.reshape(32*53, -1)
    print(full_burst_packets.shape)
    header = full_burst_packets[:headerSize]
    data = full_burst_packets[headerSize:]
    data0_indices = np.hstack([np.arange(i, 1600, 32) for i in range(8)])
    data1_indices = np.hstack([np.arange(i, 1600, 32) for i in range(8, 16)])
    data2_indices = np.hstack([np.arange(i, 1600, 32) for i in range(16, 24)])
    data3_indices = np.hstack([np.arange(i, 1600, 32) for i in range(24, 32)])
    data0_indices.sort()
    data1_indices.sort()
    data2_indices.sort()
    data3_indices.sort()
    data0 = data[data0_indices]
    data1 = data[data1_indices]
    data2 = data[data2_indices]
    data3 = data[data3_indices]
    data0 = data0.T.flatten()
    data1 = data1.T.flatten()
    data2 = data2.T.flatten()
    data3 = data3.T.flatten()
    header = header.T.flatten()
    return header, data0, data1, data2, data3


def main():

    args = parse_args()

    print(args)

    base_name = os.path.basename(args.base_directory)
    print(base_name)

    filelist = glob.glob(os.path.join(args.base_directory, "*.pcap*"))
    if len(filelist) == 0:
        args.base_directory = os.path.join(args.base_directory, base_name)
        print(args.base_directory)
        filelist = glob.glob(os.path.join(args.base_directory, "*.pcap*"))
    filelist.sort(key=natural_keys)
    print(len(filelist))

    # If arduino time definition - if arduino not used, uses time from directory name
    if args.use_arduino_time:
        time, arduino_unix_ms = get_arduino_time(args.base_directory)
    else:
        time_from_dir = base_name.split("_")[-1]
        time = datetime.datetime.utcfromtimestamp(time_from_dir)

    print(time)
    dir_name = f"{time.year}{time.month:02}{time.day:02}T{time.hour:02}{time.minute:02}{time.second:02}"
    day_dir = dir_name.split('T')[0]
    print(dir_name)
    print(day_dir)

    # Define output directories
    if not args.save_to:
        save_directory = args.base_directory
    else:
        save_directory = args.save_to

    save_directory = os.path.join(save_directory, day_dir, dir_name)

    if not args.use_arduino_time:
        save_directory += 'no_gps_time'
    print(f"\n\n\nSaving parsed files to {save_directory}")

    # Make output directories
    ch0 = os.path.join(save_directory, 'chan0')
    ch1 = os.path.join(save_directory, 'chan1')
    ch2 = os.path.join(save_directory, 'chan2')
    ch3 = os.path.join(save_directory, 'chan3')
    hds = os.path.join(save_directory, 'headers')

    make_if_not_a_dir(ch0)
    make_if_not_a_dir(ch1)
    make_if_not_a_dir(ch2)
    make_if_not_a_dir(ch3)

    if args.sec_of_data.isdigit():
        num_files = time_2_num_pcap(args.sec_of_data)
        filelist = filelist[:int(num_files)]
    elif args.sec_of_data.lower() == 'all':
        filelist = [file for file in filelist if os.path.getsize(file) > 0]
    else:
        print("Invalid number of files.")
        sys.exit(1)

    num_files = len(filelist)

    print(f"Processing {num_files} files, starting from file")
    print(f"    {filelist[0]}")
    print("to file")
    print(f"    {filelist[-1]}")
    print("\n\n\n")
    sleep(3)

    # Definition of streams
    stream0 = []
    stream1 = []
    stream2 = []
    stream3 = []
    streamHeader = []

    j = 0
    packetsTotal = 0

    for i in range(num_files):
        if i == 0:
            directory = filelist[0]
            print(directory)
        else:
            directory = filelist[0] + str(i)
            if i == 1:
                print(directory)

        header, data0, data1, data2, data3 = pcap2numpy(directory)
        packetsTotal += packets_per_file

        if packetsTotal >= packets_per_new_file:
            print("Packets Total Old: ", packetsTotal)
            packetsTotal = packetsTotal % packets_per_new_file
            print("Packets Total New: ", packetsTotal)
            print("Data0 length: ", data0.shape)
            print(packetsTotal * samples_per_packet)
            stream0.append(data0[:-packetsTotal*samples_per_packet])
            stream1.append(data1[:-packetsTotal*samples_per_packet])
            stream2.append(data2[:-packetsTotal*samples_per_packet])
            stream3.append(data3[:-packetsTotal*samples_per_packet])
            streamHeader.append(header[:-packetsTotal*header_per_packet])
            leftover_data0 = data0[-packetsTotal*samples_per_packet:]
            leftover_data1 = data1[-packetsTotal*samples_per_packet:]
            leftover_data2 = data2[-packetsTotal*samples_per_packet:]
            leftover_data3 = data3[-packetsTotal*samples_per_packet:]
            leftover_header = header[-packetsTotal*header_per_packet:]
            # If leftover_data has 0 as the first element in its shape, remove the last element from strea
            if leftover_data0.shape[0] == 0:
                stream0.pop()
                stream1.pop()
                stream2.pop()
                stream3.pop()
                print("Popped data")
            if leftover_header.shape[0] == 0:
                streamHeader.pop()
                print("Popped header")

            # save the data
            if not args.use_arduino_time:
                print("Saving at iteration: ", i)
                np.hstack(stream0).tofile(os.path.join(
                    ch0, "stream0_" + str(j) + ".dat"))
                np.hstack(stream1).tofile(os.path.join(
                    ch1, "stream1_" + str(j) + ".dat"))
                np.hstack(stream2).tofile(os.path.join(
                    ch2, "stream2_" + str(j) + ".dat"))
                np.hstack(stream3).tofile(os.path.join(
                    ch3, "stream3_" + str(j) + ".dat"))
                np.hstack(streamHeader).tofile(os.path.join(
                    save_directory, "streamHeader_" + str(j) + ".dat"))
            else:
                stream0 = np.hstack(stream0).flatten()
                stream1 = np.hstack(stream1).flatten()
                stream2 = np.hstack(stream2).flatten()
                stream3 = np.hstack(stream3).flatten()
                streamHeader = np.hstack(streamHeader).flatten()
                arduino_times = np.linspace(
                    arduino_unix_ms + j*500, arduino_unix_ms + 499 + j*500, 500).astype(np.uint64)
                shift_amounts = np.array([48, 32, 16, 0])
                insert_arrays = np.array([np.bitwise_and(np.right_shift(
                    arduino_times[l], shift_amounts), np.uint64(0xFFFF)).astype(np.uint16) for l in range(500)])
                stream0 = stream0.reshape(-1, 100000)
                stream1 = stream1.reshape(-1, 100000)
                stream2 = stream2.reshape(-1, 100000)
                stream3 = stream3.reshape(-1, 100000)
                stream0 = np.hstack([insert_arrays, stream0])
                stream1 = np.hstack([insert_arrays, stream1])
                stream2 = np.hstack([insert_arrays, stream2])
                stream3 = np.hstack([insert_arrays, stream3])
                print("Saving at iteration: ", i)
                stream0.tofile(os.path.join(
                    ch0, "stream0_" + str(j) + ".dat"))
                stream1.tofile(os.path.join(
                    ch1, "stream1_" + str(j) + ".dat"))
                stream2.tofile(os.path.join(
                    ch2, "stream2_" + str(j) + ".dat"))
                stream3.tofile(os.path.join(
                    ch3, "stream3_" + str(j) + ".dat"))

            j += 1
            if leftover_data0.shape[0] != 0:
                stream0 = [leftover_data0]
                stream1 = [leftover_data1]
                stream2 = [leftover_data2]
                stream3 = [leftover_data3]
            else:
                stream0 = []
                stream1 = []
                stream2 = []
                stream3 = []
            if leftover_header.shape[0] != 0:
                streamHeader = [leftover_header]
            else:
                streamHeader = []
        else:
            stream0.append(data0)
            stream1.append(data1)
            stream2.append(data2)
            stream3.append(data3)
            streamHeader.append(header)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted.")
