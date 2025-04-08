
"""
    Time handling relatad functions.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 13, 2024 - MCT
"""

import numpy as np

import glob
import os

from snowwi_tools.lib.file_handling import get_headers_only

from snowwi_tools.utils import natural_keys

# Number of seconds between Unix epoch (1 January 1970) and GPS epoch (6 January 1980)
GPS_UNIX_EPOCH_DIFF = 315964800

# Number of seconds in a week
SECONDS_IN_WEEK = 604800

# GPS leap seconds (ahead of UNIX)
LEAP_SECONDS = 18


def gps_to_unix(gps_week, gps_seconds):
    return convert_week_to_seconds(gps_week, gps_seconds) + GPS_UNIX_EPOCH_DIFF - LEAP_SECONDS

def gps_sec_to_unix(gps_seconds):
    return gps_seconds + GPS_UNIX_EPOCH_DIFF - LEAP_SECONDS


def convert_week_to_seconds(week, seconds):
    return week * SECONDS_IN_WEEK + seconds


def seconds_to_week_seconds(seconds):
    # Calculate GPS weeks and the remaining seconds
    gps_week = seconds // SECONDS_IN_WEEK
    gps_seconds = seconds % SECONDS_IN_WEEK

    return int(gps_week), gps_seconds
    

def unix_to_gps_time(unix_time, output='week seconds'):
    # Calculate GPS time in seconds
    gps_time_seconds = unix_time - GPS_UNIX_EPOCH_DIFF + LEAP_SECONDS

    if output == 'week seconds':
        return seconds_to_week_seconds(gps_time_seconds)

    return gps_time_seconds


def timestamp_from_header(header):
    left = 4*7
    right = left + 4
    radio_time = header[:, left:right]
    print(radio_time.shape)
    vec = np.array([2**16, 2**0, 2**48, 2**32])
    sum = np.dot(radio_time.astype(np.uint64), vec.astype(np.uint64))
    return sum/122.88e6/4


def timestamp_from_header_4x2(header, time_offset=0):
    print(header.shape)
    vec = np.array([2**48, 2**32, 2**16, 2**0])
    sum = np.dot(header.astype(np.uint64),
                 vec.astype(np.uint64))
    return sum / 1000 + (time_offset + LEAP_SECONDS)  # GPS time in s


def timestamp_to_week_seconds(gps_timestamp):  # gps_timestamp in s
    week = np.floor(gps_timestamp / SECONDS_IN_WEEK)
    seconds = gps_timestamp % SECONDS_IN_WEEK
    return {'Week': week, 'GPSSeconds': seconds}


def timestamp_from_files(filename, n_datasamps: int = 100_000, n_headersamps: int = 4, mode='snowwi', output='unix'):
    if "*" in filename:  # List all files following a wildcard character
        filelist = glob.glob(os.path.abspath(filename))
        filelist = sorted(filelist, key=natural_keys)
    else:  # Assumed single file - TODO: improve this
        filelist = [filename]
    headers = get_headers_only(filelist, n_datasamps, n_headersamps)
    if mode == 'snowwi':
        timestamps = timestamp_from_header_4x2(headers)  # In GPS time
    elif mode == 'ettus':
        timestamps = timestamp_from_header(headers)  # In GPS time

    if output == 'unix':
        # Unix = GPS Time + 315964800 - LEAP_SECONDS
        return timestamps + GPS_UNIX_EPOCH_DIFF - LEAP_SECONDS
    return timestamps
