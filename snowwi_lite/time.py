"""
    Time handling relatad functions.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 13, 2024 - MCT
"""

import numpy as np


# Number of seconds between Unix epoch (1 January 1970) and GPS epoch (6 January 1980)
GPS_UNIX_EPOCH_DIFF = 315964800

# Number of seconds in a week
SECONDS_IN_WEEK = 604800

# GPS leap seconds (ahead of UNIX)
LEAP_SECONDS = 18

def gps_to_unix(gps_week, gps_seconds):
    return convert_week_to_seconds(gps_week, gps_seconds) + GPS_UNIX_EPOCH_DIFF


def convert_week_to_seconds(week, seconds):
    return week * SECONDS_IN_WEEK + seconds


def unix_to_gps_time(unix_time):
    # Calculate GPS time in seconds
    gps_time_seconds = unix_time - GPS_UNIX_EPOCH_DIFF
    
    # Calculate GPS weeks and the remaining seconds
    gps_week = gps_time_seconds // SECONDS_IN_WEEK
    gps_seconds = gps_time_seconds % SECONDS_IN_WEEK
    
    return int(gps_week), int(gps_seconds)


def timestamp_from_header(header):
    left = 4*7
    right = left + 4
    radio_time = header[:, left:right]
    print(radio_time.shape)
    vec = np.array([2**16, 2**0, 2**48, 2**32])
    sum = np.dot(radio_time.astype(np.uint64), vec.astype(np.uint64))
    return sum/122.88e6/4