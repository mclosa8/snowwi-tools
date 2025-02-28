import numpy as np

from snowwi_tools.novatel import get_ecef, get_llh, get_velocities, get_ypr

from snowwi_tools.utils import average_n_rows

from scipy.constants import speed_of_light

# Number of seconds between Unix epoch (1 January 1970) and GPS epoch (6 January 1980)
GPS_UNIX_EPOCH_DIFF = 315964800

# Number of seconds in a week
SECONDS_IN_WEEK = 604800


def calculate_doppler_from_fl(attitude_dict, look, f0):
    # Technically, Doppler centroid is calculated using elevation angle
    # Ref: Davidson & Cumming, 1997

    wvl = speed_of_light / f0

    ypr = attitude_dict['ypr'][0]
    v_mags = attitude_dict['mag_vels']

    psi = np.deg2rad(ypr[0])
    delta = np.deg2rad(ypr[1])
    gamma = np.deg2rad(90 - look)

    s_psi = np.sin(psi)
    c_psi = np.cos(psi)

    s_delta = np.sin(delta)

    s_gamma = np.sin(gamma)
    c_gamma = np.cos(gamma)

    s_theta = s_gamma * s_psi + c_gamma * s_delta * c_psi

    f_dopp = 2 * s_theta * v_mags / wvl

    return f_dopp


def generate_doppler_model(yaw, pitch, roll, vel, wvl, prf, y_offset=0, p_offset=0):
    psi = np.deg2rad(yaw + y_offset)
    delta = np.deg2rad(pitch + p_offset)
    gamma = np.deg2rad(roll)

    azm_nyq = prf/2

    term1 = np.outer(
        np.sin(psi),
        np.sin(gamma)
    )

    term2 = np.outer(
        np.sin(delta)*np.cos(psi),
        np.cos(gamma)
    )

    s_theta = term1 + term2
    f_dopp = (2*s_theta.T*vel/wvl).T
    # Not sure if this is doing the right thing...
    aliased = np.mod(f_dopp + azm_nyq, prf) - azm_nyq
    print(f_dopp.max())
    print(aliased.max())

    return f_dopp, aliased


def get_attitude_for_doppler(novatel_df):

    time_array = np.array(
        novatel_df['Week'] + novatel_df['GPSSeconds'], dtype=float
    )

    xyz = get_ecef(novatel_df)
    ypr, ypr_means = get_ypr(novatel_df)
    v, v_mags = get_velocities(xyz, time_array)

    return ypr, ypr_means, v, v_mags


def pulse_pair_doppler(data, prf, av_factor=0):
    if av_factor < 1:
        av_factor = 1/av_factor
        print(f"Using the inverse of the average factor: {av_factor}")

    # Pulse-Pair Doppler
    pp_phase = np.conjugate(data[:-1:]) * data[1::]

    # Accum pulses
    return -np.angle(average_n_rows(pp_phase, av_factor)) * prf / 2 / np.pi


def doppler_centroids(data, az_length, rg_length):
    # TODO - implement doppler centroid calculation
    pass
