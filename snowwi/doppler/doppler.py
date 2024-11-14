import numpy as np

from snowwi_novatel import get_ecef, get_llh, get_velocities, get_ypr

# Number of seconds between Unix epoch (1 January 1970) and GPS epoch (6 January 1980)
GPS_UNIX_EPOCH_DIFF = 315964800

# Number of seconds in a week
SECONDS_IN_WEEK = 604800

def generate_doppler_model(yaw, pitch, roll, vel, wvl, prf, y_offset=0, p_offset=0):
    psi = np.deg2rad(yaw + y_offset)
    delta  = np.deg2rad(pitch + p_offset)
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

    sin_theta = term1 + term2
    f_dopp = (2*sin_theta.T*vel/wvl).T
    aliased = np.mod(f_dopp + azm_nyq, prf) - azm_nyq # Not sure if this is doing the right thing...
    
    return aliased

def get_attitude_for_doppler(novatel_df):
    
    time_array = np.array(
        novatel_df['Week'] + novatel_df['GPSSeconds'], dtype=float
    )

    xyz = get_ecef(novatel_df)
    ypr, ypr_means = get_ypr(novatel_df)
    v, v_mags = get_velocities(xyz, time_array)

    return ypr, ypr_means, v, v_mags