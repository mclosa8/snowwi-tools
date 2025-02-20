# Library for reading the NovAtel file - SNOWWI
import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

SECONDS_PER_WEEK = 604_800

def read_novatel(file_path, skiprows=18, column_names=None):
    """
    Reads the NovAtel file from given path and returns it as a Pandas dataframe.

    Inputs:
        - file_path: Path to the NovAtel .txt file
        - skiprows: Number of rows to ommit from beginning of file.
                    Default: 18 rows (Ok for SNOWWI)
        - column_names: List with column names in NovAtel file.
                        Default value is up to date for 2024 data.

    Outputs:
        - dataframe with the entirety of the NovAtel data                    
    """

    # Define the column names
    if (column_names == None):
        column_names = [
            "GPSTime", "Date", "Week", "GPSSeconds",
            "Latitude", "Longitude", "H-Ell", "H-MSL",
            "Undulation", "X-ECEF", "Y-ECEF", "Z-ECEF",
            "Pitch", "Roll", "Heading", "GPSCOG"
        ]
    df = pd.read_csv(
        file_path,
        skiprows=skiprows,  # Adjust this based on the number of lines in the metadata
        sep='\s+',
        names=column_names,
        low_memory=False
    )

    df['GPSSeconds'] = pd.to_numeric(df['GPSSeconds'], errors='coerce')
    df['Week'] = pd.to_numeric(df['Week'], errors='coerce')

    return df


def get_ecef(novatel, start_idx=0, end_idx=-1):
    """
    Reads the NovAtel dataframe and builds a 3xN Numpy array with the GPS
    position data in ECEF coordinates. Can be previously truncated or not.

    Inputs:
        - novatel: Pandas dataframe from read with read_novatel()
                   NOTE: Can be truncated before passing it as argument.
        - start_idx: Index of last element -1 to consider. Default: first element in df.
        - end_idx: Index of last element to consider. Default: last element in df.

    Outputs:
        - ecef: 3xN Numpy array with the ECEF coordinates. Rows will contain the
                individual XYZ coordinates respectively. I.e.:
                - X_ECEF = ecef[0]
                - Y_ECEF = ecef[1]
                - Z_ECEF = ecef[2]
    """
    if end_idx == -1:
        end_idx = None

    x = np.array(novatel['X-ECEF'][start_idx:end_idx], dtype=float)
    y = np.array(novatel['Y-ECEF'][start_idx:end_idx], dtype=float)
    z = np.array(novatel['Z-ECEF'][start_idx:end_idx], dtype=float)

    ecef = np.vstack((x, y, z))  # Set of coordinates every row
    print(ecef.shape)
    return ecef


def get_llh(novatel, start_idx=0, end_idx=-1):
    """
    Reads the NovAtel dataframe and builds a 3xN Numpy array with the GPS
    position data in LLH coordinates. Can be previously truncated or not.

    Inputs:
        - novatel: Pandas dataframe from read with read_novatel()
                   NOTE: Can be truncated before passing it as argument.
        - start_idx: Index of last element -1 to consider. Default: first element in df.
        - end_idx: Index of last element to consider. Default: last element in df.

    Outputs:
        - llh: 3xN Numpy array with the LLH coordinates. Rows will contain the
                individual LLH coordinates respectively. I.e.:
                - LAT = ecef[0]
                - LON = ecef[1]
                - HEI = ecef[2] (ellipsoidal height)
    """

    if end_idx == -1:
        end_idx = None

    lat = np.array(novatel['Latitude'][start_idx:end_idx], dtype=float)
    lon = np.array(novatel['Longitude'][start_idx:end_idx], dtype=float)
    h = np.array(novatel['H-Ell'][start_idx:end_idx], dtype=float)

    llh = np.vstack((lat, lon, h))  # Set of coordinates every row
    print(llh.shape)
    return llh


def get_ypr(novatel, start_idx=0, end_idx=-1):
    """
    Reads the NovAtel dataframe and builds a 3xN Numpy array with the attitude
    data in degrees. Can be previously truncated or not.

    Inputs:
        - novatel: Pandas dataframe from read with read_novatel()
                   NOTE: Can be truncated before passing it as argument.
        - start_idx: Index of last element -1 to consider. Default: first element in df.
        - end_idx: Index of last element to consider. Default: last element in df.

    Outputs:
        - ypr: 3xN Numpy array with the LLH coordinates. Rows will contain the
                individual attitude parameters respectively. Follows YPR order.
                I.e.:
                - yaw   = ypr[0] (calculated from heading and Course over Ground)
                - pitch = ypr[1] (absolute)
                - roll  = ypr[2] (absolute)
        - ypr_means: 3x1 Nump array with the averages of each attitude parameter.
                     Follows YPR order.
                - mean_yaw   = ypr_means[0]
                - mean_pitch = ypr_means[1]
                - mean_roll  = ypr_means[2]

    """

    if end_idx == -1:
        end_idx = None
    roll = np.array(novatel['Roll'][start_idx:end_idx], dtype=float)
    pitch = np.array(novatel['Pitch'][start_idx:end_idx], dtype=float)
    heading = np.array(novatel['Heading'][start_idx:end_idx], dtype=float)

    cog = np.array(novatel['GPSCOG'][start_idx:end_idx])
    cog = np.array(
        [cog_i if cog_i < 180 else cog_i - 360 for cog_i in cog],
        dtype=float)

    yaw = np.mod(heading - cog + 360, 360)

    yaw = np.array(
        [y if y < 180 else y - 360 for y in yaw],
        dtype=float
    )

    ypr_means = (
        np.mean(yaw)*np.ones_like(yaw),
        np.mean(pitch)*np.ones_like(pitch),
        np.mean(roll)*np.ones_like(roll),
    )

    return np.vstack((yaw, pitch, roll)), ypr_means


def get_velocities(ecef, times):

    # Velocity = d_xyz/dt
    d_xyz = np.diff(ecef, axis=1)

    # Assuming constant sampling in time
    dt = np.diff(times)

    v = d_xyz/dt

    t_init = np.average([times[0], times[1]])
    t_end = np.average([times[-2], times[-1]])

    t_cs = np.linspace(t_init, t_end, len(dt))

    # Let's interpolate to obtain the original shape
    cs_x = CubicSpline(t_cs, v[0])
    cs_y = CubicSpline(t_cs, v[1])
    cs_z = CubicSpline(t_cs, v[2])

    v_interp = np.array([
        cs_x(times),
        cs_y(times),
        cs_z(times)
    ])

    v_mags = np.linalg.norm(v_interp, axis=0)

    return v_interp, v_mags


def novatel_to_dict(novatel_file):
    raw = np.loadtxt(novatel_file, skiprows=15, dtype=str,
                     usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    labels = raw[0]
    units = raw[1]
    data = raw[2:]
    novatel_dict = {}
    for i in range(len(labels)):
        novatel_dict[labels[i]] = {
            'unit': units[i],
            'data': data[:, i].astype(float)
        }

    return novatel_dict


# TDBP uses by default line heading in the positive interval [0, 360].
def get_cog(novatel, non_causal=False):
    cog = novatel['GPSCOG'].to_numpy() # COG by default is [0, 360].
    print(type(cog))
    if non_causal: # Normalize cog to [-180, 180] interval
        cog = np.array(
            [cog_i if cog_i < 180 else cog_i - 360 for cog_i in cog]
        )
    return cog


def get_attitude_dictionary(novatel_df, fl_info):
    """
    Returns the position/attitude of the flightline specified by fl_info,
    retrieved from the excel database.

    Inputs:
    ----------
        - novatel_df: NovAtel data loaded as Pandas dataframe
        - fl_info: Flightline information retrieved from the excel database, as Pandas dataframe.

    Outputs:
    ----------
        - Python dictionary: 
            {'xyz': ECEF coordinates of flightline,
             'llh': Lat, Lon, Hei of flightline,
             'ypr': Yaw, pitch, roll of flightline,
             'vels': Velocities wrt. ECEF coordinates}
    """
    # Get flightline from full file
    flightline = retrieve_flightline(novatel_df, fl_info)
    
    # Get time vector from flightline for velocities from GPS epoch
    time_vect = np.array(flightline['GPSSeconds'], dtype=float)
    time_vect += SECONDS_PER_WEEK * np.array(flightline['Week'])

    ecef = get_ecef(flightline)
    v_ecef, v_mags = get_velocities(ecef, time_vect)

    return {
        'xyz': ecef,
        'ypr': get_ypr(flightline),
        'llh': get_llh(flightline),
        'ecef_vels': v_ecef,
        'mag_vels': v_mags,
        'time': time_vect
    }

def retrieve_flightline(df, fl_info):
    return df.loc[(df['GPSSeconds'] >= fl_info['Start']) & (df['GPSSeconds'] <= fl_info['Stop'])]


def read_excel_database(flightline_xl, campaign):
    # Read excel and retrieve flightline information
    # Use pandas to read Excel
    maxcol = 15
    cols = np.arange(maxcol)
    with pd.ExcelFile(flightline_xl) as xls:
        df1 = pd.read_excel(xls, campaign, skiprows=1, usecols=cols)
    print(df1.keys())
    return df1


def retrieve_date_from_excel_database(dataframe, date):
    # Get the columns from date
    flightlines = dataframe.loc[(dataframe['Date (local)'] == date)]
    flightlines = flightlines.reset_index()
    print(flightlines.shape)
    return flightlines

def read_excel_database_and_get_date(flightline_xl, campaign, date):
    df1 = read_excel_database(flightline_xl, campaign)
    return retrieve_date_from_excel_database(df1, date)


def ecef_2_tcn(ecef, time):
    
    # Reference time to 0
    time = time - time[0]

    ref = np.mean(ecef, axis=1)
    print(ref.shape)

    ecef_rel = ecef.T - ref
    print(ecef_rel.shape)

    ideal_x = np.polyval(
        np.polyfit(time, ecef[0], 1), time
    )
    ideal_y = np.polyval(
        np.polyfit(time, ecef[1], 1), time
    )
    ideal_z = np.polyval(
        np.polyfit(time, ecef[2], 1), time
    )

    ideal_ecef = np.vstack((
        ideal_x,
        ideal_y,
        ideal_z
    ))

    print(ideal_ecef.shape) # 3xN
    print(ideal_ecef[:, 0], ideal_ecef[:, -1])

    diff = ideal_ecef[:, -1] - ideal_ecef[:, 0]

    t_unit = diff/np.linalg.norm(diff)
    n_unit = ref / np.linalg.norm(ref)
    c_unit = np.cross(n_unit, t_unit)

    tcn = np.vstack((
        np.dot(ecef_rel, t_unit),
        np.dot(ecef_rel, c_unit),
        np.dot(ecef_rel, n_unit),
    ))

    return tcn