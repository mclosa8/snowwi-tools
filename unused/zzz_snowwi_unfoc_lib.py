# Ancillary functions
import bottleneck as bn
import numpy as np

from multiprocessing import Pool

from copy import deepcopy
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage import convolve
from scipy.signal import chirp, butter, sosfilt, filtfilt, correlate
from scipy import fft

import boto3
import glob
import h5py
import os
import psutil
import re
import time

channel_dictionary = {
    'low': {
        0: 'chan0',
        1: 'chan2'
    },
    'high': {
        0: 'chan1',
        1: 'chan3'
    }
}


def atoi(text): return int(text) if text.isdigit() else text


def average_n_rows(matrix, n):
    # Number of rows in the matrix
    N = matrix.shape[0]

    # Number of full groups of n rows
    num_full_groups = N // n

    # Number of remaining rows
    remainder = N % n

    # If there are no remaining rows, we can simply reshape and take the mean
    if remainder == 0:
        return matrix.reshape(num_full_groups, n, -1).mean(axis=1)

    # If there are remaining rows, we need to handle them separately
    else:
        # Average each full group of n rows
        averaged_matrix = matrix[:num_full_groups *
                                 n].reshape(num_full_groups, n, -1).mean(axis=1)

        # Average the remaining rows and append to the result
        averaged_matrix = np.vstack(
            (averaged_matrix, matrix[-remainder:].mean(axis=0)))

    return averaged_matrix


def average_rows(matrix, M):
    N = matrix.shape[0]  # get number of rows
    num_groups = N // M

    result = np.zeros((num_groups, matrix.shape[1]), dtype='complex64')

    for i in range(num_groups):
        start = i * M
        end = start + M
        rows = matrix[start:end]
        mean = np.mean(rows, axis=0)

        result[i, :] = mean

    return result


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def butter_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, analog=False, btype='high', output='ba')
    return sos


def butter_highpass_filter(data, highcut, fs, order=5):
    b, a = butter_highpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, analog=False, btype='low', output='sos')
    return sos


def butter_lowpass_filter(data, highcut, fs, order=5):
    sos = butter_lowpass(highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def compensate_range_loss(data, range_bins, order=1):
    return data*range_bins**order


def compress(data, f_h, f_l, tp, fs, N, slope='down', window='hamming', pulse='causal'):
    n = int(fs*tp)
    # Non-causal signal
    t = np.linspace(0, tp, n)
    if pulse == 'non-causal':
        t = np.linspace(-tp/2, tp/2, n)
    # cos_ch = chirp(t, f_l, tp, f_h)
    # sin_ch = chirp(t, f_l, tp, f_h, phi=-90)
    exp_ref = exp_chirp(t, f_l, tp, f_h)
    if slope == 'down':
        exp_ref = exp_ref[::-1]
        # cosRef = cosRef[::-1]
        # sinRef = sinRef[::-1]

    if window == 'hanning':
        print('Using hanning window to compress...')
        exp_ref = exp_ref*np.hanning(len(exp_ref))
    elif window == 'hamming':
        print('Using hamming window to compress...')
        exp_ref = exp_ref*np.hamming(len(exp_ref))
        # cosRef = cosRef*np.hamming(len(cosRef))
        # sinRef = cosRef*np.hamming(len(sinRef))
    else:
        print('No windowing applied...')

    print('Allocating memory...')
    compressed_data = np.zeros_like(data, dtype=np.complex64)
    print('Memory allocated.')

    ctr = 0

    for row in data:
        compressed_data[ctr] = correlate(
            row, exp_ref, mode='same', method='fft')
        ctr = ctr + 1
        if ctr % 100 == 0:
            print(f"Counter: {ctr}")

    return compressed_data


def gps_to_unix(gps_week, gps_seconds):
    return convert_week_to_seconds(gps_week, gps_seconds) + 315964800


def convert_week_to_seconds(week, seconds):
    return week*604800 + seconds


def exp_chirp(t, f0, t1, f1, phi=0):
    K = (f1 - f0) / t1
    phase = 2 * np.pi * (f0 * t + 0.5 * K * t * t)
    return np.exp(1j * (phase + phi * np.pi / 180))


def get_band_params(band, channel):
    band_params = {
        'low': {
            'f0': 13.64e9,
            'f_l': 143.04e6,
            'f_h': 223.04e6,
            'chirp_type': 'up',
            'lowcut': 140e6,
            'highcut': 225e6,
            'channels': {0: 'channel_dictionary["low"][0]', 1: 'channel_dictionary["low"][1]'}
        },
        'high': {
            'f0': 17.24e9,
            'f_l': 23.04e6,
            'f_h': 103.04e6,
            'chirp_type': 'down',
            'lowcut': 20e6,
            'highcut': 105e6,
            'channels': {0: 'channel_dictionary["high"][0]', 1: 'channel_dictionary["high"][1]'}
        }
    }

    if band not in band_params:
        raise Exception("No valid band selected")

    if channel not in band_params[band]['channels']:
        raise Exception("No valid channel selected")

    params = band_params[band].copy()
    params['channel'] = eval(params['channels'][channel])
    del params['channels']

    return tuple(params.values())


def grouped_avg(myArray, N=2):
    result = np.cumsum(myArray, 0, dtype='complex')[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result


def kn_to_mps(kn):
    return kn * 0.514444


def multilook_old(image, xLooks, yLooks):

    # Adjusts image to be able to be multilooked (no extra rows/columns)

    if image.shape[0] % yLooks != 0:
        image = image[:, :-(image.shape[0] % yLooks)]

    elif image.shape[1] % xLooks != 0:
        image = image[:, :-(image.shape[1] % xLooks)]

    ###  Take in an image, multilook in range and azimuth, return multilooked image  ###

    y_ = np.zeros_like(image[0::yLooks])
    for r in range(yLooks):
        y_ = y_ + image[r::yLooks]

    yLooked = y_ / yLooks

    x = np.zeros_like(yLooked[:, 0::xLooks])
    for a in range(xLooks):
        x = x + yLooked[:, a::xLooks]

    multilooked = x / xLooks
    return multilooked


def multilook_image(image, n, m):
    """
    Multilooks an image (matrix) by a factor of nxm.

    Args:
    image (numpy.ndarray): Input image matrix.
    n (int): Factor for multilooking along the rows.
    m (int): Factor for multilooking along the columns.

    Returns:
    numpy.ndarray: Multilooked image matrix.
    """
    height, width = image.shape
    new_height = (height + n - 1) // n
    new_width = (width + m - 1) // m
    multilooked_image = np.zeros((new_height, new_width), dtype=np.complex64)

    for i in range(new_height):
        row_start = i * n
        row_end = min(row_start + n, height)
        for j in range(new_width):
            col_start = j * m
            col_end = min(col_start + m, width)
            block = image[row_start:row_end, col_start:col_end]
            multilooked_image[i, j] = np.sum(block) / block.size

    return multilooked_image


def natural_keys(text): return [atoi(c) for c in re.split(r'(\d+)', text)]


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


def read_and_reshape(fileName, N, header_samples=0, skip_samples=0, truncate=None):
    print(f"Reading file:")
    print(f"    {fileName}")
    data = np.fromfile(fileName, dtype=np.int16)
    data = data.reshape(-1, N)[:, :truncate]
    headers = data[:, :header_samples].astype(np.uint16)
    data = np.delete(data, np.s_[:skip_samples], axis=1)
    return data, headers


def read_options_txt(bucket, prefix, file, source='fs'):
    params_fname = os.path.join(bucket, prefix, file)
    if source == 'fs':  # Read from the filesystem
        print(f"Reading options from filesystem: {params_fname}")
        inpFile = open(params_fname, 'r')
    elif source == 's3':  # Read from S3
        print(f"Reading options from S3: {params_fname}")
        s3 = boto3.resource('s3')
        bucket = params_fname.split('/')[0]
        print(f"Bucket: {bucket}")
        key = '/'.join(params_fname.split('/')[1:])
        print(f"Key: {key}")
        obj = s3.Object(bucket, key)
        print(f"Object: {obj}")
        inpFile = obj.get()['Body'].read().decode('utf-8').splitlines()

    for l in inpFile:
        lparts = l.rpartition(' ')

        if (len(lparts) > 0):
            var = lparts[0].strip(' \r\n')
            val = lparts[2].strip(' \r\n')
            if var == '--N0':
                N0 = int(val)*4
            if var == '--seconds_per_file':
                spf = eval(val)

    return N0, spf


def rolling_average(matrix, n):
    """
    Calculates the rolling average of n pulses for each element in the input matrix.

    Parameters:
        matrix (numpy.ndarray): Input matrix of shape (N, M).
        n (int): Number of pulses for rolling average.

    Returns:
        numpy.ndarray: Matrix containing the rolling average values.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix must be a numpy array.")

    if len(matrix.shape) != 2:
        raise ValueError("Input matrix must be two-dimensional.")

    if n <= 0:
        raise ValueError("Number of pulses must be positive.")

    return bn.move_mean(matrix, window=n, axis=0, min_count=1)


def timestamp_from_header(header):
    left = 4*7
    right = left + 4
    radio_time = header[:, left:right]
    print(radio_time.shape)
    vec = np.array([2**16, 2**0, 2**48, 2**32])
    sum = np.dot(radio_time.astype(np.uint64), vec.astype(np.uint64))
    return sum/122.88e6/4


def voltage_to_log_power(voltage):
    return 10 * np.log10(np.abs(voltage)**2)


def download_from_s3(bucket, key, local_path):
    s3 = boto3.client('s3')
    with open(local_path, 'wb') as f:
        s3.download_fileobj(bucket, key, f)
    print(f"Downloaded {key} from {bucket} to {local_path}")


def save_h5_to_s3(h5_file, bucket, key):
    s3 = boto3.resource('s3')
    filename = h5_file.split('/')[-1]
    s3.Bucket(bucket).upload_file(h5_file, os.path.join(key, filename))
    print(f"Uploaded {h5_file} to {bucket}/{key}/{filename}")


def read_and_compress_aws(bucket, key, local_path,
                          N, header_samples, skip_samples, last_samp,
                          chirp, window, filter=None
                          ):
    print(bucket)
    print(key)
    print(local_path)
    filename = key.split('/')[-1]
    t1 = time.time()
    print(f"Downloading file: {filename} from {bucket+'/'+key}\n")
    download_from_s3(bucket, key, os.path.join(local_path, filename))
    print(f"Done downloading fi(le: {key} from {bucket}")
    print(f"Elapsed: {time.time() - t1}")
    timestamp = filename.split('_')[-1][:-4]
    ftd = os.path.join(local_path, filename)
    print(ftd)
    reshaped_data, headers = read_and_reshape(ftd, N,
                                              header_samples=header_samples,
                                              skip_samples=skip_samples,
                                              truncate=last_samp)
    print(f'Removing file: {ftd} from {local_path}...')
    os.remove(ftd)
    if filter is not None:
        # filter[0] = lowcut
        # filter[1] = highcut
        # filter[2] = fs
        reshaped_data = butter_bandpass_filter(
            reshaped_data, filter[0], filter[1], filter[2])

    print(f"Reshaped data shape: {reshaped_data.shape}")
    print('Compressing data...')
    compressed_data = compress(reshaped_data,
                               chirp['f_h'],
                               chirp['f_l'],
                               chirp['tp'],
                               chirp['fs'],
                               N, slope=chirp['chirp_type'],
                               window=window)
    print('Data compressed.')
    del reshaped_data

    return {
        'data': compressed_data,
        'headers': headers,
        'timestamp': timestamp
    }


def read_and_compress_local(data_path,
                            N, header_samples, skip_samples, last_samp,
                            chirp, window, filter=None
                            ):
    print(data_path)
    print(N, header_samples, skip_samples, last_samp)
    filename = data_path.split('/')[-1]
    timestamp = filename.split('_')[-1][:-4]
    print('timestamp', timestamp)
    print(f"Reading {data_path} from Synology disk...")

    reshaped_data, headers = read_and_reshape(data_path, N,
                                              header_samples=header_samples,
                                              skip_samples=skip_samples,
                                              truncate=last_samp)
    if filter is not None:
        # filter[0] = lowcut
        # filter[1] = highcut
        # filter[2] = fs
        reshaped_data = butter_bandpass_filter(
            reshaped_data, filter[0], filter[1], filter[2])

    print(f"Reshaped data shape: {reshaped_data.shape}")
    print('Compressing data...')
    compressed_data = compress(reshaped_data,
                               chirp['f_h'],
                               chirp['f_l'],
                               chirp['tp'],
                               chirp['fs'],
                               N, slope=chirp['chirp_type'],
                               window=window)
    print('Data compressed.')
    del (reshaped_data)

    return {
        'data': compressed_data,
        'headers': headers,
        'timestamp': timestamp
    }


def merge_h5_from_dir(dir_path, output_file, as_filetype='h5'):
    h5_files = glob.glob(os.path.join(dir_path, 'rc_tmp*.h5'))
    # h5_files = [f for f in os.listdir(dir_path) if f.endswith('.h5')]
    h5_files.sort(key=natural_keys)
    output_file = output_file.split('.')[0]
    out_path = os.path.join(dir_path, output_file)

    print(f"Files to merge: {h5_files}")

    if as_filetype == 'h5':
        out_path = out_path + '.h5'
        with h5py.File(out_path, 'w') as f:
            for i, h5_file in enumerate(h5_files):
                print(f"Reading file: {h5_file}")
                with h5py.File(os.path.join(dir_path, h5_file), 'r') as f2:
                    print(f2)
                    data = f2['data'][:]
                    headers = f2['headers'][:]
                    if i == 0:
                        f.create_dataset('data', data=data,
                                         maxshape=(None, data.shape[1]))
                        f.create_dataset('headers', data=headers,
                                         maxshape=(None, headers.shape[1]))
                    else:
                        f['data'].resize(
                            (f['data'].shape[0] + data.shape[0]), axis=0)
                        f['data'][-data.shape[0]:] = data
                        f['headers'].resize(
                            (f['headers'].shape[0] + headers.shape[0]), axis=0)
                        f['headers'][-headers.shape[0]:] = headers
                print(f"Done reading file: {h5_file}")
                print(f"Removing file: {h5_file}")
                os.remove(os.path.join(dir_path, h5_file))
                print(f"Done removing file: {h5_file}")

    elif as_filetype == 'npz':
        out_path = out_path + '.npy'
        data = []
        headers = []
        for i, h5_file in enumerate(h5_files):
            print(f"Reading file: {h5_file}")
            if i == 0:
                with h5py.File(os.path.join(dir_path, h5_file), 'r') as f2:
                    data = f2['data'][:]
                    headers = f2['headers'][:]
                num_rows = data.shape[0]
                num_cols = data.shape[1] + 1
                matrix = np.zeros((num_rows, num_cols), dtype=np.complex64)
                matrix[:, 1:] = data
                matrix[:, 0] = headers
            else:
                with h5py.File(os.path.join(dir_path, h5_file), 'r') as f2:
                    data = f2['data'][:]
                    headers = f2['headers'][:]
                num_rows = data.shape[0]
                matrix[num_rows*i:, 1:] = data
                matrix[num_rows*i:, 0] = headers
            print(f"Done reading file: {h5_file}")
            print(f"Removing file: {h5_file}")
            os.remove(os.path.join(dir_path, h5_file))
            print(f"Done removing file: {h5_file}")
        np.save(out_path, matrix)
    print(f"Done merging files to: {output_file}")
    return os.path.abspath(out_path)


def h5_opener(filename, label, thresh=0):  # regular version
    with h5py.File(filename, 'r') as hf:
        data = hf[label][:, thresh:]
    return data


def get_flightline_params_from_rc(filename):
    # Filename uses the following convention:
    #     rc_YYYYMMDDTHHMMSS_<band>_<channel>_<from_file>_<to_file>.h5
    elements = filename.split('_')
    flightline = elements[1]
    band = elements[2].split('-')[1]
    channel = elements[3]
    print(channel)

    channel_mapping = {
        ('low', 'chan0'): 0,
        ('low', 'chan2'): 1,
        ('high', 'chan1'): 0,
        ('high', 'chan3'): 1
    }

    channel = channel_mapping.get((band, channel), None)

    from_to = (eval(elements[4]), eval(elements[5].split('.')[0]))
    return flightline, band, channel, from_to


def compute_fft(data, axis):
    print("Calculating FFT...")
    fft_o = fft.fft(data, axis=axis, workers=-1).astype(np.complex64)
    fft_o = np.fft.fftshift(fft_o, axes=axis)
    print('Done!')
    return fft_o


def parallel_fft(data, axis=0):
    print("Using parallelized FFT...")
    # num_rows, num_cols = data.shape
    # num_processes = os.cpu_count()
    # chunks = np.array_split(data, num_processes, axis=1)
    # print(chunks[0].shape)
    # print(chunks[-1].shape)

    # results = p.starmap(compute_fft,
    #                     [(chunk, axis) for chunk in chunks])
    # print(results[0])
    # print(results[0].shape)
    # print(results[-1].shape)
    # output = np.hstack(results)
    # del(results)
    # print(output.shape)

    t1 = time.time()
    # output = fft.fft(data, axis=axis, workers=-1)
    # output = fft.fftshift(data, axes=0)
    output = compute_fft(data, axis=axis)
    print(f"Elapsed: {time.time() - t1}")

    return output


def compute_ifft(data, axis):
    print("Computing IFFT...")
    ifft_o = fft.ifft(data, axis=axis, workers=-1).astype(np.complex64)
    print("Done!")
    return ifft_o


def parallel_ifft(data, axis=0):
    print("Using parallelized IFFT...")
    num_rows, num_cols = data.shape
    num_processes = os.cpu_count()

    # Chunks in the az direction
    # chunks = np.array_split(data, num_processes, axis=1)

    # results = p.starmap(compute_ifft,
    #                     [(chunk, axis) for chunk in chunks])

    # output = np.hstack(results)
    t1 = time.time()
    # output = fft.ifft(data, axis=axis, workers=-1)
    output = compute_ifft(data, axis)
    print(f"Elapsed: {time.time() - t1}")

    return output


def make_chirp_dict(f0, f_l, f_h, tp, chirp_type, fs):
    return {
        'f0': f0,
        'f_l': f_l,
        'f_h': f_h,
        'tp': tp,
        'chirp_type': chirp_type,
        'fs': fs
    }


def list_files_from_bucket(bucket, prefix, flightline, channel):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    # Read from Bucket
    session = boto3.Session()
    s3 = session.resource('s3')

    print(s3)

    my_bucket = s3.Bucket(bucket)

    print(prefix)

    datafile_list = []

    for object in my_bucket.objects.filter(Prefix=prefix):
        if object.key.endswith('.dat'):
            datafile_list.append(object.key)

    datafile_list.sort(key=natural_keys)
    pp.pprint(datafile_list)
    pp.pprint(f"Number of files: {len(datafile_list)}")

    timestamps_from_files = [
        float(f.split('/')[-1].split('_')[-1][:-4]) for f in datafile_list]
    # print(timestamps_from_files)

    return datafile_list, timestamps_from_files


def list_files_from_dir(directory):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    datafile_list = glob.glob(os.path.join(directory, '*.dat'))
    datafile_list.sort(key=natural_keys)
    pp.pprint(datafile_list)
    pp.pprint(f"Number of files: {len(datafile_list)}")

    timestamps_from_files = [
        float(f.split('/')[-1].split('_')[-1][:-4]) for f in datafile_list]
    # print(timestamps_from_files)
    return datafile_list, timestamps_from_files


def define_files_to_process(datafile_list, read_from_to):
    dat_files = [
        file for file in datafile_list if file.split('.')[-1] == 'dat']

    print(read_from_to)

    if read_from_to == 'all':
        print(f"Reading all files in flightline directory.")
        first_file = 0
        last_file = len(dat_files) - 1
        print(
            f"Reading from file {dat_files[first_file]} to file {dat_files[last_file]} in flightline directory...\n")
    elif max(read_from_to) > len(dat_files):
        print(f"Invalid file indices. Reading to last file in flightline directory.")
        first_file = read_from_to[0]
        last_file = len(dat_files)-1
        print(
            f"Reading from file {dat_files[first_file]} to file {dat_files[last_file]} in flightline directory...\n")
    elif max(read_from_to) < 0:
        print(f"Invalid file indices. Reading from first file in flightline directory.")
        first_file = 0
        last_file = read_from_to[1]
        print(
            f"Reading from file {dat_files[first_file]} to file {dat_files[last_file]} in flightline directory...\n")
    else:
        first_file = read_from_to[0]
        last_file = read_from_to[1]
        print(
            f"Reading from file {dat_files[first_file]} to file {dat_files[last_file]} in flightline directory...\n")
    files_to_read = dat_files[first_file:last_file]
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(files_to_read)

    return files_to_read, first_file, last_file


def create_thread_pool(processes=os.cpu_count()):
    return Pool(processes=processes)


def phase_preserving_average(image, az_factor, rng_factor):
    """
    Phase-preserving average of pixels in a complex matrix.

    Parameters:
    - image: Complex matrix (2D array) containing image data.
    - kernel_size: Size of the averaging kernel (default is 3x3).

    Returns:
    - Smoothed complex matrix with preserved phase information.
    """

    # Define the averaging kernel
    kernel = np.ones((az_factor, rng_factor),
                     dtype=np.float32) / (az_factor * rng_factor)
    print(kernel.shape)

    # Separate the real and imaginary parts
    real_part = np.real(image)
    imag_part = np.imag(image)

    # Apply average filtering to the real and imaginary parts separately
    smoothed_real = convolve(real_part, kernel)
    smoothed_imag = convolve(imag_part, kernel)

    # Combine the smoothed real and imaginary parts
    smoothed_image = smoothed_real + 1j * smoothed_imag
    print(smoothed_image.shape)

    return smoothed_image


def combine_results(rets):
    num_of_files = len(rets)
    print(f"Number of files: {num_of_files}")

    num_samps = rets[0]['data'].shape[1]

    min_length = min([ret['data'].shape[0] for ret in rets])
    max_length = max([ret['data'].shape[0] for ret in rets])

    headers_len = rets[0]['headers'].shape[1]

    data = np.empty((num_of_files * max_length, num_samps), dtype=np.complex64)
    timestamps = np.empty(num_of_files*max_length)
    headers = np.empty((num_of_files*max_length, headers_len))

    if min_length != max_length:
        print(f"Minimum length: {min_length}")
        print(f"Maximum length: {max_length}")
        data = np.empty(((num_of_files - 1) * max_length +
                        min_length, num_samps), dtype=np.complex64)
        timestamps = np.empty((num_of_files - 1) * max_length + min_length)
        headers = np.empty(
            ((num_of_files - 1) * max_length + min_length, headers_len))

    for i, ret in enumerate(rets):
        data[i*max_length:(i+1)*max_length, :] = ret['data']
        timestamps[i*max_length:(i+1)*max_length] = ret['timestamp']
        headers[i*max_length:(i+1)*max_length, :] = ret['headers']

    print(data.dtype)

    return data, timestamps, headers


def process_rcmc_chunk(xin, data, Rfn, rng_samp):
    print('Correcting cell migration for given chunk.')
    az_samp = data.shape[0]
    rcmc = np.empty_like(data)
    for i in range(az_samp):
        x = xin - Rfn[i]
        cs = CubicSpline(x, data[i], extrapolate=False)
        # Replace NaNs from Extrapolate to 1s.
        rcmc[i] = np.nan_to_num(cs(np.arange(rng_samp)), nan=1)        
    print('Done!')
    return rcmc


def parallel_rcmc(data, lambda_, fs_az, fs_rng, Rmin, Rmax, vp):
    print(f"1. RAM usage: {psutil.virtual_memory().percent} %")
    print('Using parallel RCMC.')
    num_processes = os.cpu_count()
    print(f"Number of processes available: {num_processes}")
    print(data.dtype)
    az_samp = data.shape[0]
    rng_samp = data.shape[1]
    print(f"2. RAM usage: {psutil.virtual_memory().percent} %")

    fn = np.linspace(-fs_az/2, fs_az/2, az_samp)

    print(f"3. RAM usage: {psutil.virtual_memory().percent} %")

    R = np.linspace(Rmin, Rmax, rng_samp,
                    dtype=np.float32).reshape((-1, rng_samp))
    print(R.shape)
    print(f"4. RAM usage: {psutil.virtual_memory().percent} %")
    fn = (np.linspace(-fs_az/2, fs_az/2, az_samp,
          dtype=np.float32) ** 2).reshape((az_samp, -1))
    print(fn.shape)
    Rfn = fn.dot(R)
    print(Rfn[0])
    print(Rfn.dtype)
    del (fn)
    del (R)
    print(f"5. RAM usage: {psutil.virtual_memory().percent} %")
    print(Rfn.shape)
    np.multiply(lambda_**2, Rfn, out=Rfn)
    np.divide(Rfn, 8*vp**2, out=Rfn)
    print(Rfn.shape)
    print(f"6. RAM usage: {psutil.virtual_memory().percent} %")

    np.multiply(2*fs_rng/3e8, Rfn, out=Rfn)
    print(Rfn.shape)

    print(f"7. RAM usage: {psutil.virtual_memory().percent} %")

    xin = np.arange(rng_samp)

    # Create arguments for each row to be processed
    print(f"8. RAM usage: {psutil.virtual_memory().percent} %")

    # Use multiprocessing Pool to process rows in parallel
    # TODO - Use vstack to append results and forget about the append.
    # Or make a for loop making things memory-efficient.
    assert Rfn.shape == data.shape, "Matrix dimensions do not match."
    with Pool(num_processes) as p:
        Rfn = np.array_split(Rfn, num_processes)
        args_for_starmap = [(xin, chunk, Rfn[i], rng_samp)
                            for i, chunk in enumerate(np.array_split(data, num_processes))]
        results = p.starmap(process_rcmc_chunk, args_for_starmap)
    print(f"9. RAM usage: {psutil.virtual_memory().percent} %")
    print("Appending results...")
    return np.vstack(results)


def get_timestamp_from_filename(filename):
    full = filename.split('_')[-1].split('.')[0]
    fract = filename.split('_')[-1].split('.')[1]
    return full, fract


def average_submatrix(matrix, sub_size_vertical, sub_size_horizontal):
    nrows, ncols = matrix.shape
    print(matrix.shape)
    print(matrix.dtype)

    print(sub_size_vertical, sub_size_horizontal)

    # Calculate the number of averaged rows and columns
    avg_nrows = nrows // sub_size_vertical
    avg_ncols = ncols // sub_size_horizontal

    # Reshape the matrix into sub-matrices of custom sizes
    reshaped_matrix = matrix[:avg_nrows * sub_size_vertical, :avg_ncols * sub_size_horizontal].reshape(
        avg_nrows, sub_size_vertical, avg_ncols, sub_size_horizontal
    )
    print('Reshaped')
    print(reshaped_matrix.shape)

    # Compute the mean along the last two axes to get the average of each sub-matrix
    averaged_matrix = np.mean(reshaped_matrix, axis=(1, 3))
    print('Averaged')
    print(averaged_matrix.shape)

    return averaged_matrix


def range_correct(scene):
    print("Correcting range loss....")
    ptp = 20*np.log10(abs(scene[500:520].T))
    ptp_mean = np.mean(ptp, axis=1)

    rg_samps = np.arange(len(ptp_mean))

    ptp_mean_fit = np.polyfit(rg_samps, ptp_mean, 1)

    rg_comp = rg_samps*(-ptp_mean_fit[0])
    assert len(rg_comp) == scene.shape[1], "Invalid scene shape"
    return scene*10**(rg_comp/20)