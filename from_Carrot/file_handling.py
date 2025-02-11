"""
   File handling (raw data) -related functions and dictionaries.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 13, 2024 - MCT
"""

import numpy as np

import boto3
import glob
import h5py
import os
import time

from radar_processing import compress
from aasignal import butter_bandpass_filter
from utils import natural_keys


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

    dict = read_and_reshape(data_path, N,
                            header_samples=header_samples,
                            skip_samples=skip_samples,
                            truncate=last_samp)
    reshaped_data = dict['data']
    headers = dict['headers']
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
                               type=chirp['chirp_type'],
                               window=window)
    print('Data compressed.')
    del (reshaped_data)

    return {
        'data': compressed_data,
        'headers': headers,
        'timestamp': timestamp
    }


def get_timestamp_from_filename(filename):
    full = filename.split('_')[-1].split('.')[0]
    fract = filename.split('_')[-1].split('.')[1]
    return full, fract


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


def list_files_from_dir(directory, fr=0, to=-1):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    datafile_list = glob.glob(os.path.join(directory, '*.dat'))
    datafile_list.sort(key=natural_keys)
    pp.pprint(datafile_list)
    pp.pprint(f"Number of files: {len(datafile_list)}")

    timestamps_from_files = [
        float(f.split('/')[-1].split('_')[-1][:-4]) for f in datafile_list]
    # print(timestamps_from_files)
    return datafile_list[fr:to], timestamps_from_files[fr:to]


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


def read_and_reshape(filename, N, header_samples=0, skip_samples=0, truncate=None):
    print(f"Reading file:")
    print(f"    {filename}")

    print(N, header_samples, skip_samples)
    print(N + header_samples)
    n = N + header_samples

    data = np.fromfile(filename, dtype=np.int16)
    data = data.reshape(-1, N + header_samples)[:, :truncate + header_samples]
    print(data.shape)

    headers = data[:, :header_samples].astype(np.uint16)
    print(headers.shape)

    data = np.delete(data, np.s_[:skip_samples + header_samples], axis=1)
    print(data.shape)

    timestamp = float(filename.split('_')[-1][:-4])
    timestamps = np.ones(data.shape[0]) * timestamp

    return {
        'data': data,
        'timestamp': timestamps,
        'headers': headers
    }


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
