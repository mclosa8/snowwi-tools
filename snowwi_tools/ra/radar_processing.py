"""
    Radar processing -related functions and dictionaries.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 13, 2024 - MCT
"""

import numpy as np

import os
import psutil

from multiprocessing import Pool
from scipy.interpolate import CubicSpline

from scipy.signal import correlate
from snowwi_tools.lib.signal_processing import exp_chirp, butter_bandpass_filter

from snowwi_tools.ra.data_handling import average_submatrix


def compensate_range_loss(data, range_bins, order=1):
    return data*range_bins**order


def compress(data, f_h, f_l, tp, fs, type='down', window='hamming', pulse='causal', precision='double'):
    n = int(fs*tp)

    if pulse == 'non-causal':
        # print('Using non-casual.')
        t = np.linspace(-tp/2, tp/2, n)
    else:
        # print('Using casual.')
        t = np.linspace(0, tp, n)

    exp_ref = exp_chirp(t, f_l, tp, f_h, type)

    if window == 'hanning':
        # print('Using hanning window to compress...')
        exp_ref = exp_ref*np.hanning(len(exp_ref))
    elif window == 'hamming':
        # print('Using hamming window to compress...')
        exp_ref = exp_ref*np.hamming(len(exp_ref))
    else:
        print('No windowing applied...')

    # print('Allocating memory...')
    if precision == 'single':
        dtype = np.complex64
        print(precision)
    else:
        dtype = np.complex128
        print(precision)
    compressed_data = np.zeros_like(data, dtype=dtype)
    # print('Memory allocated.')

    ctr = 0

    for row in data:
        compressed_data[ctr] = correlate(
            row, exp_ref, mode='same', method='fft')
        ctr = ctr + 1
        # if ctr % 100 == 0:
            # print(f"Counter: {ctr}")

    return compressed_data


def range_loss_correct(scene):
    print("Correcting range loss....")
    idxs = int(scene.shape[0] / 2)
    ptp = 20*np.log10(abs(scene[idxs:idxs+20].T))
    ptp_mean = np.mean(ptp, axis=1)

    rg_samps = np.arange(len(ptp_mean))

    ptp_mean_fit = np.polyfit(rg_samps, ptp_mean, 1)

    rg_comp = rg_samps*(-ptp_mean_fit[0])
    assert len(rg_comp) == scene.shape[1], "Invalid scene shape"
    return scene*10**(rg_comp/20)


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


def filter_and_compress(data, params_dict, chirp_dict, ml, order=6):
    # Filter according to specified band
    print('Filtering data according to:')
    print(params_dict)
    filtered = butter_bandpass_filter(
        data,
        params_dict['lowcut'],
        params_dict['highcut'],
        chirp_dict['fs'],
        order=order
    )

    # Range compress and return
    print("Compressing data using:")
    print(chirp_dict)
    compressed = compress(
        filtered,
        chirp_dict['f_h'],
        chirp_dict['f_l'],
        chirp_dict['tp'],
        chirp_dict['fs'],
        chirp_dict['chirp_type']
    )

    if ml:
        print(f"Multilooking... {ml}")
        return average_submatrix(
            abs(compressed),
            ml[0],
            ml[1]
        )
    return compressed
