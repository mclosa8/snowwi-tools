
import numpy as np

import os
import time

from scipy import fft

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