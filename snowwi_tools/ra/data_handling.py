"""
    Data handling relatad functions.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 13, 2024 - MCT
"""

import bottleneck as bn
import numpy as np

import boto3
import os

from scipy.ndimage import convolve


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


def grouped_avg(myArray, N=2):
    result = np.cumsum(myArray, 0, dtype='complex')[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result


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
