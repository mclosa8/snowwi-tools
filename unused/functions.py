import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.signal import chirp
from scipy.interpolate import interp1d
from simplekml import Kml
import time
import re
import bottleneck as bn
import pandas as pd


######################  Functions for Importing Files  #####################################################################################################################

def list_files(filepath, key='rcv', pnt=True):

    ##  This will return a list, and print all file paths for radar data in specified directory  ##
    ##  filepath is a directory 
    ##  Returns a list of all file names

    fileList = os.listdir(filepath)
    files = []

    for name in fileList:  ## finding all files in the directory with the desired keyword
        if key and '.dat' in name:
            files.append(os.path.join(filepath, name))

    files = sorted(files, key=lambda x: float(x.split('_')[-1][:-4]))#.split('.')[0]))  ## sorting by time stamp... gets weird order without this

    if pnt == True:
        print(f'Files with keyword: {key}')
        print('-----------------------------------------------------------------------------------')

        for f in files:
            print(f)

        print('-----------------------------------------------------------------------------------')
        print(f'Num Files listed: {len(files)}')

    return files

def read_first(file, N):

    data = np.fromfile(file, dtype=np.int16, count=N)

    return data

def checkdir(directory, numfiles):  

    ##############################
    ###  give this the sentinel directory with the path incliuding the disk then nvme
    ###  will return a list of file paths from that flight that have 'numfiles' in the data collect
    ###  Idea being that the "good" data collects are opnes that we let run for a while
    ### Hence, they will have many files in the chan0 folder (or chan1 chan2 chan3)
    ##############################

    datacollects = os.listdir(directory)

    lines = []

    for i in datacollects:


        run = os.path.join(directory, i)
        channel = os.path.join(run, 'chan0')


        if 'Thumbs.db' not in run and len(os.listdir(channel)) >= numfiles and 'chan0_Quick_look.png' not in os.listdir(run):
            lines.append(run)
                
    return lines    

def mk_output_dir(directory):

    outputpath = os.path.join(directory, 'outputs')
    print(f'Checking for outputs directory at {outputpath}')

    if os.path.exists(outputpath) == False:
        os.mkdir(outputpath)
        print(f'Directory created at {outputpath}')

    else:
        print(f'Output path exists at {outputpath}')

    return outputpath


def load_data(file_list, start_idx=0, stop_idx=-1, start_sample=0, stop_sample=0):

    ###  Given a list of file paths to .dat files of binary radar data
    ###  Looks at date code, determines how many samples should belong in each window
    ###  Returns a single matrix that holds all desired data

    first_path = file_list[0]
    pattern = r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})'  ## using this pattern to find a timestamp

    year = int(re.search(pattern, first_path).group(1)) 

    if year < 2024:
        n_samp = 60000

    else:
        n_samp = 64000

    file_data = []
    idx = 0
    total_pulses = 0

    for f in file_list[start_idx:stop_idx]:  ## iterate through files of given indeces
        temp_data = np.fromfile(f, dtype=np.int16)
        temp_data = temp_data.reshape(-1, n_samp)

        print(f'File Index: {idx} with shape {temp_data.shape}')
        file_data.append(temp_data)  ##  adding data to a list. file_data becomes a list of 2D matrices, each item corresponds to a single .dat file

        idx +=1
        total_pulses+= temp_data.shape[0]  ##  this is counting number of pulses in each file

    full_data = np.zeros((total_pulses, n_samp))  ## Allocating memory for the full data array

    idx = 0   ## resetting counter

    for data in file_data:  ## iterate through items (2D matrices) adding them to specific indeces of full data matrix
        n_pulses = data.shape[0]
        full_data[idx:idx+n_pulses] = data  ## indexing where each data chunk belongs
        del(data)
        idx += n_pulses

    del(file_data)
    del(idx)
    del(total_pulses)

    if start_sample or stop_sample != 0:
        full_data = full_data[:, start_sample:stop_sample]


    print(f'Full Data Shape: {full_data.shape}')
    print(f'First Filename: {file_list[start_idx]}')
    print(f'Last Filename: {file_list[stop_idx]}')

    return full_data/(2**4)  ###  divide by 16 because it is 12 bit data loaded as 16 bit data

def readNovatel(novatel_directory, novatel_filename, start_time, stop_time):
    novatel_file = os.path.join(novatel_directory, novatel_filename)

    cols = ['GPSTime [HMS]', 'Date', 'Week', 'GPSTime [sec]', 'Latitude', 'Longitude', 'H-Ell', 'X-ECEF', 'Y-ECEF', 'Z-ECEF', 'Pitch', 'Roll', 'Heading', 'COG']
    novatel = pd.read_csv(novatel_file, delim_whitespace=True, skiprows=17, skipfooter=3, on_bad_lines='warn', names=cols, engine='python')

    gps_start = start_time - (novatel['Week'][0]*604800)  ## choosing some arbitrary start/stop time for testing code
    gps_stop = stop_time - (novatel['Week'][0]*604800)

    flightline = novatel[(novatel['GPSTime [sec]'] >= gps_start) & (novatel['GPSTime [sec]'] <= gps_stop)]  ## truncates data to only take the desired flightline
    flightline.reset_index(inplace=True)


    return flightline

##########  Functions for writing an SLC Product... then reading it back in  ##############################################################

def write_annotation_file(directory, channel, timestamp, start_file, stop_file, vp, H, f0, fs, prf, az_BW, chirp_samps, range_samples, az_samples):

    text = f'''Processing Parameters for {channel} at {timestamp} using Range Doppler processor...

The data is single look complex in slant range...
Each real / imaginary component is 16 bits. 
They are interleved: real, imaginary, real, imaginary... (similar to UAVSAR)

Data loaded from: {directory} 
Start File:       {start_file} 
Stop File:        {stop_file} 
            
Some parameters specific for the matched filters...
Center Frequency: {f0} [Hz]
fs:               {fs} [Hz]
PRF:              {prf} [Hz]
H:                {H} [m]
vp:               {vp} [m/s]
chirp samples:    {chirp_samps}
Azimuth BW:       {az_BW} [Hz]

Some parameters useful for loading data later...
Range Samples:    {range_samples}
Azimuth Samples:  {az_samples} 
'''
    
    outputname = f'{channel}_{timestamp}_Annotation_File.txt'
    outputpath = os.path.join(directory, outputname)

    print(f'Saving annotation file at {outputpath}')

    with open(outputpath, 'w') as f:
        f.write(text)

    print(f'Saved annotation file at {outputpath}')

            
def write_slc(focused, directory, channel, timestamp, start_file, stop_file, vp, H, f0, fs, prf, az_BW, chirp_samps):

    #############################
    #  Writes a .dat file with a focused SLC
    #############################

    az_samples, range_samples = focused.shape

    write_annotation_file(directory, channel, timestamp, start_file, stop_file, vp, H, f0, fs, prf, az_BW, chirp_samps, range_samples, az_samples)


    focused = focused / int(chirp_samps)

    real = np.real(focused.flatten()).astype(np.float16)
    imaginary = np.imag(focused.flatten()).astype(np.float16)

    data_length = 2*az_samples*range_samples   ##  az x range for the matrix, then x2 to save each real/imaginary number separately
    not_complex = np.zeros(data_length, dtype=np.float16)
    not_complex[0::2] = real
    not_complex[1::2] = imaginary

    del(real)
    del(imaginary)

    outputname = f'{channel}_{timestamp}_slc.dat'
    outputpath = os.path.join(directory, outputname)

    print(f'Saving SLC at {outputpath}')

    with open(outputpath, 'wb') as f:
        f.write(not_complex)

    print(f'Saved SLC at {outputpath}')


def read_SNOWWI_SLC(directory, filename, az_samples, range_samples):

    complex_ = np.zeros((az_samples, range_samples))

    item = os.path.join(directory, filename)

    print(f'Looking for SLC at {item} to open...')

    with open(item, 'rb') as f:
        data = np.fromfile(f, dtype=np.float16, count=-1)

    real = np.reshape(data[0::2].astype(np.float16), (az_samples, range_samples))
    im = np.reshape(np.multiply(data[1::2].astype(np.float16), 1j), (az_samples, range_samples))
    
    complex_ = real + im

    print(f'Succesfully loaded SLC from {item} ...')

    return complex_

#########  Functions for Doing Math  #########################################################################################################

def average_rows(matrix, N):
    # Calculate the number of resulting rows after averaging
    num_rows, num_cols = matrix.shape
    num_result_rows = num_rows // N

    # Reshape the matrix to group rows for averaging
    reshaped_matrix = matrix[:num_result_rows * N, :].reshape(num_result_rows, N, num_cols)

    # Calculate the row-wise averages
    averages = np.mean(reshaped_matrix, axis=1)

    return averages

def lowpass(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    
    return y

def bandpass(data, low_cutoff, high_cutoff, fs, order):
    nyq = 0.5*fs
    normal_low = low_cutoff/nyq
    normal_high = high_cutoff/nyq

    b, a = signal.butter(order, [normal_low, normal_high], btype='band', analog=False)
    y = signal.filtfilt(b, a, data)

    return y

def frequency_convert(data, fLO, fs, axis=1):
    #####
    # Requires complex data
    # fLO is freq of LO to mix with
    # fs is sample frequency of data
    # direction is direction of conversion - 'up' or 'down'
    #####

    if axis == 0:
        data = data.T

    samps = data.shape[1]
    tmin = 0
    tmax = samps/fs
    n = fLO.shape[0]

    fLO = fLO.reshape((n, -1))

    t = np.linspace(tmin, tmax, samps).reshape((-1, samps))  ## Make time array
    ft = fLO.dot(t)

    lo = np.exp(-1j*2*np.pi*ft)
    data = data*lo

    if axis == 0:
        data = data.T

    return data

def multilook(image, xLooks, yLooks):

    ###  Adjusts image to be able to be multilooked (no extra rows/columns)

    if image.shape[0]%yLooks != 0:
        image = image[:-(image.shape[0]%yLooks), :]


    if image.shape[1]%xLooks !=0:
        image = image[:, :-(image.shape[1]%xLooks)]
    
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

def truncate_swath(data, swath_width, H, tp, fs, look_angle=45, bw_el=45):
    
    c = 3e8
    samp0 = int((tp*fs)/2)  ## after compression, this becomes new "sample zero"

    # R_start = np.sqrt(H**2*(1+np.tan(look_angle - bw_el/2)**2))
    # R_stop = np.sqrt(H**2 + (H*np.tan(look_angle - bw_el/2) + swath_width)**2)
    R_start = H/np.cos(np.radians(look_angle - bw_el/2))
    R_stop = np.sqrt(H**2 + (H*np.tan(np.radians(look_angle - bw_el/2)) + swath_width)**2)

    first_return = int(((2*R_start/c)*fs)/2) ## after compression this becomes first sample with a return
    final_return = int(((2*R_stop/c)*fs)/2)

    start_swath = samp0 + first_return
    stop_swath = samp0 + final_return

    print(start_swath)
    print(stop_swath)



    if stop_swath >= data.shape[1]:
        stop_swath = data.shape[1]

    data = data[:, start_swath:stop_swath]
    return data, R_start, R_stop


def rolling_avg(array, window_size, axis=None):

    ######################################
    ## axis = 1 rolls average across a row
    ## axis = 0 rolls average down a column
    ######################################

    if array.ndim == 1:
        return bn.move_mean(array, window=window_size)
    elif array.ndim == 2:
        # Pad the array if axis is specified
        if axis is not None:
            pad_width = [(0, 0)] * array.ndim
            pad_width[axis] = (window_size - 1, 0)
            padded_array = np.pad(array, pad_width, mode='edge')

            avgd = bn.move_mean(padded_array, window=window_size, axis=axis)

            if axis == 1:
                avgd = avgd[:, (window_size-1):]

            else:
                avgd = avgd[(window_size-1):, :]
            
            return avgd
        else:
            raise ValueError("Axis must be specified for 2D array")
    else:
        raise ValueError("Array must be 1D or 2D")
    

def smooth_doppler(data, fs, B):
    
    range_avgs = int(fs/(2*B))  ## this is found by comparing radar range resolution to ettus sampling resolution
    az_avgs = 100  ## what is a smart way to do this
    range_avgs = 100

    smoothed_doppler = rolling_avg(abs(data), range_avgs, axis=1)
    smoothed_doppler = rolling_avg(smoothed_doppler, az_avgs, axis=0)

    # extra_rows = abs(data.shape[0] - smoothed_doppler.shape[0])
    # extra_columns = abs(data.shape[1] - smoothed_doppler.shape[1])


    # truncate_data = data[int(np.ceil(extra_rows/2)):-int(np.floor(extra_rows/2)), :smoothed_doppler.shape[1]]

    return smoothed_doppler#, truncate_data

def fit_doppler(data, prf, fs, B, order, snr=True):   ### this needs to be tested


    smoothed_doppler = smooth_doppler(data, fs, B)  ##  Running some rolling averages over azimuth doppler centroids
    az_freq = np.linspace(-prf/2, prf/2, data.shape[0])

    doppler_centroid_idx = np.argmax(smoothed_doppler, axis=0)

    if snr == True:
        means = np.mean(abs(smoothed_doppler), axis=0)
        maxs = np.max(abs(smoothed_doppler), axis=0)

        badsnr = np.where(maxs<(1.3*means))[0]  ## Finding indeces where snr < ...dB

        doppler_centroid_idx[badsnr[0]:] = doppler_centroid_idx[badsnr[0] - 1]  ## replacing indeces with bad snr to the last index with a good snr
    
    doppler_centroids = az_freq[doppler_centroid_idx]

    range_samples = np.linspace(0, doppler_centroids.shape[0]-1, doppler_centroids.shape[0])
    coeff = np.polyfit(range_samples, doppler_centroids, order)  ## finding coefficients for polynomial
    doppler_fit = np.zeros_like(range_samples)  ##  making space

    for i in range(order):
        doppler_fit = doppler_fit + coeff[i]*range_samples**(order-i)  ##   creating an array for the polynomial
        
    doppler_fit += coeff[order]

    return doppler_fit, doppler_centroids


def write_doppler_fit(doppler_fit, dopplerpath):

    #############################
    #  Writes a .dat file with fit Doppler curve which can be applied to other channels
    #############################

    doppler_fit = doppler_fit.astype(np.float16)

    # outputname = f'{timestamp}_doppler_fit.dat'
    # outputpath = os.path.join(directory, outputname)

    print(f'Saving Doppler Curve at {dopplerpath}')

    with open(dopplerpath, 'wb') as f:
        f.write(doppler_fit)

    print(f'Saved Doppler Curve at {dopplerpath}')


def load_doppler_fit(dopplerpath):

    print(f'Looking for Doppler Curve at {dopplerpath} to open...')

    with open(dopplerpath, 'rb') as f:
        doppler_fit = np.fromfile(f, dtype=np.float16, count=-1)

    print(f'Succesfully loaded Doppler Curve from {dopplerpath} ...')

    return doppler_fit


def correct_height(data, novatel, fs):

    Ts = 1/fs
    N = data.shape[0]
    gps_times = novatel['GPSTime [sec]'] - (novatel['GPSTime [sec]'][0])

    start_time = 0
    stop_time = novatel['GPSTime [sec]'][novatel.index[-1]] - novatel['GPSTime [sec]'][0]
    times = np.linspace(start_time, stop_time, N)

    Hmean = novatel['H-Ell'].mean()
    H_interpolate = np.interp(times, gps_times, novatel['H-Ell'])

    dH = H_interpolate - Hmean
    dH_samples = (2*dH/(3e8 * Ts)).astype(int)

    for row, shift in enumerate(dH_samples):
        data[row, :] = np.roll(data[row, :], shift)


    print(dH_samples)

    max_roll = max(abs(dH_samples))
    data = data[:, max_roll:-max_roll]

    return data

    ####  dH_samples should be the number of samples to correct the range by... need to np.roll for each pulse ###

##########  Functions for doing Compression  #################################################################################################

def compress(data, tp, fs, fl, fh, direction, window=True, plot=True):
    
    ref_samp = int(tp*fs)  ##  finding number of samples that belong in the reference chirp

    t_ref = np.linspace(0, tp, ref_samp)
    f_ref = np.linspace(-fs/2, fs/2, ref_samp)
    freq_mhz = f_ref/1e6
    print(f'Chirp Samples: {ref_samp}')

    if direction == 'up':  ##  Generating sin and cosine up chirps
        sinChirp = chirp(t_ref, fl, t_ref[-1], fh, method='linear').astype(np.float32)
        cosChirp = chirp(t_ref, fl, t_ref[-1], fh, method='linear', phi=90).astype(np.float32)

    elif direction == 'down':  ##  Generating sin and cosine down chirps
        sinChirp = chirp(t_ref, fh, t_ref[-1], fl, method='linear').astype(np.float32)
        cosChirp = chirp(t_ref, fh, t_ref[-1], fl, method='linear', phi=90).astype(np.float32)

    if window == True:  ##  Adding a window if desired
        hamming = np.hamming(ref_samp).astype(np.float32)
        sinChirp = hamming*sinChirp
        cosChirp = hamming*cosChirp
        
    compressed = np.zeros_like(data, dtype=np.complex64)
    az_samp = data.shape[0]

    for i in range(az_samp):
        sin_corr = signal.correlate(data[i], 1j*sinChirp, mode='same', method='fft')
        cos_corr = signal.correlate(data[i], cosChirp, mode='same', method='fft')
        compressed[i] = cos_corr + sin_corr

    if plot == True:

        sin_fft = np.fft.fftshift(np.fft.fft(sinChirp)) 
        cos_fft = np.fft.fftshift(np.fft.fft(cosChirp))

        fig, ax = plt.subplots(figsize=[14, 6], nrows=1, ncols=2)
        ax[0].plot(t_ref*1e6, sinChirp, color='b')
        ax[0].plot(t_ref*1e6, cosChirp, color='r')

        # ax[1].plot(freq_mhz, 20*np.log10(fft_data[1]), color='k', alpha=0.7, label='Filtered Data')
        ax[1].plot(freq_mhz, 20*np.log10(abs(sin_fft)), color='b', linestyle='--', label='sin() Ref')
        ax[1].plot(freq_mhz, 20*np.log10(abs(cos_fft)), color='r', linestyle=':', label='cos() Ref')

        ax[0].set_title('Ref Signal')
        ax[0].set_xlabel('Time [microsec]')
        ax[0].set_xlim(0, t_ref[-1]*1e6)
        ax[0].grid()

        ax[1].set_title('Ref Signal - FFTs')
        ax[1].set_xlabel('Frequency [MHz]')
        ax[1].set_ylabel('[dB]')
        ax[1].set_xlim(0, max(freq_mhz))
        ax[1].grid()
        ax[1].legend()

        plt.show()

    return compressed

def rcmc(data, lambda_, fs_az, fs_rng, Rmin, Rmax, vp):

    az_samp = data.shape[0]
    rng_samp = data.shape[1]

    y = np.arange(az_samp)
    # x = np.arange(rng_samp)

    fn = np.linspace(-fs_az/2, fs_az/2, az_samp)

    # R0 = (Rmin + Rmax) / 2

    R = np.linspace(Rmin, Rmax, rng_samp, dtype=np.float32).reshape((-1, rng_samp))
    fn = (np.linspace(-fs_az/2, fs_az/2, az_samp, dtype=np.float32)**2).reshape((az_samp, -1))  ## squared because of range migration equation
    Rfn = fn.dot(R)
    dR = (lambda_**2 * Rfn) / (8*vp**2)

    Rshift = (2*dR*fs_rng) / 3e8
    # Rshift = x - np.array(shift_samples)[:, np.newaxis]

    new_matrix = np.zeros_like(data)

    # print(Rshift)
    # return Rshift
    xin = np.arange(rng_samp)
    
    # print(xin)
    for i in range(az_samp):
        x = xin - Rshift[i, :]
        interp_func = interp1d(x, data[i], kind='linear', bounds_error=False, fill_value=1)
        new_matrix[i] = interp_func(np.arange(rng_samp))
    
    return new_matrix


# def azimuth_compress(data, lambda_, fs, BW, doppler_centroids, Rmin, Rmax, vp):  ##  Returns Azimuth matched filter... must be multiplied in frequency domain

#     r_samps = data.shape[1]
#     freq_samps = data.shape[0]

#     ###  Creating matched filter magnitude window  ###
#     Hz_samp = freq_samps/fs
#     window_samps = int(BW*Hz_samp)
#     before = int(np.ceil((fs - BW)/2 * Hz_samp))
#     after = int(np.floor((fs - BW)/2 * Hz_samp))

#     # before = int((fs/2 + BW/2) * Hz_samp)
#     # after = int((fs/2 - BW/2) * Hz_samp)


#     window = np.hamming(window_samps)
#     window = np.pad(window, (before, after), mode='minimum')
#     window = np.tile(window, (r_samps, 1)).T
#     centroid_idx = (doppler_centroids*Hz_samp).astype(int)

#     for col, shift in enumerate(centroid_idx):
#         window[:, col] = np.roll(window[:, col], shift)

#     ###  Creating matched filter phase matrix  ###
#     freq_vector = np.linspace(-fs/2, fs/2, freq_samps).reshape((freq_samps, -1))  ## 1 dimensional vector
#     freq_matrix = np.tile(freq_vector, (1, r_samps))  ##  2D matrix
#     del(freq_vector)

#     fdop = np.tile(doppler_centroids, (freq_samps, 1))  #*(BW/fs)
#     freq_matrix = (freq_matrix - fdop)**2
#     del(fdop)

#     R = np.linspace(Rmin, Rmax, r_samps)  ## 1D vector
#     R = np.tile(R, (freq_samps, 1))  ## 2D Matrix
#     Ka = 2*vp**2 / (lambda_*R)

#     Haz = np.exp(1j*np.pi*freq_matrix / Ka) * window

#     return Haz

def azimuth_compress(data, lambda_, fs, BW, doppler_centroids, Rmin, Rmax, vp):  ##  Returns Azimuth matched filter... must be multiplied in frequency domain

    r_samps = data.shape[1]
    freq_samps = data.shape[0]

    ###  Creating matched filter magnitude window  ###
    Hz_samp = freq_samps/fs
    window_samps = int(BW*Hz_samp)
    before = int(np.ceil((fs - BW)/2 * Hz_samp))
    after = int(np.floor((fs - BW)/2 * Hz_samp))

    # before = int((fs/2 + BW/2) * Hz_samp)
    # after = int((fs/2 - BW/2) * Hz_samp)


    window = np.hamming(window_samps)
    window = np.pad(window, (before, after), mode='minimum')
    window = np.tile(window, (r_samps, 1)).T
    centroid_idx = (doppler_centroids*Hz_samp).astype(int)

    for col, shift in enumerate(centroid_idx):
        window[:, col] = np.roll(window[:, col], shift)

    ###  Creating matched filter phase matrix  ###
    freq_vector = np.linspace(-fs/2, fs/2, freq_samps).reshape((freq_samps, -1))  ## 1 dimensional vector
    freq_matrix = np.tile(freq_vector, (1, r_samps))  ##  2D matrix
    del(freq_vector)

    # doppler_centroids = np.zeros_like(doppler_centroids)
    fdop = np.tile(doppler_centroids, (freq_samps, 1))  #*(BW/fs)
    freq_matrix = (freq_matrix - fdop)**2
    del(fdop)
    print('test')

    R = np.linspace(Rmin, Rmax, r_samps)  ## 1D vector
    R = np.tile(R, (freq_samps, 1))  ## 2D Matrix
    Ka = 2*vp**2 / (lambda_*R)

    Haz = np.exp(1j*np.pi*freq_matrix / Ka) * window

    return Haz