"""
    DSP-relatad functions.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 13, 2024 - MCT
"""

import numpy as np

from scipy.signal import butter, filtfilt, sosfilt

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

def exp_chirp(t, f0, t1, f1, phi=0):
    K = (f1 - f0) / t1
    phase = 2 * np.pi * (f0 * t + 0.5 * K * t * t)
    return np.exp(1j * (phase + phi * np.pi / 180))


def voltage_to_log_power(voltage):
    return 10 * np.log10(np.abs(voltage)**2)