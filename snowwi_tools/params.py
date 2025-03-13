"""
    Parameter-related functions and dictionaries.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v1

    Changelog:
        - v0: Initial version - Nov 13, 2024 - MCT
        - v1: Re-written functions and added DAQ params - Feb 28, 2025 - MCT
"""


_CHANNEL_DICT = {
    'low': {
        0: 'chan0',
        1: 'chan2'
    },
    'high': {
        0: 'chan1',
        1: 'chan3'
    }
}

_DAQ_PARAMS_4x2 = {
    'prf': 1e3,
    'data_samps': 100_000,
    'header_samps': 4,
    'fs': 1.2288e9
}

_DAQ_PARAMS_ETTUS = {
    'prf': 1e3,
    'data_samps': 63_952,  # 64000 total samples - 48 header samples
    'header_samps': 48,
    'fs': 983.064e6
}


_BAND_PARAMS_ETTUS = {
    'low': {
        'f0': 13.64e9,
        'f_l': 143.04e6,
        'f_h': 223.04e6,
        'chirp_type': 'up',
        'lowcut': 140e6,
        'highcut': 225e6,
        'channels': {0: '_CHANNEL_DICT["low"][0]', 1: '_CHANNEL_DICT["low"][1]'}
    },
    'high': {
        'f0': 17.24e9,
        'f_l': 23.04e6,
        'f_h': 103.04e6,
        'chirp_type': 'down',
        'lowcut': 20e6,
        'highcut': 105e6,
        'channels': {0: '_CHANNEL_DICT["high"][0]', 1: '_CHANNEL_DICT["high"][1]'}
    },
    'daq': {**_DAQ_PARAMS_ETTUS}
}


_BAND_PARAMS_4x2 = {
    'low': {
        'f0': 13.64e9,
        'f_l': 100e6,
        'f_h': 180e6,
        'chirp_type': 'up',
        'lowcut': 95e6,
        'highcut': 185e6
    },
    'high': {
        'f0': 17.24e9,
        'f_l': 220e6,
        'f_h': 300e6,
        'chirp_type': 'down',
        'lowcut': 215e6,
        'highcut': 305e6
    },
    'c': {
        'f0': 5.39e9,
        'f_l': 340e6,
        'f_h': 420e6,
        'chirp_type': 'up',  # I actually don't know for sure. Up makes sense
        'lowcut': 315e6,
        'highcut': 425e6
    },
    'daq': {**_DAQ_PARAMS_4x2}
}


def get_band_params_ettus(band, channel=None):

    if not band in _BAND_PARAMS_ETTUS:
        raise Exception(
            f"No valid band selected. Use {list(_BAND_PARAMS_ETTUS.keys())}")

    if band == 'daq':
        return _BAND_PARAMS_ETTUS[band]

    if channel not in _BAND_PARAMS_ETTUS[band]['channels']:
        raise Exception("No valid channel selected.")

    params = _BAND_PARAMS_ETTUS[band].copy()
    params['channel'] = eval(params['channels'][channel])
    del params['channels']

    return params


def get_band_params_4x2(band: str) -> dict:

    if not band in _BAND_PARAMS_4x2:
        raise KeyError(
            f"No valid band selected. Use {list(_BAND_PARAMS_4x2.keys())}.")

    return _BAND_PARAMS_4x2[band]
