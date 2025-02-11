"""
    Parameter-related functions and dictionaries.

    Author: Marc Closa Tarres (MCT)
    Date: 2024-11-13
    Version: v0

    Changelog:
        - v0: Nov 13, 2024 - MCT
"""


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

    return params


def get_band_params_4x2(band, channel):
    band_params = {
        'low': {
            'f0': 13.64e9,
            'f_l': 100e6,
            'f_h': 180e6,
            'chirp_type': 'up',
            'lowcut': 95e6,
            'highcut': 185e6,
        },
        'high': {
            'f0': 17.24e9,
            'f_l': 220e6,
            'f_h': 300e6,
            'chirp_type': 'down',
            'lowcut': 215e6,
            'highcut': 305e6,
        },
        'c': {
            'f0': 17.24e9,
            'f_l': 340e6,
            'f_h': 420e6,
            'chirp_type': 'up', # I actually don't know for sure. Up makes sense
            'lowcut': 315e6,
            'highcut': 425e6,
        }
    }

    if band not in band_params:
        raise Exception("No valid band selected")

    params = band_params[band].copy()
    # params['channel'] = eval(params['channels'][channel])
    # del params['channels']

    return params
