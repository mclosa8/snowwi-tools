#!/usr/bin/env python3

import numpy as np

import argparse
import h5py as h5


def parse_args():
    
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('filename', help="Filename to retrieve values.")
    arg_parser.add_argument('-o', '--output', help="Filename for output.",
                            default="out.h5")
    arg_parser.add_argument('-p', '--power', help='Save power Tiff',
                            action='store_true', default=False)
    arg_parser.add_argument('-lp', '--logpower', help='Save power Tiff',
                            action='store_true', default=False)

    return arg_parser.parse_args()


def main():
    
    args = parse_args()

    h5in = h5.File(args.filename, 'r')
    keys = h5in.keys()
    print(keys)
    field_name = 'slc_distributed' if 'slc_distributed' in keys else 'value'
    print(f'Using {field_name} as field name.')
    mag = abs(h5in[field_name][:])
    phase = np.angle(h5in[field_name][:])

    with h5.File(args.output, 'w') as h5out:
        if args.power:
            pwr = mag**2
            if args.logpower:
                h5out.create_dataset('magnitude', data=10*np.log10(pwr))
                print(f"{args.filename} converted into log power and saved in {args.output}.")
            else:    
                h5out.create_dataset('magnitude', data=pwr)
                print(f"{args.filename} converted into log power and saved in {args.output}.")
        else:
            h5out.create_dataset('magnitude', data=mag)
            h5out.create_dataset('angle', data=phase)
            print(f"{args.filename} converted into mag and phase and saved in {args.output}.")
    





if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass