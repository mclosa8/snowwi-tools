#!/usr/bin/env python3

import numpy as np

import argparse

from h5py import File


def parse_args():
    
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('filename', help="File to correct.")
    arg_parser.add_argument('--limit', default=500, type=int)
    arg_parser.add_argument('--alpha', default=0.15, type=float)
    arg_parser.add_argument('--out', default='corrected.h5')
    arg_parser.add_argument("--mag-phase", action='store_true')

    args = arg_parser.parse_args()
    args.out = f"{args.filename.split('.')[0]}_{args.out}"
    
    return args


def main():
    
    args = parse_args()
    print(args)

    h5in = File(args.filename, 'r')

    # Todo Make this dynamically.
    weights = (h5in['pulse_count'][:] - args.limit)*args.alpha + args.limit

    corrected = h5in['value']/weights

    h5out = File(args.out, 'w')

    if args.mag_phase:
        h5out.create_dataset('magnitude', data=abs(corrected))
        h5out.create_dataset('phase', data=np.angle(corrected))
    else:
        h5out.create_dataset('value', data=corrected)
    
    h5out.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
