#!/usr/bin/env python3

import numpy as np

import argparse
import h5py as h5


def parse_args():
    
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('filename', help="Filename to retrieve values.")
    arg_parser.add_argument('-o', '--output', help="Filename for output.",
                            default="out.h5")

    return arg_parser.parse_args()


def main():
    
    args = parse_args()

    h5in = h5.File(args.filename, 'r')
    mag = abs(h5in['value'][:])
    phase = np.angle(h5in['value'][:])

    with h5.File(args.output, 'w') as h5out:
        h5out.create_dataset('magnitude', data=mag)
        h5out.create_dataset('angle', data=phase)
    
    print(f"{args.filename} converted into mag and phase and saved in {args.output}.")





if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass