# I/O library for SNOWWI data
# Author: Marc Closa Tarres (MCT)
# Date: 2024-04-12
# Version: 0.1
#
# Changelog:
# 0.1 - Initial version
#

import numpy as np
import os
import re


class SnowwiDataset:
    """
    SnowwiDataset class to store data and headers from a SNOWWI file.

    Arguments:
    ----------
    data: np.ndarray
        Raw pulses reshaped into specified shape (number of samples per pulse)

    headers: np.ndarray
        Value of binary headers converted into uint16

    timestamps: np.ndarray
        Data UNIX timestamps converted into float.

    Attributes:
    -----------
    data: np.ndarray
        Raw pulses reshaped into specified shape (number of samples per pulse)

    headers: np.ndarray
        Value of binary headers converted into uint16

    timestamps: np.ndarray
        Data UNIX timestamps converted into float.

    Methods:
    --------
    __init__(data, headers=None, timestamps=None):
        Initialize the SnowwiDataset object.

    __str__():
        Return a string representation of the SnowwiDataset object.
    """

    def __init__(self, data, headers=None, timestamps=None):
        self.data = data
        self.headers = headers
        self.timestamps = timestamps

    def __str__(self):
        return f"SnowwiDataset with data shape: {self.data.shape} and headers shape: {self.headers.shape}"


class Reader:
    """
    Reader class to read data from file/s and/or reshape it to a dataset.

    Arguments:
    ----------
    filename: str or list of str
        Name of the file/s to read. If multiple files are provided, they are going to be read in order and concatenated.

    Attributes:
    -----------
    filename: str or list of str
        Name of the file/s to read.

    data: np.ndarray
        Data read from the file/s. It is going to be formatted after calling the reshape method.

    headers: np.ndarray
        Headers read from the file/s. It is going to be formatted after calling the reshape method.

    unshaped: list of np.ndarray
        Data read from the file/s. It is not going to be formatted until the reshape method is called.

    timestamps: np.ndarray
        Timestamps extracted from the headers. It is going to be calculated after calling the header_to_timestamp method.

    Methods:
    --------
    read():
        Read the data from the file/s.

    reshape(n_samp, header_samp=0, skip_samp=0, truncate=None):
        Reshape the data read from the file/s. It is going to be formatted and stored in the data attribute.

    header_to_timestamp():
        Extract the timestamps from the headers and store them in the timestamps attribute.

    to_dataset():
        Return the data formatted as a SnowwiDataset object.
    """

    def __init__(self, filename):
        init_msg = f"""
        Initializing Reader class.

        Reading data from file/s: {filename}

        WARNING: The data is not going to be formatted until the reshape method is called. Therefore,
        the data is going to be stored in the unshaped attribute.

        Once the resape() method is called, the data is going to be formatted and stored in the data attribute,
        and unshaped data is going to be deleted.
        """
        print(init_msg)

        self.datafile = filename

        self.data = None
        self.headers = None
        self.unshaped = None
        self.timestamps = None

        self.read()

    def read(self):
        """
        Read the data from the file/s.
        """
        assert self.datafile is not None, "No filename provided."

        # Read files if one or multiple
        self.unshaped = [np.fromfile(f, dtype=np.int16) for f in self.datafile]

    def reshape(self, n_samp, header_samp=0, skip_samp=0, truncate=None):
        """
        Reshape the data read from the file/s. It is going to be formatted and stored in the data attribute.

        Arguments:
        ----------
        n_samp: int
            Number of samples per frame.

        header_samp: int
            Number of samples per header.

        skip_samp: int
            Number of samples to skip at the beginning of the frame.

        truncate: int or None
            Number of samples to keep at the end of the frame. If None, all samples are kept.
        """
        print(f'Reshaping data with shape:')
        print(f'    Total samples per frame: {n_samp}')
        print(f'    Header samples: {header_samp}')
        print(f'    Skipped samples: {skip_samp}')
        print(f'    Truncated to: {truncate}')

        n_files = len(self.unshaped)
        file_shapes = [d.shape[0] for d in self.unshaped]
        n_rows = file_shapes[0] // n_samp
        read_until = int(n_rows * n_samp)

        # Allocate memory for data and headers
        total_rows = n_rows * n_files
        self.data = np.zeros((total_rows, n_samp), dtype=np.int16)
        if header_samp > 0:
            self.headers = np.zeros((total_rows, header_samp), dtype=np.uint16)

        # Reshape and save to self.data
        for i, d in enumerate(self.unshaped):
            self.data[i*n_rows:(i+1)*n_rows,
                      :] = d[:read_until].reshape(-1, n_samp)

            # Save headers
            if header_samp > 0:
                self.headers[i*n_rows:(i+1)*n_rows, :] = self.data[i *
                                                                   n_rows:(i+1)*n_rows, :header_samp].astype(np.uint16)

        # Delete the skipped samples - will automatically delete the headers from the data
        self.data = np.delete(self.data, np.s_[:skip_samp], axis=1)

        # Delete after truncated
        if truncate is not None:
            self.data = self.data[:, :truncate]

    def header_to_timestamp(self):
        """
        Extract the timestamps from the headers and store them in the timestamps attribute.
        """

        assert self.headers is not None, "No headers found in data."
        left = 4*7
        right = left + 4
        radio_time = self.headers[:, left:right]
        vec = np.array([2**16, 2**0, 2**48, 2**32])
        sum = np.dot(radio_time.astype(np.uint64), vec.astype(np.uint64))
        self.timestamps = sum/122.88e6/4

    def to_dataset(self):
        """
        Return the data formatted as a SnowwiDataset object.
        """
        return SnowwiDataset(self.data, self.headers, self.timestamps)

    def read_to_dataset(self, n_samp, header_samp=0, skip_samp=0, truncate=None):
        """
        Read the data from the file/s and reshape it to a dataset.

        Arguments:
        ----------
        n_samp: int
            Number of samples per frame.

        header_samp: int
            Number of samples per header.

        skip_samp: int
            Number of samples to skip at the beginning of the frame.

        truncate: int or None
            Number of samples to keep at the end of the frame. If None, all samples are kept.

        Returns:
        --------
        SnowwiDataset
            Formatted dataset with data and headers.
        """
        self.reshape(n_samp, header_samp, skip_samp, truncate)
        self.header_to_timestamp()
        return self.to_dataset()

    @staticmethod
    def read_ann_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = {}
        category = None

        for line in lines:
            # Check for category headers
            if "GENERAL PARAMETERS" in line:
                category = "GENERAL PARAMETERS"
                data[category] = {}
            elif "ACQUISITION PARAMETERS" in line:
                category = "ACQUISITION PARAMETERS"
                data[category] = {}
            elif "OUTPUT PARAMETERS" in line:
                category = "OUTPUT PARAMETERS"
                data[category] = {}

            # Parse lines with data
            match = re.match(r'(.+?)\s*=\s*(.+)', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Remove units in brackets from the key
                key = re.sub(r'\s*\(.*?\)', '', key).strip()
                
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string if not a float

                if category:
                    data[category][key] = value
                else:
                    data[key] = value

        return data


class Writer:
    """
    Writer class to write SLCs following the SNOWWI format.

    Samples are encoded in float16 in IQ format. This is:

    """

    def __init__(self, out_path, filename, anno_file="snowwi_ann.txt", factor=1):
            
            self.output_path = out_path
            self.filename = os.path.join(out_path, filename)
            self.ann_file = os.path.join(out_path, anno_file)
            self.factor = factor

    def write_slc(self, data):
        """
        Write the data to a file.

        Arguments:
        ----------
        data: np.ndarray
            Data to write to the file.
        """

        az_samps, rng_samps = data.shape
        print(data.shape)
        print(len(data))

        data *= self.factor
        print(f"Multiplying by {self.factor}")

        total_samps = az_samps * rng_samps * 2  # Interleaved real and complex
        print(total_samps)
        to_file = np.zeros(total_samps, dtype=np.float16)
        print(to_file.shape)
        to_file[::2] = data.real.flatten().astype(np.float16)
        to_file[1::2] = data.imag.flatten().astype(np.float16)
        
        print(to_file[:2])

        print(f"Writing data to file: {self.filename}")

        with open(self.filename, 'wb') as f:
            f.write(to_file)
        print(f"Done writing output file: {self.filename}.")


    def write_annotation(self, args, cfg, timestamps, mlooks_params={}):
        ann = f"""SNOWWI SLC ANNOTATION FILE - Flightline: {args.flightline}

        GENERAL PARAMETERS:
        Flightline ID                            =    {args.flightline}
        First raw data timestamp (unix)          =    {timestamps[0]}
        Last raw data timestamp (unix)           =    {timestamps[-1]}


        ACQUISITION PARAMETERS:
        Sampling frequency (Hz)                  =    {cfg.getfloat("Sampling parameters", "Sampling frequency"):.4e}
        Pulse Repetition Frequency (Hz)          =    {cfg.getfloat("Sampling parameters", "PRF")}

        
        OUTPUT PARAMETERS:
        Original azimuth samples                 =    {mlooks_params["Initial shape"][0]}
        Original range samples                   =    {mlooks_params["Initial shape"][1]}
        Original azimuth resolution (m)          =    {mlooks_params["Initial az resolution"]}
        Original range resolution (m)            =    {mlooks_params["Initial rg resolution"]}

        SLC azimuth samples                      =    {mlooks_params["Output shape"][0]}
        SLC range samples                        =    {mlooks_params["Output shape"][1]}
        SLC azimuth resolution (m)               =    {mlooks_params["Output az resolution"]}
        SLC range resolution (m)                 =    {mlooks_params["Output rg resolution"]}
        
        Scaling factor                           =    {self.factor}"""
        
        out_ann = os.path.join(self.output_path, self.ann_file)
        print(f"Writing annotation ")
        with open(out_ann, 'w+') as f:
            f.write(ann)


    def append_slc(self, data):
        """
        Append data to an existing file.

        Arguments:
        ----------
        data: np.ndarray
            Data to append to the file.
        """

        az_samps, rng_samps = data.shape

        data = data / self.factor

        total_samps = az_samps * rng_samps * 2  # Interleaved real and complex
        to_file = np.zeros(total_samps, dtype=np.float16)
        to_file[::2] = data.real.flatten().astype(np.float16)
        to_file[1::2] = data.imag.flatten().astype(np.float16)

        print(f"Writing data to file: {self.datafile}")

        with open(self.datafile, 'ab') as f:
            f.write(to_file)

    @staticmethod
    def write_doppler(doppler, outpath, filename):
        outfile = os.path.join(outpath, filename)

        dopp_32bit = doppler.astype(np.float32)
        print(f"Writting doppler to: {outfile}...")
        with open(outfile, 'wb') as f:
            f.write(dopp_32bit)
        print("Done!")


"""
def read_and_reshape(fileName, N, header_samples=0, skip_samples=0, truncate=None):
    print(f"Reading file:")
    print(f"    {fileName}")
    data = np.fromfile(fileName, dtype=np.int16)
    data = data.reshape(-1, N)[:, :truncate]
    headers = data[:, :header_samples].astype(np.uint16)
    data = np.delete(data, np.s_[:skip_samples], axis=1)
    return data, headers
"""
