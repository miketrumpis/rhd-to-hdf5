#!/usr/bin/env python

import os
import sys
import json
import numpy as np
import h5py
import logging
from rhdlib.rhd_reader import RHDFile, signed_arrays


def convert(filenames, outfile):
    """
    Concatenate the contents of the RHD files into contiguous HDF5 datasets

    Parameters
    ----------
    filenames: sequence
        Sequence of RHD paths.
    outfile: Path, str
        Output path for the HDF5 file

    """
    info = logging.getLogger().info
    info('Creating RHD file maps')
    rhd_files = [RHDFile(f) for f in filenames]
    with h5py.File(outfile, 'w') as data:
        first_file = rhd_files[0]
        arrays = first_file.struct_map
        hdf_arrays = dict()
        hdf_offsets = dict()
        for name, shape in arrays:
            if name in signed_arrays:
                dtype = 'h'
            else:
                dtype = first_file.rhd_map[name].dtype
            num_channels = shape[0]
            all_samples = np.sum([rhd.array_sizes[name][1] for rhd in rhd_files])
            shape = (num_channels, all_samples) if num_channels > 1 else (all_samples,)
            info('dataset name {}, dtype {}, shape {}'.format(name, dtype, shape))
            arr = data.create_dataset(name, dtype=dtype, shape=shape, chunks=True)
            hdf_arrays[name] = arr
            hdf_offsets[name] = 0

        for n, rhd in enumerate(rhd_files):
            print('Converting {} ({} of {})'.format(rhd.file_path, n + 1, len(rhd_files)))
            rhd.to_arrays(arrays=hdf_arrays, offsets=hdf_offsets, apply_scales=False)
            info('Offsets: {}'.format(list(hdf_offsets.items())))
        data.attrs['JSON_header'] = json.dumps(first_file.header)
        info('End of conversion')
    print('Done: {}'.format(outfile))


if __name__ == '__main__':
    import argparse
    from glob import glob


    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass


    app_description = \
        """Combine and convert packetized Intan RHD files into continuous arrays in HDF5 format.
        
        Arrays in the output file may include:
        * amplifier_data: converted to signed int16, scale to uV by multiplying 0.195
        * aux_input_data: unsigned uint16, scale by 37.4e-6 for Volts (sampled at 1/4 rate as amplifier data)
        * board_adc_data: converted to signed int16, scale by 312.5e-6 for Volts
        * supply_voltage_data: uint16, scale by 74.8e-6 for Volts (sampled once per data block)
        * temp_sensor_data: uint16, scale by 0.01
        * board_dig_in_data: boolean
        * board_dig_out_data: boolean
        
        To load (Python example):
        >>> import h5py
        >>> f = h5py.File('ecog_256_array.h5', 'r')
        >>> electrodes_uv = f['amplifier_data'][:, 100:200] * 0.195
        >>> electrodes_uv.shape
        (256, 100)
        >>> f['amplifier_data'].shape   # total available data
        (256, 7200000)
        
        The original header information is stored as a JSON string, which can be parsed like this:
        
        >>> import json
        >>> header = json.loads(f.attrs['JSON_header'])
        >>> header['sample_rate']
        20000.0
        """
    ap = argparse.ArgumentParser(description=app_description, formatter_class=CustomFormatter)
    ap.add_argument('--doc', action='store_true', help='Print usage message to README.txt')
    ap.add_argument('hdf_file', type=str, nargs='?', help='Output filename (.h5 extension will be issued)',
                    default='out')
    ap.add_argument('-i', '--input-files', type=str, nargs='+', help='Input RHD files in order')
    ap.add_argument('-d', '--input-dir', type=str, help='Input directory: RHD files will be converted in sorted order')
    ap.add_argument('-v', '--verbose', action='store_true', help='Turn on logging channel')
    args = ap.parse_args()
    if args.doc:
        with open('README.txt', 'w') as fw:
            fw.write(app_description)
        sys.exit(0)
    if args.verbose:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
    if args.input_files is not None and len(args.input_files):
        input_files = args.input_files
    elif args.input_dir is not None and len(args.input_dir):
        input_files = glob(os.path.join(os.path.abspath(args.input_dir), '*.rhd'))
        input_files = sorted(input_files)
    else:
        print('No input files were given!')
        ap.print_usage()
        sys.exit(2)
    print(input_files)
    outfile = os.path.splitext(args.hdf_file)[0] + '.h5'
    print(outfile)
    convert(input_files, outfile)
