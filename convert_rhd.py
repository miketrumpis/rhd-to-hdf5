#!/usr/bin/env python

import os
import sys
import struct
import json
import numpy as np
import h5py

# ****************************************************************************************************
# Lightly modified Intan-provided code from:
#
# Michael Gibson 23 April 2015
# Modified Adrian Foy Sep 2018
#
# Intan methods used:
# * read_header -- converts header bytes to Python dictionary
# * get_bytes_per_data_block -- finds number bytes per block based on activated chip & board channels
# * read_one_data_block -- reads out bytes from one block and splits across appropriate arrays
# ****************************************************************************************************


def read_qstring(fid):
    """Read Qt style QString.

    The first 32-bit unsigned number indicates the length of the string (in bytes).
    If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """

    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16):
        return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1):
        print(length)
        raise Exception('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for i in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    a = ''.join([chr(c) for c in data])

    return a


def read_header(fid):
    """Reads the Intan File Format header from the given file."""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    magic_number, = struct.unpack('<I', fid.read(4))
    if magic_number != int('c6912702', 16):
        raise Exception('Unrecognized file type.')

    header = {}
    # Read version number.
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4))
    header['version'] = version

    print('')
    print('Reading Intan Technologies RHD2000 Data File, Version {}.{}'.format(version['major'], version['minor']))
    print('')

    freq = {}

    # Read information of sampling rate and amplifier frequency settings.
    header['sample_rate'], = struct.unpack('<f', fid.read(4))
    (freq['dsp_enabled'], freq['actual_dsp_cutoff_frequency'], freq['actual_lower_bandwidth'],
     freq['actual_upper_bandwidth'],
     freq['desired_dsp_cutoff_frequency'], freq['desired_lower_bandwidth'],
     freq['desired_upper_bandwidth']) = struct.unpack('<hffffff', fid.read(26))

    # This tells us if a software 50/60 Hz notch filter was enabled during
    # the data acquisition.
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']

    (freq['desired_impedance_test_frequency'], freq['actual_impedance_test_frequency']) = struct.unpack('<ff',
                                                                                                        fid.read(8))

    note1 = read_qstring(fid)
    note2 = read_qstring(fid)
    note3 = read_qstring(fid)
    header['notes'] = {'note1': note1, 'note2': note2, 'note3': note3}

    # If data file is from GUI v1.1 or later, see if temperature sensor data was saved.
    header['num_temp_sensor_channels'] = 0
    if (version['major'] == 1 and version['minor'] >= 1) or (version['major'] > 1):
        header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))

    # If data file is from GUI v1.3 or later, load eval board mode.
    header['eval_board_mode'] = 0
    if ((version['major'] == 1) and (version['minor'] >= 3)) or (version['major'] > 1):
        header['eval_board_mode'], = struct.unpack('<h', fid.read(2))

    header['num_samples_per_data_block'] = 60
    # If data file is from v2.0 or later (Intan Recording Controller), load name of digital reference channel
    if (version['major'] > 1):
        header['reference_channel'] = read_qstring(fid)
        header['num_samples_per_data_block'] = 128

    # Place frequency-related information in data structure. (Note: much of this structure is set above)
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['aux_input_sample_rate'] = header['sample_rate'] / 4
    freq['supply_voltage_sample_rate'] = header['sample_rate'] / header['num_samples_per_data_block']
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']

    header['frequency_parameters'] = freq

    # Create structure arrays for each type of data channel.
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['aux_input_channels'] = []
    header['supply_voltage_channels'] = []
    header['board_adc_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []

    # Read signal summary from data file header.

    number_of_signal_groups, = struct.unpack('<h', fid.read(2))
    print('n signal groups {}'.format(number_of_signal_groups))

    for signal_group in range(1, number_of_signal_groups + 1):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (signal_group_enabled, signal_group_num_channels, signal_group_num_amp_channels) = struct.unpack('<hhh',
                                                                                                         fid.read(6))

        if (signal_group_num_channels > 0) and (signal_group_enabled > 0):
            for signal_channel in range(0, signal_group_num_channels):
                new_channel = {'port_name': signal_group_name, 'port_prefix': signal_group_prefix,
                               'port_number': signal_group}
                new_channel['native_channel_name'] = read_qstring(fid)
                new_channel['custom_channel_name'] = read_qstring(fid)
                (new_channel['native_order'], new_channel['custom_order'], signal_type, channel_enabled,
                 new_channel['chip_channel'], new_channel['board_stream']) = struct.unpack('<hhhhhh', fid.read(12))
                new_trigger_channel = {}
                (new_trigger_channel['voltage_trigger_mode'], new_trigger_channel['voltage_threshold'],
                 new_trigger_channel['digital_trigger_channel'],
                 new_trigger_channel['digital_edge_polarity']) = struct.unpack('<hhhh', fid.read(8))
                (
                new_channel['electrode_impedance_magnitude'], new_channel['electrode_impedance_phase']) = \
                    struct.unpack('<ff', fid.read(8))

                if channel_enabled:
                    if signal_type == 0:
                        header['amplifier_channels'].append(new_channel)
                        header['spike_triggers'].append(new_trigger_channel)
                    elif signal_type == 1:
                        header['aux_input_channels'].append(new_channel)
                    elif signal_type == 2:
                        header['supply_voltage_channels'].append(new_channel)
                    elif signal_type == 3:
                        header['board_adc_channels'].append(new_channel)
                    elif signal_type == 4:
                        header['board_dig_in_channels'].append(new_channel)
                    elif signal_type == 5:
                        header['board_dig_out_channels'].append(new_channel)
                    else:
                        raise Exception('Unknown channel type.')

    # Summarize contents of data file.
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_aux_input_channels'] = len(header['aux_input_channels'])
    header['num_supply_voltage_channels'] = len(header['supply_voltage_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(header['board_dig_out_channels'])

    return header


def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 60 or 128 sample datablock."""

    # Each data block contains 60 or 128 amplifier samples.
    num_samples = header['num_samples_per_data_block']
    bytes_per_block = num_samples * 4  # timestamp data
    bytes_per_block = bytes_per_block + num_samples * 2 * header['num_amplifier_channels']

    # Auxiliary inputs are sampled 4x slower than amplifiers
    bytes_per_block = bytes_per_block + (num_samples / 4) * 2 * header['num_aux_input_channels']

    # Supply voltage is sampled 60 or 128x slower than amplifiers
    bytes_per_block = bytes_per_block + 1 * 2 * header['num_supply_voltage_channels']

    # Board analog inputs are sampled at same rate as amplifiers
    bytes_per_block = bytes_per_block + num_samples * 2 * header['num_board_adc_channels']

    # Board digital inputs are sampled at same rate as amplifiers
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block = bytes_per_block + num_samples * 2

    # Board digital outputs are sampled at same rate as amplifiers
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block = bytes_per_block + num_samples * 2

    # Temp sensor is sampled 60 or 128x slower than amplifiers
    if header['num_temp_sensor_channels'] > 0:
        bytes_per_block = bytes_per_block + 1 * 2 * header['num_temp_sensor_channels']

    return int(bytes_per_block)


def read_one_data_block(data, header, indices, fid):
    """Reads one 60 or 128 sample data block from fid into data, at the location indicated by indices."""

    # In version 1.2, we moved from saving timestamps as unsigned
    # integers to signed integers to accommodate negative (adjusted)
    # timestamps for pretrigger data['
    num_samples = header['num_samples_per_data_block']
    amp_index = indices['amplifier']

    if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
        tmp = np.array(struct.unpack('<' + 'i' * num_samples, fid.read(4 * num_samples)))
        data['t_amplifier'][amp_index:(amp_index + num_samples)] = tmp
    else:
        tmp = np.array(struct.unpack('<' + 'I' * num_samples, fid.read(4 * num_samples)))
        data['t_amplifier'][amp_index:(amp_index + num_samples)] = tmp

    if header['num_amplifier_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=num_samples * header['num_amplifier_channels'])
        tmp = tmp.reshape(header['num_amplifier_channels'], num_samples)
        # the astype() call seems to do the right thing with subtract overflow values?
        data['amplifier_data'][:, amp_index:(amp_index + num_samples)] = (tmp - 32768).astype('int16')

    if header['num_aux_input_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=int((num_samples / 4) * header['num_aux_input_channels']))
        tmp = tmp.reshape(header['num_aux_input_channels'], int(num_samples/4))
        data['aux_input_data'][:, indices['aux_input']:int(indices['aux_input'] + (num_samples / 4))] = tmp

    if header['num_supply_voltage_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=1 * header['num_supply_voltage_channels'])
        tmp = tmp.reshape(header['num_supply_voltage_channels'], 1)
        data['supply_voltage_data'][:, indices['supply_voltage']:(indices['supply_voltage']+1)] = tmp

    if header['num_temp_sensor_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=1 * header['num_temp_sensor_channels'])
        tmp = tmp.reshape(header['num_temp_sensor_channels'], 1)
        data['temp_sensor_data'][:, indices['supply_voltage']:(indices['supply_voltage'] + 1)] = tmp

    if header['num_board_adc_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=num_samples * header['num_board_adc_channels'])
        tmp = tmp.reshape(header['num_board_adc_channels'], num_samples)
        tmp = (tmp - 32768).astype('int16')
        data['board_adc_data'][:, indices['board_adc']:(indices['board_adc'] + num_samples)] = tmp

    if header['num_board_dig_in_channels'] > 0:
        tmp = np.array(struct.unpack('<' + 'H' * num_samples, fid.read(2 * num_samples)))
        data['board_dig_in_raw'][indices['board_dig_in']:(indices['board_dig_in'] + num_samples)] = tmp

    if header['num_board_dig_out_channels'] > 0:
        tmp = np.array(struct.unpack('<' + 'H' * num_samples, fid.read(2 * num_samples)))
        data['board_dig_out_raw'][indices['board_dig_out']:(indices['board_dig_out'] + num_samples)] = tmp


def convert(filenames, outfile):

    headers = []
    num_blocks_per_file = []
    num_amplifier_samples = []
    num_aux_samples = []
    num_adc_samples = []
    num_dig_in_samples = []
    num_dig_out_samples = []
    num_supply_voltage_samples = []
    for filename in filenames:
        with open(filename, 'rb') as fid:
            # do a size consistency check per file
            file_size = os.path.getsize(filename)
            header = read_header(fid)
            headers.append(header)
            data_bytes = file_size - fid.tell()
            bytes_per_block = get_bytes_per_data_block(header)
            if data_bytes % bytes_per_block > 0:
                raise RuntimeError('File {} has incomplete blocks'.format(filename))

            num_blocks = data_bytes // bytes_per_block
            num_blocks_per_file.append(num_blocks)
            num_amplifier_samples.append(header['num_samples_per_data_block'] * num_blocks)
            num_aux_samples.append(int((header['num_samples_per_data_block'] / 4) * num_blocks))
            num_supply_voltage_samples.append(num_blocks)
            num_adc_samples.append(header['num_samples_per_data_block'] * num_blocks)
            num_dig_in_samples.append(header['num_samples_per_data_block'] * num_blocks)
            num_dig_out_samples.append(header['num_samples_per_data_block'] * num_blocks)

    # quick consistency check
    for key in ('num_amplifier_channels', 'num_aux_input_channels', 'num_board_adc_channels',
                'num_board_dig_in_channels', 'num_board_dig_out_channels'):
        chans = [header.get(key) for header in headers]
        if len(np.unique(chans)) > 1:
            raise RuntimeError('Different number of channels of type {}: {}'.format(key, chans))

    with h5py.File(outfile, 'w') as data:
        header = headers[0]
        if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
            data.create_dataset('t_amplifier',
                                shape=(sum(num_amplifier_samples),),
                                dtype=np.int32,
                                chunks=True)
        else:
            data.create_dataset('t_amplifier',
                                shape=(sum(num_amplifier_samples),),
                                dtype=np.uint32,
                                chunks=True)
        # Store amplifier data as signed int16 (subtract 2 ** 15 offset from blocks)
        n_amp_channels = header['num_amplifier_channels']
        if n_amp_channels:
            data.create_dataset('amplifier_data',
                                shape=(n_amp_channels, sum(num_amplifier_samples)),
                                dtype=np.int16,
                                chunks=True)
        # AUX data is unsigned -- just multiply unsigned values by 37.4e-6 for Volts
        n_aux_channels = header['num_aux_input_channels']
        if n_aux_channels:
            data.create_dataset('aux_input_data',
                                shape=(n_aux_channels, sum(num_aux_samples)),
                                dtype=np.uint16,
                                chunks=True)
        # Voltage supply is unsigned -- just multiply 74.8e-6 for Volts
        n_volt_supply_channels = header['num_supply_voltage_channels']
        if n_volt_supply_channels:
            data.create_dataset('supply_voltage_data',
                                shape=(n_volt_supply_channels, sum(num_supply_voltage_samples)),
                                dtype=np.uint16,
                                chunks=True)
        # Temp sensor unsigned -- scale by 0.01
        n_temp_channels = header['num_temp_sensor_channels']
        if n_temp_channels:
            data.create_dataset('temp_sensor_data',
                                shape=(n_temp_channels, sum(num_supply_voltage_samples)),
                                dtype=np.uint16,
                                chunks=True)

        # ADC data will be signed by subtracting 2 ** 15 offset (scale by 312.5e-6 for Volts)
        n_adc_channels = header['num_board_adc_channels']
        if n_adc_channels:
            data.create_dataset('board_adc_data',
                                shape=(n_adc_channels, sum(num_adc_samples)),
                                dtype=np.int16,
                                chunks=True)

        # by default, this script interprets digital events (digital inputs and outputs) as booleans
        # if unsigned int values are preferred(0 for False, 1 for True), replace the 'dtype=np.bool' argument with
        # 'dtype=np.uint' as shown the commented line below illustrates this for digital input data; the same can be
        # done for digital out

        n_dig_in_channels = header['num_board_dig_in_channels']
        if n_dig_in_channels:
            data.create_dataset('board_dig_in_data',
                                shape=(n_dig_in_channels, sum(num_dig_in_samples)),
                                dtype=np.bool,
                                chunks=True)
            data.create_dataset('board_dig_in_raw', shape=(sum(num_dig_in_samples),), dtype=np.uint16, chunks=True)

        n_dig_out_channels = header['num_board_dig_out_channels']
        if n_dig_out_channels:
            data.create_dataset('board_dig_out_data',
                                shape=(n_dig_out_channels, sum(num_dig_out_samples)),
                                dtype=np.bool,
                                chunks=True)
            data.create_dataset('board_dig_out_raw', shape=(sum(num_dig_out_samples),), dtype=np.uint, chunks=True)

        indices = dict()
        indices['amplifier'] = 0
        indices['aux_input'] = 0
        indices['supply_voltage'] = 0
        indices['board_adc'] = 0
        indices['board_dig_in'] = 0
        indices['board_dig_out'] = 0
        print('Reading data blocks')
        for n in range(len(filenames)):
            filename = filenames[n]
            num_blocks = num_blocks_per_file[n]
            file_size = os.path.getsize(filename)
            print('File {} ({}/{})'.format(filename, n + 1, len(filenames)))

            print_increment = 10
            percent_done = print_increment
            with open(filename, 'rb') as fid:
                header = read_header(fid)
                for i in range(num_blocks):
                    read_one_data_block(data, header, indices, fid)

                    # Increment indices
                    indices['amplifier'] += header['num_samples_per_data_block']
                    indices['aux_input'] += int(header['num_samples_per_data_block'] / 4)
                    indices['supply_voltage'] += 1
                    indices['board_adc'] += header['num_samples_per_data_block']
                    indices['board_dig_in'] += header['num_samples_per_data_block']
                    indices['board_dig_out'] += header['num_samples_per_data_block']

                    fraction_done = 100 * (1.0 * i / num_blocks)
                    if fraction_done >= percent_done:
                        print('{}% done...'.format(percent_done))
                        percent_done = percent_done + print_increment

                # Make sure we have read exactly the right amount of data.
                bytes_remaining = file_size - fid.tell()
                if bytes_remaining != 0:
                    raise Exception('Error: End of file not reached.')

        data.attrs['JSON_header'] = json.dumps(header)
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
    args = ap.parse_args()
    if args.doc:
        with open('README.txt', 'w') as fw:
            fw.write(app_description)
        sys.exit(0)

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
