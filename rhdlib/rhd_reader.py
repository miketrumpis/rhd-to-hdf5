import os
from pathlib import Path
import struct
import numpy as np


# Possible data streams
rhd_stream_channels = ('amplifier_data',
                       'aux_input_data',
                       'supply_voltage_data',
                       'temp_sensor_data',
                       'board_adc_data',
                       'board_dig_in_raw',
                       'board_dig_out_raw')


# Streams to convert to signed ints
signed_arrays = ('amplifier_data', 'board_adc_data')


class RHDFile:
    """
    Map an RHD using a structured array to represent block items

    """
    def __init__(self, file_path: Path):
        """
        Parameters
        ----------
        file_path: Path
            Path to the RHD file

        """
        self.file_path = file_path
        with open(file_path, 'rb') as fid:
            self.header = read_header(fid)
            self.header_bytes = fid.tell()
            s = os.stat(self.file_path)
            data_bytes = s.st_size - self.header_bytes
        self.samps_per_block = self.header['num_samples_per_data_block']
        # Start the string that will encode the entire block structure
        if is_v12(self.header):
            dtype_code = [('t_amplifier', '<{:d}i'.format(self.samps_per_block))]
        else:
            dtype_code = [('t_amplifier', '<{:d}I'.format(self.samps_per_block))]
        # Start a corresponding list of (name, num_channels) for each block section
        struct_map = [('t_amplifier', (1, self.samps_per_block))]
        rates = self.header['frequency_parameters']
        base_rate = rates['amplifier_sample_rate']
        for stream in rhd_stream_channels:
            stream_base = '_'.join(stream.split('_')[:-1])
            num_chans = self.header['num_' + stream_base + '_channels']
            if num_chans:
                rate = rates[stream_base + '_sample_rate']
                num_samps = int(self.samps_per_block * rate / base_rate)
                # struct_code = struct_code + '{:d}H'.format(num_chans * num_samps)
                dtype_code.append((stream, '<{:d}H'.format(num_chans * num_samps)))
                struct_map.append((stream, (num_chans, num_samps)))
        self.dtype_code = dtype_code
        self.struct_map = struct_map
        num_blocks = data_bytes / np.dtype(dtype_code).itemsize
        if num_blocks != int(num_blocks):
            print('WARNING: This file has incomplete blocks ({})'.format(os.path.split(self.file_path)[-1]))
        num_blocks = int(num_blocks)
        self.array_sizes = dict([(name, (c, b * num_blocks)) for name, (c, b) in self.struct_map])
        self.rhd_map = np.memmap(self.file_path, dtype=np.dtype(dtype_code), offset=self.header_bytes, mode='r')

    def to_arrays(self, arrays=dict(), offsets=dict(), apply_scales=False):
        """
        Read from the memory map and append to existing arrays (with offsets), or create new arrays.

        """
        scales = {'amplifier_data': 0.195,
                  'aux_input_data': 37.4e-6,
                  'supply_voltage_data': 74.8e-6,
                  'temp_sensor_data': 0.01}
        if self.header['eval_board_mode'] == 1:
            scales['board_adc_data'] = 152.59e-6
        elif self.header['eval_board_mode'] == 13:
            scales['board_adc_data'] = 312.5e-6
        else:
            scales['board_adc_data'] = 50.354e-6
        for name, shape in self.struct_map:
            map_arr = self.rhd_map[name]
            blocks = map_arr.shape[0]
            chans, block_samps = shape
            total_samples = blocks * block_samps
            scale = scales.get(name, 1) if apply_scales else 1
            if name in signed_arrays:
                dtype = 'd' if apply_scales else 'h'
                mem_arr = (map_arr.reshape(blocks, chans, block_samps) - 32768).astype(dtype)
            else:
                dtype = 'd' if apply_scales else 'H'
                mem_arr = map_arr.reshape(blocks, chans, block_samps).astype(dtype, copy=False)
            if scale != 1:
                mem_arr *= scale
            if name in arrays:
                offset = offsets.get(name, 0)
                sl = np.s_[offset:offset + total_samples]
                if chans > 1:
                    sl = (slice(None), sl)
                arrays[name][sl] = mem_arr.transpose(1, 0, 2).reshape(chans, total_samples).squeeze()
                offsets[name] = offset + total_samples
            else:
                arrays[name] = mem_arr.transpose(1, 0, 2).reshape(chans, total_samples).squeeze()
        return arrays


def is_v12(header):
    return (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1)


# The following code is provided by Intan, copyright
# Michael Gibson 17 July 2015
# Modified Adrian Foy Sep 2018


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
