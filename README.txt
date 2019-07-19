Combine and convert packetized Intan RHD files into continuous arrays in HDF5 format.
   
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
