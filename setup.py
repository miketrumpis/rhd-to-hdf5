from setuptools import setup

if __name__=='__main__':
    setup(
        name = 'rhd_to_hdf5',
        version = '0.1',
        scripts=['convert_rhd.py'],
        install_requires=['numpy', 'h5py']
    )