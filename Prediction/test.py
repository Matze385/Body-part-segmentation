import h5py
import numpy as np



with h5py.File('movie500.h5','r') as f:
    print f['data'].shape
