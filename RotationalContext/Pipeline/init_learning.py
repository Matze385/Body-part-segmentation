import h5py
import numpy as np


learningF = h5py.File('learningcurve.h5','w')
learningF.create_dataset('xAxis', (1000,), dtype = np.float32)
learningF.create_dataset('f1_measure', (1000, 20), dtype = np.float32)
learningF.create_dataset('accuracy', (1000, 20), dtype = np.float32)
learningF.create_dataset('precision', (1000, 20), dtype = np.float32)
learningF.create_dataset('recall', (1000, 20), dtype = np.float32)
learningF.create_dataset('index', (1,), dtype = np.int32)
learningF['index'][0] = 0
learningF.close()
