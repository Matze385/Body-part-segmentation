import numpy as np

#merge classes to one class 
def merge_classes(y_label, merger):
#y_lable: array of shape (samples) with labels
#merger:1-D array with indices of classes that should be merged, eg. [1,2,3] if classes 1,2,3 should be merged
    v_merge = np.vectorize(lambda x: merger.min() if x in merger else x)    
    return v_merge(y_label)
