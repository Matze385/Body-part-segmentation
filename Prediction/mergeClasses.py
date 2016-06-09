import numpy as np

#merge classes of hard segementation to one class 
def merge_classes(y_label, merger):
#y_lable: array of shape (samples) with labels
#merger:1-D array with indices of classes that should be merged, eg. [1,2,3] if classes 1,2,3 should be merged
    v_merge = np.vectorize(lambda x: merger.min() if x in merger else x)    
    return v_merge(y_label)

#add prob maps in merger to soft segmentation :
def merge_probs(prob_map, merger):
#prob_map: [x,y,n_classes]
    prob_map_new = np.zeros((prob_map.shape[0], prob_map.shape[1]), dtype=np.float32)
    for merged_class in merger:
        #-1: class begin with 1 array indices with 0
        prob_map_new[:,:] += prob_map[:,:,merged_class-1] 
    return prob_map_new
