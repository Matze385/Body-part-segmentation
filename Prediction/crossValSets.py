"""
produce 10 folds for  crossvalidation and save it in foldsCoord.h5 in dataset 'data'
-each fold on different picture
-number of total trainingspixels is flexible, is 9 times number of fold size

for data augmentation through rotation 
-for all pixels within each fold rotated pixel are added, approximate 
-number of trainingspixels in each frame is stored in n
"""

import sys, getopt
import numpy as np
import h5py
import vigra as vg
import skimage as skim

from skimage.transform import rotate 
from sampling import *

#parameters
N_TRAIN = 500.                         #approximate number of trainingspixel in total, reset by command line parameter
N_IMG = 5                               #number of labeled images correspond to number of folds in crossvalidation
N_CLASS = 7                             #number of classes

"""
X_CENTER = 510		                #center coordinates for rotation
Y_CENTER = 514
N_ROT = 16                              #number of rotations per image, determines rotational angle
"""

#derived parameters
n_pix_per_class = 0     #number of pixel per class per image
size_of_fold = 0        #size of each fold, is also size of testset
n_train_exact = 0       #exact number of trainingspixel for unrotated case
n_train_exact_rot=0

def main(argv):
    #N_train is given by command line parameter
    try:
        opts, args = getopt.getopt(argv,"hn:",["n_train="])
    except getopt.GetoptError:
        print 'crossValSets.py -n <number of train pxl>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'crossValSets.py -n <number of train pxl>'
            sys.exit()
        elif opt in ("-n", "--n_train"):
            N_TRAIN = int(arg)
    #set derived parameters
    n_pix_per_class = int(round(N_TRAIN/(N_IMG-1)/N_CLASS))     #number of pixel per class per image
    size_of_fold = n_pix_per_class*N_CLASS  #size of each fold, is also size of testset
    n_train_exact = size_of_fold*(N_IMG-1)          #exact number of trainingspixel for unrotated case

    with h5py.File('Labels012349.h5','r') as train_lab_f: #trainingsLabels.h5
        #data structure to store selected coordinates of pixels for test/training-set for each image, number of rotated labels is due to nearest neighbor interpolation not fix->size_of_fold*1.1
        """
        coord = np.zeros((N_IMG*N_ROT, 2, int(size_of_fold*1.1)), dtype=np.int32) 
        labels = np.zeros((N_IMG*N_ROT,int(size_of_fold*1.1)), dtype=np.uint8)
        length = np.zeros((N_IMG*N_ROT), dtype=np.int32)
        """
        coord = np.zeros((N_IMG, 2, size_of_fold), dtype = np.int32)
        labels = np.zeros((N_IMG, size_of_fold), dtype=np.uint8)
        for i in np.arange(N_IMG):
            lab_img = train_lab_f['exported_data'][i,:,:,0]
            selected_map = selectPixelsBalanced(lab_img, n_pix_per_class, N_CLASS)
            #save coordinates
            x, y = np.nonzero(selected_map)
            coord[i,0,:] = x
            coord[i,1,:] = y
            labels[i,:] = selected_map[x,y]

        """
            angles = np.linspace(0., 360., num=N_ROT, endpoint=False )
            for j,angle in enumerate(angles):
                #order=0 for preserving ones, no interpolation
                selected_map_rot = rotate(selected_map, angle, resize = False, center = (Y_CENTER,X_CENTER), order =0, preserve_range=True)  
                selected_map_rot = selected_map_rot.astype(np.uint8)
                #x and y axis are in usual order?
                x, y = np.nonzero(selected_map_rot)
                #print(j, len(x), coord.shape[2], labels.shape[1])
                le = len(x)
                length[i*N_ROT+j] = le
                coord[i*N_ROT+j,0,:le] = x
                coord[i*N_ROT+j,1,:le] = y
                labels[i*N_ROT+j,:le] = selected_map_rot[x,y]
                n_train_exact_rot += le
        """
        with h5py.File('CrossValFolds.h5','w') as foldsF:
            #format of coord [n_img_rot, x/y, sample]: coord
            foldsF.create_dataset('folds', data=coord)       
            #format of labels [n_img_rot,sample]: label, labels with 0 are not used 
            foldsF.create_dataset('labels', data=labels) 
            """    
            #format of length [n_img_rot]: number of selected pixels in img
            foldsF.create_dataset('length', data=length)     
            """ 

if __name__ == "__main__":
   main(sys.argv[1:])

     


