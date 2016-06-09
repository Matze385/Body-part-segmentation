"""
Calculate features once avoiding recalculation in RF
produce features.h5 in format [x,y,t,c,type of feature, sigmas]
"""
import h5py 
from featureFct import * 

#parameters
sigmas = [0.3,1.0,1.6,3.5,5.0,10.0]
n_feature_types = 7
n_folds = 5
n_rot = 16

"""
sigmasGaussian = [0.3,1.0,1.6,3.5,5.0,10.0]
sigmasLoG = [0.3,1.0,1.6,3.5,5.0,10.]
sigmasGGM = [0.3,1.0,1.6,3.5,5.0,10.]
sigmasSTE = [0.3,1.0,1.6,3.5,5.0,10.]
sigmasHoGE = [0.3,1.0,1.6,3.5,5.0,10.]
"""

#calc all std features on unrotated and rotated rawdata 
calc_std_features('trainingsImages.h5','features.h5', n_folds, sigmas)      
#calc_std_features('trainingsImagesRot.h5','featuresRot.h5', n_folds*n_rot, sigmas)      

