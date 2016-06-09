import numpy as np
import h5py
import vigra as vg

from sklearn.ensemble import RandomForestClassifier

from featureFct import *
from Functions import * 
from mergeClasses import *

"""
parameters
"""

#random forest parameters
N_TREES = 600
N_STAGES = 2
#input: file with rawdata for prediction
filename_rawdata = 'movie500.h5'
#output: filename for saving segmentation results and number of predicted frames
filename_seg_hard = 'hard_segmentation.h5'
#if zero only soft prediction otherwise (eg. 1) n_seg_soft_predicted frames are hard segmented
n_seg_hard_predicted = 1
filename_seg_soft = 'prob_segmentation.h5'
#equal to number of predicted frames, must be no
n_seg_soft_predicted = 4
#possible values for sigmas: 0.3, 1.0, 1.6, 3.5, 5.0, 10.0
#sigmas for features on raw data
sigmasGaussian = [0.3,1.0,1.6,3.5, 5.0, 10.0]
sigmasLoG = [0.3,1.0,1.6,3.5, 5.0, 10.0]
sigmasGGM = [0.3,1.0,1.6,3.5, 5.0, 10.0]
sigmasSTE = [0.3,1.0,1.6,3.5, 5.0, 10.0]# [1.0,1.6,3.5]
sigmasHoGE = [0.3,1.0,1.6,3.5, 5.0, 10.0]#[1.0,1.6,3.5]
#sigmas for features on prob maps
sigmasGaussian_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]
sigmasLoG_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]
sigmasGGM_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]
sigmasSTE_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]# [1.0,1.6,3.5]
sigmasHoGE_prob = [0.3,1.0,1.6,3.5, 5.0, 10.0]#[1.0,1.6,3.5]

#number of features
n_feat_prob = len(sigmasGaussian_prob)+len(sigmasLoG_prob)+len(sigmasGGM_prob)+2*len(sigmasSTE_prob)+2*len(sigmasHoGE_prob)
n_feat_raw = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)

#dataaugmentation, rotational settings
X_CENTER = 510		                #center coordinates for rotation
Y_CENTER = 514
RADIUS = 500
N_ROT = 16                              #number of rotations per image, determines rotational angle
N_CLASSES = 7                           #number of different labels, here edge, background, boundary, head, upper body, lower body, wings
#testparameters
#auxiliary classes that should be merged by testing
merged_classes = np.array([1,2,3])              #classes background(1) edge (2) boundary(3) are merged to background for testing
#image used for qualitative analysis (idx 4 corresponds to image 461)
idx_test_img = 4 

#derived parameters (set automatically later by programm)
xDim = 0				
yDim = 0
#number of used features
n_features = 0
#number of folds
n_folds = 0
#derived automatically out of file filename_rawdata
dFrames = h5py.File(filename_rawdata, 'r')
n_frames_out = dFrames['data'].shape[0]        #format of rawdata [t,x,y,c] 
dFrames.close()


"""
read in data
"""

labelsF = h5py.File('CrossValFolds.h5','r')
#folds: [img, x/y, selected_pixels] coordinates of labels
folds = vg.readHDF5(labelsF,'folds')
#labels: labels [img, selected_pixels] c: class labels c=0 no label
labels = vg.readHDF5(labelsF,'labels')
labelsF.close()

#labels_all: [img, x, y, 0] needed for test sets with pixels out of complete interesting region
labelsAllF = h5py.File('Labels012349.h5','r')
labels_all = labelsAllF['exported_data'][:,:,:,0]
labelsAllF.close()

print("read in finished")
n_folds  = labels.shape[0]
print ('n_folds:', n_folds)

#n_features = features.shape[3]
xDim = labels_all.shape[1]
yDim = labels_all.shape[2]


#feature selection on unrotated data
#dont forget to integrate following comments into code when using subset of std feat on rawdata
#select_std_features(0,'rawdata','features.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE) #features [x,y,img,f]
#select_std_features(0,'rawdata_rotated', 'featuresRot.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE)
print("feature calc finished")

#array for number of trainingssamples in different trainingssplits of crossval
n_train = np.zeros((n_folds))



"""
main programm
"""

#n_fold crossvalidation1

for testindex in np.arange(1):#n_folds):
    #do not forget to comment out when using testindex in arange(n_folds)
    testindex += 4
    """
    training
    """
    #list for RF of different stages
    clf = []
    #cummulate trainingssamples in different stages
    n_train_1_stage = 0
    #train different stages
    for i_stage in np.arange(N_STAGES):
        #train stages on different trainingssets
        if i_stage==0:
            X_train, y_train= train_sets_stages_several_files(0, N_STAGES, labels, folds,['selected_std_feat_rawdata_stage_0.h5'], testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            #X_train, y_train= train_sets_stages(0, N_STAGES, labels, folds,'selected_std_feat_rawdata_stage_0.h5', testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            n_train_1_stage += len(y_train)
            #add RF object to list clf
            clf.append(RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1))
            #train random forest of this stage
            clf[i_stage].fit(X_train, y_train)
        else:
            #indices of images prob maps are computed on
            images = img_split(i_stage, N_STAGES, testindex, n_folds, labels, rot=False, n_rot=N_ROT)
            #images = img_split(i_stage-1, n_stages, testindex, n_folds, labels, rot=True, n_rot=N_ROT)
            print images
            #create prob map of RF of stage before, saved in 'probability_map_stage_'+str(i_stage).h5 in dataset 'data' array shape [img,x,y,c]
            create_probability_maps(i_stage-1, clf[i_stage-1], n_folds, images, N_CLASSES)
            #create_probability_maps(i_stage-1, clf[i_stage-1], n_folds*N_ROT, images, N_CLASSES, rot=True)
            print('prob maps finished of stage: ',i_stage)  
            #delete features out of prob maps of stage before and prob_map of two stages before if i_stage>=2 
            if i_stage>1:
                filename_std_feat_out_of_prob_before = 'selected_std_feat_prob_stage_'+str(i_stage-1)+'.h5' 
                f = h5py.File(filename_std_feat_out_of_prob_before,'w')
                f.create_dataset('data',(1,))
                f.close()
                filename_prob_before = 'probability_map_stage_'+str(i_stage-2)+'.h5'
                f = h5py.File(filename_prob_before,'w')
                f.create_dataset('data',(1,))
                f.close()
            #calculation of features out of prob maps of stage before, saved in 'selected_std_feat_prob_stage_'+str(i_stage)+'.h5'
            filename_prob = 'probability_map_stage_'+str(i_stage-1)+'.h5'
            filename_std_feat_out_of_prob = 'selected_std_feat_prob_stage_'+str(i_stage)+'.h5' 
            #change line down
            calc_selected_std_features(filename_prob, filename_std_feat_out_of_prob, n_folds, images, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
            #for data augmentation
            #calc_selected_std_features(filename_prob, filename_std_feat_out_of_prob, n_folds*N_ROT, images, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
            print('feature calc out of prob maps finished')
            feature_filenames = ['selected_std_feat_rawdata_stage_0.h5',filename_std_feat_out_of_prob]
            X_train, y_train= train_sets_stages_several_files(i_stage, N_STAGES, labels, folds, feature_filenames, testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            #feature_filenames = ['selected_std_feat_rawdata_rotated_stage_0.h5',filename_std_feat_out_of_prob]
            #X_train, y_train= train_sets_stages_several_files(i_stage, N_STAGES, labels, folds, feature_filenames, testindex, rot=True, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            print('trainingsdata sampling finished')
            #add RF object to list clf
            clf.append(RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1))
            #train random forest of this stage
            clf[i_stage].fit(X_train, y_train)
    

    n_train =  n_train_1_stage*N_STAGES
    print('train number', n_train)
    
    """
    prediction for rawdata
    """
    #write to Prob Map
    d = h5py.File(filename_seg_soft, 'w')
    d.create_dataset("data", (n_frames_out, xDim, yDim, N_CLASSES), dtype=np.uint32)
    h = h5py.File(filename_seg_hard, 'w')
    h.create_dataset("data", (n_frames_out, xDim, yDim, 1), dtype=np.uint32)
    #do prediction
    for img in np.arange(n_seg_soft_predicted):
        y2D_pred_soft, y2D_pred_hard = predict_one_image_new_data(img, filename_rawdata ,clf, N_CLASSES, sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
        if n_seg_soft_predicted != 0:
            d['data'][img, :,:,:] = y2D_pred_soft[:,:,:]
        if n_seg_hard_predicted != 0:
            #merge auxiliary background classes
            y2D_pred_hard = merge_classes(y2D_pred_hard.reshape((xDim*yDim,)), merged_classes).reshape((xDim, yDim))
            h['data'][img, :,:,0] = y2D_pred_hard[:,:]
        print(img)
    d.close()
    h.close()
    
     
   
    


