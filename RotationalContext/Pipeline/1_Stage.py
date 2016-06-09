import numpy as np
import h5py
import vigra as vg

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score
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


"""
textdocument for documentation of parameters and results
"""
results = open("documentation.txt", "w")
results.write("1.stage"+"\n"+"\n")
results.write("N_TREES = "+str(N_TREES)+"\n")
results.write("Sigmas Gaussian Smoothing = "+str(sigmasGaussian)+"\n")
results.write("Sigmas LoG = "+str(sigmasLoG)+"\n")
results.write("Sigmas Gaussian Gradient Magnitude = "+str(sigmasGGM)+"\n")
results.write("Sigmas Structure Tensor Eigenvalues = "+str(sigmasSTE)+"\n")
results.write("Sigmas Hessian of Gaussian Eigenvalues = "+str(sigmasHoGE)+"\n")


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

#write f1_score in h5py file for plotting learning curve later
learningF = h5py.File('learningcurve.h5','r+')
#learningcurve contain dataset 'data' with shape (3,1000) first idx corresponds to number of train pixels, average f1_score, std of f1_score in that order 
#learningcurve contain dataset 'index' of shape(1) to store running idx for insertion next data
#index for insertion of score
learn_idx = learningF['index'][0]

#feature selection on unrotated data
#dont forget to integrate following comments into code when using subset of std feat on rawdata
#select_std_features(0,'rawdata','features.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE) #features [x,y,img,f]
#select_std_features(0,'rawdata_rotated', 'featuresRot.h5', sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE)
print("feature calc finished")

#save results for confusion matrix
confMatrF = h5py.File('ConfusionMatrix.h5','w')
confMatrF.create_dataset('data', (5,5))

#save results fo feature importance
featImpF = h5py.File('featImp.h5','w')
featImpF.create_dataset('stage1', (1,n_feat_raw), dtype=np.float32)
featImpF.create_dataset('higher_stages', (N_STAGES,n_feat_raw+N_CLASSES*n_feat_prob), dtype=np.float32)

"""
create array scores and n_train to save test measures and number of trainingspxl
"""
#f1 score for each merged class and average and fly composite class
n_f1_measures = 6
#accuracy for each merged class and average and fly composite class
n_accuracy_measures = 6
#precision for each merged class and average and fly composite class
n_precision_measures = 6
#recall for each merged class and average and fly composite class
n_recall_measures = 6
#number of saved metrics 
n_measures = n_f1_measures+n_accuracy_measures+n_precision_measures+n_recall_measures
#save scores for 5 different classes and average
scores = np.zeros((n_folds,n_measures))
#array for number of trainingssamples in different trainingssplits of crossval
n_train = np.zeros((n_folds))



"""
main programm
"""

#n_fold crossvalidation1

for testindex in np.arange(1):#n_folds):
    #do not forget to comment out when using testindex in arange(n_folds)
    #testindex += 1
    """
    training
    """
    #list for RF of different stages
    clf = []
    #cummulate trainingssamples in different stages
    n_train_cum = 0
    #train different stages
    for i_stage in np.arange(N_STAGES):
        #train stages on different trainingssets
        if i_stage==0:
            X_train, y_train= train_sets_stages_several_files(0, N_STAGES, labels, folds,['selected_std_feat_rawdata_stage_0.h5'], testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            #X_train, y_train= train_sets_stages(0, N_STAGES, labels, folds,'selected_std_feat_rawdata_stage_0.h5', testindex, rot=False, n_rot=N_ROT, x_center=X_CENTER, y_center=Y_CENTER)
            n_train_cum += len(y_train)
            #add RF object to list clf
            clf.append(RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1))
            #train random forest of this stage
            clf[i_stage].fit(X_train, y_train)
            #save feature importance
            featImpF['stage1'][0,:] += clf[i_stage].feature_importances_[:] 
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
            #save feature importance
            featImpF['higher_stages'][i_stage-1,:] += clf[i_stage].feature_importances_[:]
    #delete features out of prob maps of last stage 
    #filename_std_feat_out_of_prob_last_stage = 'selected_std_feat_prob_stage_'+str(N_STAGES-1)+'.h5' 
    #f = h5py.File(filename_std_feat_out_of_prob_last_stage,'w')
    #f.create_dataset('data',(1,))
    #f.close()

    n_train[testindex] =  n_train_cum*N_STAGES
    print('train number', n_train[testindex])
    
    """
    save feature importance
    """
    
    
    """
    prediction of image corresponding to testfold
    """
    y2D_pred = predict_one_image(testindex, clf, N_CLASSES, sigmasGaussian_prob, sigmasLoG_prob, sigmasGGM_prob, sigmasSTE_prob, sigmasHoGE_prob)
    print('prediction finished')
    y2D_test = np.zeros((xDim, yDim), dtype=np.int32)
    y2D_test = labels_all[testindex,:,:]
    y_pred = y2D_to_y1D_in_circle(y2D_pred, X_CENTER, Y_CENTER, RADIUS)
    y_test = y2D_to_y1D_in_circle(y2D_test, X_CENTER, Y_CENTER, RADIUS)
    #create test sets consisting of features and labels of one fold
    #X_test, y_test = test_sets(labels, folds, features, testindex)
    #create test sets consisting of features and labels of one complete interesting region in img of corresponding fold
    #X_test, y_test = test_sets_all(labels_all, X_CENTER, Y_CENTER, RADIUS, 'selected_std_feat_rawdata_stage_0.h5', testindex, rot=False)
    #y_pred = clf[0].predict(X_test)

    
    """
    #predict one image for qualitative segmentation analysis
    if testindex<n_folds:
    """
    """
        testF = h5py.File('selected_std_feat_rawdata_stage_0.h5','r')
        features_pred = testF['data'][:,:,testindex,:].reshape((xDim*yDim,testF['data'].shape[3]))
        testF.close()
        y_pred = clf[0].predict(features_pred)
        y_pred = merge_classes(y_pred, merged_classes)
        print len(filter(lambda x: x==1, y_pred))
        print len(filter(lambda x: x==2, y_pred))
        print len(filter(lambda x: x==3, y_pred))
        print len(filter(lambda x: x==4, y_pred))
        print len(filter(lambda x: x==5, y_pred))
        print len(filter(lambda x: x==6, y_pred))
        print len(filter(lambda x: x==7, y_pred))
        print len(filter(lambda x: x==0, y_pred))
    """
    
    #copy y2Dpred to let it unchanged    
    pred = np.zeros((xDim, yDim), dtype=np.uint32)
    pred[:,:] = y2D_pred
    #take look only on intersting region
    rr, cc = circle(Y_CENTER, X_CENTER, RADIUS)
    circle_mask = np.ones((xDim, yDim), dtype=np.uint32)
    circle_mask[rr,cc] = 0
    x, y = np.nonzero(circle_mask)
    pred[x,y] = np.zeros((len(x),))
    #write to Segmentation Map
    d = h5py.File('1_SegmentationMap.h5','r+')
    #d.create_dataset("data", data=pred)
    d['data'][testindex, :,:,0] = pred[:,:]
    d.close()



    """
    begin testing with test fold
    """
    #merge background classes
    y_pred = merge_classes(y_pred, merged_classes)
    y_test = merge_classes(y_test, merged_classes)
    #merge to composite fly
    y_pred_fly = merge_classes(y_pred, np.array([4,5,6,7]))
    y_test_fly = merge_classes(y_test, np.array([4,5,6,7]))
    idx_insertion = 0
    #scores[testindex, 0] = f1_score(y_test, y_pred, labels = [4,5,6,7,1], average = 'macro')
    #f1 score for each class
    scores[testindex, 1:n_f1_measures-1] = f1_score(y_test, y_pred, labels = [1,4,5,6,7], average = None)[1:n_f1_measures-1]
    scores[testindex, 0] = scores[testindex, 1:n_f1_measures-1].mean()
    print('f1-score fold', testindex, ': ', scores[testindex, 1:n_f1_measures-1].mean())
    #f1 score for composite fly
    scores[testindex, n_f1_measures-1] = f1_score(y_test_fly,y_pred_fly, pos_label=4  )
    idx_insertion = n_f1_measures
    #accuracy 
    conf_matrix = confusion_matrix(y_test, y_pred)
    confMatrF['data'][:,:] += conf_matrix[:,:]
    #cut backgroundclass out
    #conf_matrix_small = np.delete(conf_matrix, 0,0)
    #conf_matrix_small = np.delete(conf_matrix_small, 0,1)    
    accuracy_class = np.zeros((4,), dtype = np.float32 )
    for cl in np.arange(4):
        #confusion matrix without row and column of considered class for true negative counting
        cl += 1
        conf_matrix_true_negative = np.delete(conf_matrix, cl,0)
        conf_matrix_true_negative = np.delete(conf_matrix_true_negative, cl,1)
        n_true_negative = conf_matrix_true_negative.sum()
        accuracy_class[cl-1] = float(conf_matrix[cl,cl]+n_true_negative)/float(conf_matrix.sum())
    scores[testindex, idx_insertion] = accuracy_class.mean()
    scores[testindex, idx_insertion+1:idx_insertion+n_accuracy_measures-1] = accuracy_class[:]
    #accuracy for composite fly
    scores[testindex, idx_insertion+n_accuracy_measures-1] = accuracy_score(y_test_fly, y_pred_fly)
    idx_insertion += n_accuracy_measures
    #print('score:',score)
    #save precision
    #scores[testindex, idx_insertion] = precision_score(y_test, y_pred, labels = [1,4,5,6,7], average = 'macro')
    scores[testindex, idx_insertion+1: idx_insertion+n_precision_measures-1] = precision_score(y_test, y_pred, labels = [1,4,5,6,7], average = None)[1:n_precision_measures-1]
    scores[testindex, idx_insertion] = scores[testindex, idx_insertion+1: idx_insertion+n_precision_measures-1].mean()
    #precision for composite fly
    scores[testindex, idx_insertion+n_precision_measures-1] = precision_score(y_test_fly, y_pred_fly, pos_label=4)
    idx_insertion += n_precision_measures
    #save recall    
    #scores[testindex, idx_insertion] = recall_score(y_test, y_pred, labels = [4,5,6,7], average = 'macro')
    scores[testindex, idx_insertion+1: idx_insertion+n_recall_measures-1] = recall_score(y_test, y_pred, labels = [1,4,5,6,7], average = None)[1:n_recall_measures-1]
    scores[testindex, idx_insertion] = scores[testindex, idx_insertion+1: idx_insertion+n_recall_measures-1].mean()
    #recall for composite fly
    scores[testindex, idx_insertion+n_recall_measures-1] = recall_score(y_test_fly, y_pred_fly, pos_label=4)



"""
write testresults in hdf5 file
"""

#write score in hdf5 file for learning curve
learningF['xAxis'][learn_idx] = np.mean(n_train)
#columns in dataset 'f1_measure' are f1 average, error f1 average, f1 single class, error f1 single class, 
idx_in_scores = 0
for score_idx in np.arange(n_f1_measures):
    learningF['f1_measure'][learn_idx, score_idx*2] = scores[:,score_idx].mean()
    learningF['f1_measure'][learn_idx, score_idx*2+1] = scores[:, score_idx].std()/np.sqrt(n_folds)
print('f1_score: ',scores[:,0].mean(),'+-', scores[:, score_idx].std()/np.sqrt(n_folds))
idx_in_scores = n_f1_measures
#columns in dataset 'accuracy' are accuracy, error accuracy
for score_idx in np.arange(n_accuracy_measures):
    learningF['accuracy'][learn_idx, score_idx*2] = scores[:, idx_in_scores+score_idx].mean()
    learningF['accuracy'][learn_idx, score_idx*2+1] = scores[:, idx_in_scores+score_idx].std()/np.sqrt(n_folds)
idx_in_scores += n_accuracy_measures
#columns in dataset 'precision' are precision average, error precision average, precision single class, error precision single class
for score_idx in np.arange(n_precision_measures):
    learningF['precision'][learn_idx, score_idx*2] = scores[:,idx_in_scores+score_idx].mean()
    learningF['precision'][learn_idx, score_idx*2+1] = scores[:,idx_in_scores+score_idx].std()/np.sqrt(n_folds)
idx_in_scores += n_precision_measures
#columns in dataset 'recall' are recall average, error recall average, recall single class, error recall single class
for score_idx in np.arange(n_recall_measures):
    learningF['recall'][learn_idx, score_idx*2] = scores[:, idx_in_scores+score_idx].mean()
    learningF['recall'][learn_idx, score_idx*2+1] = scores[:, idx_in_scores+score_idx].std()/np.sqrt(n_folds)


learningF['index'][0] = learn_idx+1
learningF.close()

confMatrF.close()
featImpF.close()

results.close()



