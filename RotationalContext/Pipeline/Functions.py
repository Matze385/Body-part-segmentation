import numpy as np
import vigra as vg
import matplotlib.pyplot as plt
import skimage as skim
import h5py


from skimage.draw import circle
from skimage.transform import rotate 


"""
helper-fct for train_sets_stages fct
partition indices of given length into n_groups of almost equal size(maximal diffference in length of one, first groups get evt. one index more), each group consist of subsequent indices 
"""
def split_equ(i, n_groups, length):
#return start imdex and length of i group as tuple (start_idx, length_i)
#i: index of group between 0 and n_groups-1
#n_groups: number of groups
    appr_length_group = int(length/n_groups)
    rest = length-appr_length_group*n_groups
    if i < rest:
        start_idx = (appr_length_group+1)*i
        length_i = appr_length_group+1
        return (start_idx, length_i)
    else:
        start_idx = (appr_length_group+1)*rest+(i-rest)*appr_length_group
        length_i = appr_length_group
        return (start_idx, length_i)

"""
helper-fct for train_sets_stages fct
given a range defined by a start index and length (summarized in variable split) and the number of labels per fold the function calculates the folds corresponding to this range and the start and end index 
"""
def split_img_idx(split, n_labels_per_fold):
#returns tuple of (start_img, start_idx, end_img, end_idx)
#split contains start index and length
#n_labels_per_fold: number of labels within one fold/image
    #image index of first image used for trainingsdata
    start_img = split[0]//n_labels_per_fold
    start_idx = split[0]-start_img*n_labels_per_fold
    #end_img: last used image
    end_img = (split[0]+split[1]-1)//n_labels_per_fold
    end_idx = 0
    if end_img-start_img>=1:
        end_idx = split[1]-(end_img-start_img-1)*n_labels_per_fold-(n_labels_per_fold-start_idx)
    else:
        end_idx = split[1]+start_idx
    return (start_img, start_idx, end_img, end_idx) 


"""
helper-fct of train_sets_stages
given a map of choosen labels (selected_map) of an unrotated image (with index img) and a feature map of rotated versions of this image the function produces trainingsdata and labels for rotated versions by selecting the matching rotated features and with its labels 
"""
def rotate_features_labels(img, selected_map, feature_filename, n_rot, y_center, x_center):
#img: index of img the selected_map correspond to
#selected_map: array with shape of image with 0 for no labeling and 1,2,.. for labeled with class 1,2,..
#features feature_array [x,y,imgRot,f] 
#n_rot: number of rotations per image
#x_center, y_center: coordinates of center for rotation
    
    #open feature file
    featuresF = h5py.File(feature_filename,'r')
    #parameters
    xDim = featuresF['data'].shape[0]
    yDim = featuresF['data'].shape[1]
    n_features = featuresF['data'].shape[3]
    #create array for features of one image
    features = np.zeros((xDim, yDim,n_features), dtype=np.float32)
    #estimate maximal size of trainingsset
    x, y = np.nonzero(selected_map)
    X_train_big = np.zeros((int(len(x)*1.1)*n_rot, n_features))
    y_train_big = np.zeros((int(len(x)*1.1)*n_rot))
    insert_idx = 0
    angles = np.linspace(0., 360., num=n_rot, endpoint=False )
    for j,angle in enumerate(angles):
    #order=0 for preserving ones, no interpolation
        selected_map_rot = rotate(selected_map, angle, resize = False, center = (y_center, x_center), order =0, preserve_range=True)  
        selected_map_rot = selected_map_rot.astype(np.uint8)
        x_rot, y_rot = np.nonzero(selected_map_rot)
        features = featuresF['data'][:,:,img*n_rot+j,:]
        X_train_big[insert_idx:insert_idx+len(x_rot),:] = features[x_rot, y_rot, :]
        y_train_big[insert_idx:insert_idx+len(x_rot)] = selected_map_rot[x_rot,y_rot]
        insert_idx += len(x_rot)
    X_train = np.zeros((insert_idx, n_features))
    y_train = np.zeros((insert_idx))
    X_train = X_train_big[:insert_idx, :]
    y_train = y_train_big[:insert_idx]
    featuresF.close()
    return (X_train, y_train)


"""
produce trainings feature and label array for stage i of n_stages in total
"""
def train_sets_stages(stage_i, n_stages, labels, coord, feature_filename, testindex, rot=False, n_rot=1, x_center=0, y_center=0):
#returns tuple of training features and labels  [X_train, y_train] X_train [nPixelTrain, nFeatures] y_train [nPixelTrain] 
#stage_i: uint between 0 and n_stages-1 indicating for wich stage trainingsdata are used
#n_stages: n of stages in total 
#labels: labels [img, selected_pixels] c: class labels c=0 no label
#coord: [img, x/y, selected_pixels] coordinates of labels
#feature_filename: filename of hdf5 file containing dataset 'data' with feature_array [x,y,img/imgRot,f] img/imgRot depend on rot, if rot is true imgRot 
#testindex: between 0 and 9 indicate fold used for testing all others are used for training
#n_rot: number of rotations per image
#rot: if rotated images are used rot=True
#x_center, y_center: coordinates of center for rotation
    
    #open feature file
    featuresF = h5py.File(feature_filename,'r')

    #global parameters
    n_folds = int(labels.shape[0])
    n_labels_per_fold = labels.shape[1] 
    n_features = featuresF['data'].shape[3]
    xDim = featuresF['data'].shape[0]
    yDim = featuresF['data'].shape[1]
    #number of trainings labels in total
    n_labels_train = (n_folds-1)*n_labels_per_fold
    #number of trainings labels in stage_i
    n_labels_train_stage = 0
    #split trainingsdata wrt stages, split contains start_idx and length of trainingsdata in stage_i
    split = split_equ(stage_i, n_stages, n_labels_train)
    n_labels_train_stage = split[1]
    #calculate range of split according to image indices
    start_img, start_idx, end_img, end_idx = split_img_idx(split, n_labels_per_fold)
    #trainingsset without rotation
    #print(start_img,start_idx, end_img, end_idxa)
    if rot == False:
        print('rot=False')
        X_train = np.zeros((n_labels_train_stage, n_features))
        y_train = np.zeros((n_labels_train_stage))
        for i in np.arange(start_img, end_img+1):
            features = np.zeros((xDim, yDim, n_features), dtype=np.float32)
            if i==start_img:
                #consider that trainingsimages are not consecutive because of testset
                if i>=testindex:
                    i += 1 
                #print('start_img:',i)
                if start_img==end_img:
                    x = coord[i, 0, start_idx:end_idx]
                    y = coord[i, 1, start_idx:end_idx]
                    features[:,:,:] = featuresF['data'][:,:,i,:]
                    X_train[:len(x),:] = features[x,y,:] 
                    y_train[:len(x)] = labels[i, start_idx:end_idx]
                else:
                    x = coord[i, 0, start_idx:]
                    y = coord[i, 1, start_idx:]
                    features[:,:,:] = featuresF['data'][:,:,i,:]
                    X_train[:len(x),:] = features[x,y,:] 
                    y_train[:len(x)] = labels[i, start_idx:]
            elif i==end_img:
                if i>=testindex:
                    i += 1 
                #print('end_img:',i)
                x = coord[i, 0, :end_idx]
                y = coord[i, 1, :end_idx] 
                features[:,:,:] = featuresF['data'][:,:,i,:]
                X_train[n_labels_train_stage-len(x):,:] = features[x,y,:] 
                y_train[n_labels_train_stage-len(x):] = labels[i, :end_idx]
            else:
                #helper index
                j = 0
                if i>=testindex:
                    j = i+1
                #print(j) 
                x = coord[j, 0, :]
                y = coord[j, 1, :] 
                start_range = n_labels_per_fold-start_idx+(i-start_img-1)*n_labels_per_fold
                end_range = start_range+n_labels_per_fold
                features[:,:,:] = featuresF['data'][:,:,j,:]
                X_train[start_range:end_range,:] = features[x,y,:] 
                y_train[start_range:end_range] = labels[j, :]
        featuresF.close()
        return (X_train, y_train)
    #trainingsset with rotation
    else:
        #size of a trainingsset not known, take maximal possible size and cut it later
        X_train_big = np.zeros((int(n_labels_train_stage*1.1)*n_rot, n_features))
        y_train_big = np.zeros((int(n_labels_train_stage*1.1)*n_rot))
        insert_idx = 0
        for i in np.arange(start_img, end_img+1):
            if i==start_img:
                #consider that trainingsimages are not consecutive because of testset
                if i>=testindex:
                    i += 1 
                print('start_img:',i)
                if start_img==end_img:
                    x = coord[i, 0, start_idx:end_idx]
                    y = coord[i, 1, start_idx:end_idx]
                    selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                    selected_map[x,y]=labels[i, start_idx:end_idx] 
                    X_train_rot, y_train_rot = rotate_features_labels(i, selected_map, feature_filename, n_rot, y_center, x_center)
                    X_train_big[:len(y_train_rot),:] = X_train_rot[:,:]
                    y_train_big[:len(y_train_rot)] = y_train_rot[:]
                    insert_idx += len(y_train_rot)
                else:
                    x = coord[i, 0, start_idx:]
                    y = coord[i, 1, start_idx:]
                    selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                    selected_map[x,y]=labels[i, start_idx:] 
                    X_train_rot, y_train_rot = rotate_features_labels(i, selected_map, feature_filename, n_rot, y_center, x_center)
                    X_train_big[insert_idx:insert_idx+len(y_train_rot),:] = X_train_rot[:,:]
                    y_train_big[insert_idx:insert_idx+len(y_train_rot)] = y_train_rot[:]
                    insert_idx += len(y_train_rot)
            elif i==end_img:
                if i>=testindex:
                    i += 1 
                print('end_img:',i)
                x = coord[i, 0, :end_idx]
                y = coord[i, 1, :end_idx] 
                selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                selected_map[x,y]=labels[i, :end_idx] 
                X_train_rot, y_train_rot = rotate_features_labels(i, selected_map, feature_filename, n_rot, y_center, x_center)
                X_train_big[insert_idx:insert_idx+len(y_train_rot),:] = X_train_rot[:,:]
                y_train_big[insert_idx:insert_idx+len(y_train_rot)] = y_train_rot[:]
                insert_idx += len(y_train_rot)           
            else:
                if i>=testindex:
                    i += 1
                print('img:',i)
                x = coord[i, 0, :]
                y = coord[i, 1, :] 
                selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                selected_map[x,y]=labels[i,:] 
                X_train_rot, y_train_rot = rotate_features_labels(i, selected_map, feature_filename, n_rot, y_center, x_center)
                X_train_big[insert_idx:insert_idx+len(y_train_rot),:] = X_train_rot[:,:]
                y_train_big[insert_idx:insert_idx+len(y_train_rot)] = y_train_rot[:]
                insert_idx += len(y_train_rot)  
        X_train = np.zeros((insert_idx, n_features))
        y_train = np.zeros((insert_idx))
        X_train[:,:] = X_train_big[:insert_idx,:]
        y_train[:] = y_train_big[:insert_idx] 
        featuresF.close()
        return (X_train, y_train)

"""
helper-fct of train_sets_stages_several_files: only difference to train_sets_stages with feature_filenames several filenames are given as list
given a map of choosen labels (selected_map) of an unrotated image (with index img) and a feature map of rotated versions of this image the function produces trainingsdata and labels for rotated versions by selecting the matching rotated features and with its labels 
"""
def rotate_features_labels_several_files(img, selected_map, feature_filenames, n_rot, y_center, x_center):
#img: index of img the selected_map correspond to
#selected_map: array with shape of image with 0 for no labeling and 1,2,.. for labeled with class 1,2,..
#feature_filenames: list of feature filenames of hdf5 files containing dataset 'data' with feature_array [x,y,imgRot,f] 
#n_rot: number of rotations per image
#x_center, y_center: coordinates of center for rotation
    
    #open feature files
    featuresF = []
    for feature_filename in feature_filenames:
        featuresF.append(h5py.File(feature_filename,'r'))
    
    #parameters
    xDim = featuresF[0]['data'].shape[0]
    yDim = featuresF[0]['data'].shape[1]
    n_features = 0
    for k in np.arange(len(featuresF)):
        n_features += featuresF[k]['data'].shape[3]
    #create array for features of one image
    features = np.zeros((xDim, yDim,n_features), dtype=np.float32)
    #estimate maximal size of trainingsset
    x, y = np.nonzero(selected_map)
    X_train_big = np.zeros((int(len(x)*1.1)*n_rot, n_features))
    y_train_big = np.zeros((int(len(x)*1.1)*n_rot))
    insert_idx = 0
    angles = np.linspace(0., 360., num=n_rot, endpoint=False )
    for j,angle in enumerate(angles):
    #order=0 for preserving ones, no interpolation
        selected_map_rot = rotate(selected_map, angle, resize = False, center = (y_center, x_center), order =0, preserve_range=True)  
        selected_map_rot = selected_map_rot.astype(np.uint8)
        x_rot, y_rot = np.nonzero(selected_map_rot)
        insert_idx_files = 0
        for k in np.arange(len(featuresF)):
            n_added_feat = featuresF[k]['data'].shape[3]
            features[:,:,insert_idx_files:insert_idx_files+n_added_feat] = featuresF[k]['data'][:,:,img*n_rot+j,:]
            insert_idx_files += n_added_feat
        X_train_big[insert_idx:insert_idx+len(x_rot),:] = features[x_rot, y_rot, :]
        y_train_big[insert_idx:insert_idx+len(x_rot)] = selected_map_rot[x_rot,y_rot]
        insert_idx += len(x_rot)
    X_train = np.zeros((insert_idx, n_features))
    y_train = np.zeros((insert_idx))
    X_train = X_train_big[:insert_idx, :]
    y_train = y_train_big[:insert_idx]
    for k in np.arange(len(featuresF)):
        featuresF[k].close()
    return (X_train, y_train)




"""
produce trainings feature and label array for stage i of n_stages in total
"""
def train_sets_stages_several_files(stage_i, n_stages, labels, coord, feature_filenames, testindex, rot=False, n_rot=1, x_center=0, y_center=0):
#returns tuple of training features and labels  [X_train, y_train] X_train [nPixelTrain, nFeatures] y_train [nPixelTrain] 
#stage_i: uint between 0 and n_stages-1 indicating for wich stage trainingsdata are used
#n_stages: n of stages in total 
#labels: labels [img, selected_pixels] c: class labels c=0 no label
#coord: [img, x/y, selected_pixels] coordinates of labels
#feature_filenames: list of filenames of hdf5 files containing dataset 'data' with feature_array [x,y,img/imgRot,f] img/imgRot depend on rot, if rot is true imgRot 
#testindex: between 0 and 9 indicate fold used for testing all others are used for training
#n_rot: number of rotations per image
#rot: if rotated images are used rot=True
#x_center, y_center: coordinates of center for rotation
    
    #open feature file
    featuresF = []
    for feature_filename in feature_filenames:
        featuresF.append(h5py.File(feature_filename,'r'))

    #global parameters
    n_folds = int(labels.shape[0])
    n_labels_per_fold = labels.shape[1] 
    n_features = 0
    for k in np.arange(len(featuresF)):
        n_features += featuresF[k]['data'].shape[3]
    xDim = featuresF[0]['data'].shape[0]
    yDim = featuresF[0]['data'].shape[1]
    #number of trainings labels in total
    n_labels_train = (n_folds-1)*n_labels_per_fold
    #number of trainings labels in stage_i
    n_labels_train_stage = 0
    #split trainingsdata wrt stages, split contains start_idx and length of trainingsdata in stage_i
    split = split_equ(stage_i, n_stages, n_labels_train)
    n_labels_train_stage = split[1]
    #calculate range of split according to image indices
    start_img, start_idx, end_img, end_idx = split_img_idx(split, n_labels_per_fold)
    #trainingsset without rotation
    #print(start_img,start_idx, end_img, end_idxa)
    if rot == False:
        print('rot=False')
        X_train = np.zeros((n_labels_train_stage, n_features))
        y_train = np.zeros((n_labels_train_stage))
        for i in np.arange(start_img, end_img+1):
            features = np.zeros((xDim, yDim, n_features), dtype=np.float32)
            if i==start_img:
                #consider that trainingsimages are not consecutive because of testset
                if i>=testindex:
                    i += 1 
                print('start_img:',i)
                if start_img==end_img:
                    x = coord[i, 0, start_idx:end_idx]
                    y = coord[i, 1, start_idx:end_idx]
                    insert_idx = 0
                    for k in np.arange(len(featuresF)):
                        n_added_feat = featuresF[k]['data'].shape[3]
                        features[:,:,insert_idx:insert_idx+n_added_feat] = featuresF[k]['data'][:,:,i,:]
                        insert_idx += n_added_feat
                    X_train[:len(x),:] = features[x,y,:] 
                    y_train[:len(x)] = labels[i, start_idx:end_idx]
                else:
                    x = coord[i, 0, start_idx:]
                    y = coord[i, 1, start_idx:]
                    insert_idx = 0
                    for k in np.arange(len(featuresF)):
                        n_added_feat = featuresF[k]['data'].shape[3]
                        features[:,:,insert_idx:insert_idx+n_added_feat] = featuresF[k]['data'][:,:,i,:]
                        insert_idx += n_added_feat
                    X_train[:len(x),:] = features[x,y,:] 
                    y_train[:len(x)] = labels[i, start_idx:]
            elif i==end_img:
                if i>=testindex:
                    i += 1 
                print('end_img:',i)
                x = coord[i, 0, :end_idx]
                y = coord[i, 1, :end_idx] 
                insert_idx = 0
                for k in np.arange(len(featuresF)):
                    n_added_feat = featuresF[k]['data'].shape[3]
                    features[:,:,insert_idx:insert_idx+n_added_feat] = featuresF[k]['data'][:,:,i,:]
                    insert_idx += n_added_feat
                X_train[n_labels_train_stage-len(x):,:] = features[x,y,:] 
                y_train[n_labels_train_stage-len(x):] = labels[i, :end_idx]
            else:
                #helper index
                j = 0
                j = i
                if i>=testindex:
                    j = i+1
                print(j) 
                x = coord[j, 0, :]
                y = coord[j, 1, :] 
                start_range = n_labels_per_fold-start_idx+(i-start_img-1)*n_labels_per_fold
                end_range = start_range+n_labels_per_fold
                insert_idx = 0
                for k in np.arange(len(featuresF)):
                    n_added_feat = featuresF[k]['data'].shape[3]
                    features[:,:,insert_idx:insert_idx+n_added_feat] = featuresF[k]['data'][:,:,j,:]
                    insert_idx += n_added_feat
                X_train[start_range:end_range,:] = features[x,y,:] 
                y_train[start_range:end_range] = labels[j, :]
        for k in np.arange(len(featuresF)):
            featuresF[k].close()
        return (X_train, y_train)
    #trainingsset with rotation
    else:
        #size of a trainingsset not known, take maximal possible size and cut it later
        X_train_big = np.zeros((int(n_labels_train_stage*1.1)*n_rot, n_features))
        y_train_big = np.zeros((int(n_labels_train_stage*1.1)*n_rot))
        insert_idx = 0
        for i in np.arange(start_img, end_img+1):
            if i==start_img:
                #consider that trainingsimages are not consecutive because of testset
                if i>=testindex:
                    i += 1 
                print('start_img:',i)
                if start_img==end_img:
                    x = coord[i, 0, start_idx:end_idx]
                    y = coord[i, 1, start_idx:end_idx]
                    selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                    selected_map[x,y]=labels[i, start_idx:end_idx] 
                    X_train_rot, y_train_rot = rotate_features_labels_several_files(i, selected_map, feature_filenames, n_rot, y_center, x_center)
                    X_train_big[:len(y_train_rot),:] = X_train_rot[:,:]
                    y_train_big[:len(y_train_rot)] = y_train_rot[:]
                    insert_idx += len(y_train_rot)
                else:
                    x = coord[i, 0, start_idx:]
                    y = coord[i, 1, start_idx:]
                    selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                    selected_map[x,y]=labels[i, start_idx:] 
                    X_train_rot, y_train_rot = rotate_features_labels_several_files(i, selected_map, feature_filenames, n_rot, y_center, x_center)
                    X_train_big[insert_idx:insert_idx+len(y_train_rot),:] = X_train_rot[:,:]
                    y_train_big[insert_idx:insert_idx+len(y_train_rot)] = y_train_rot[:]
                    insert_idx += len(y_train_rot)
            elif i==end_img:
                if i>=testindex:
                    i += 1 
                print('end_img:',i)
                x = coord[i, 0, :end_idx]
                y = coord[i, 1, :end_idx] 
                selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                selected_map[x,y]=labels[i, :end_idx] 
                X_train_rot, y_train_rot = rotate_features_labels_several_files(i, selected_map, feature_filenames, n_rot, y_center, x_center)
                X_train_big[insert_idx:insert_idx+len(y_train_rot),:] = X_train_rot[:,:]
                y_train_big[insert_idx:insert_idx+len(y_train_rot)] = y_train_rot[:]
                insert_idx += len(y_train_rot)           
            else:
                if i>=testindex:
                    i += 1
                print('img:',i)
                x = coord[i, 0, :]
                y = coord[i, 1, :] 
                selected_map = np.zeros((xDim, yDim), dtype=np.uint8)  
                selected_map[x,y]=labels[i,:] 
                X_train_rot, y_train_rot = rotate_features_labels_several_files(i, selected_map, feature_filenames, n_rot, y_center, x_center)
                X_train_big[insert_idx:insert_idx+len(y_train_rot),:] = X_train_rot[:,:]
                y_train_big[insert_idx:insert_idx+len(y_train_rot)] = y_train_rot[:]
                insert_idx += len(y_train_rot)  
        X_train = np.zeros((insert_idx, n_features))
        y_train = np.zeros((insert_idx))
        X_train[:,:] = X_train_big[:insert_idx,:]
        y_train[:] = y_train_big[:insert_idx] 
        for i in np.arange(len(featuresF)):
            featuresF[i].close()
        return (X_train, y_train)




"""
returns train- and labelset for testing
"""
def test_sets(labels, coord, feature_filename, testindex, rot=False):# n_rot=1, x_center=0, y_center=0):
#returns (X_test, y_test)
#labels: labels [img, selected_pixels] c: class labels c=0 no label
#coord: [img, x/y, selected_pixels] coordinates of labels
#feature_filename: filename of hdf5 file with dataset 'data' containing feature_array [x,y,img/imgRot,f] 
#testindex: between 0 and k indicate fold used for testing all others are used for training
#rot: True if featuresarray with imgRot is used
    
    #open feature file
    featuresF = h5py.File(feature_filename,'r')
    #global parameters
    n_labels_per_fold = labels.shape[1] 
    n_features = featuresF['data'].shape[3]
    xDim = featuresF['data'].shape[0]
    yDim = featuresF['data'].shape[1]
    
    features = np.zeros((xDim, yDim, n_features), dtype=np.float32)
    
    x = coord[testindex,0, :]
    y = coord[testindex,1, :]

    X_test = np.zeros((n_labels_per_fold, n_features))
    y_test = np.zeros((n_labels_per_fold))
    if rot==False:
        features[:,:,:] = featuresF['data'][:,:,testindex,:]
        X_test[:, :] = features[x,y,:]
    else:
        features[:,:,:] = featuresF['data'][:,:,testindex*n_rot,:]
        X_test[:, :] = features[x,y,:]
    y_test[:] = labels[testindex,:] 
    featuresF.close()
    return (X_test, y_test)

"""
returns train- and labelset for all pixels within complete interesting(circle without edge) region for testing
"""
def test_sets_all(labels_all, x_center, y_center, radius, feature_filename, testindex, rot=False):# n_rot=1, x_center=0, y_center=0):
#returns (X_test, y_test)
#labels_all: labels [img, x, y] c: class labels c=0 no label
# x_center, y_center, radius: parameters for interesting region trainingspxls are used from
#feature_filename: filename of hdf5 file with dataset 'data' containing feature_array [x,y,img/imgRot,f] 
#testindex: between 0 and k indicate fold used for testing all others are used for training
#rot: True if featuresarray with imgRot is used
    
    #open feature file
    featuresF = h5py.File(feature_filename,'r')

    #global parameters
    n_features = featuresF['data'].shape[3]
    xDim = featuresF['data'].shape[0]
    yDim = featuresF['data'].shape[1]
    
    #array for features of one frame
    features = np.zeros((xDim, yDim, n_features), dtype=np.float32)

    x, y = circle(y_center, x_center, radius)

    X_test = np.zeros((len(x), n_features))
    y_test = np.zeros((len(x),))
    if rot==False:
        features[:,:,:] = featuresF['data'][:,:,testindex,:]
        X_test[:, :] = features[x,y,:]
    else:
        features[:,:,:] = featuresF['data'][:,:,testindex*n_rot,:]
        X_test[:, :] = features[x,y,:]
    y_test[:] = labels_all[testindex, x, y] 
    featuresF.close()
    return (X_test, y_test)

"""
considers only values of array y2D [xDim,yDim] in interesting circle region and tranform it into 1D array
"""
def y2D_to_y1D_in_circle(y2D, x_center, y_center, radius):
#returns: y1D, see description above
#y2D: image with labels [xDim,yDim]
# x_center, y_center, radius: parameters for interesting region trainingspxls are used from
    x, y = circle(y_center, x_center, radius)
    l = len(x)
    y1D = np.zeros((l,), dtype=np.int32)
    y1D = y2D[x,y]
    return y1D


"""
helper fct for create create_probability_maps, return indices of images, probabilities need to computed for training classifier
"""
def img_split(i_stage, n_stages, testindex, n_folds, labels, rot=False, n_rot=1):
#return array with indices of images on which probabilitymaps will be computed on later
#i_stage: stage involved pixels are computed for
#n_stages: number of stages in total
#testindex: index of image used for testing
#n_folds: number of folds
#labels: labels [img, selected_pixels] c: class labels c=0 no label
#rot: if data augmentation takes place


    n_labels_per_fold = labels.shape[1] 
    #number of trainings labels in total
    n_labels_train = (n_folds-1)*n_labels_per_fold
    #split trainingsdata wrt stages, split contains start_idx and length of trainingsdata in stage_i
    split = split_equ(i_stage, n_stages, n_labels_train)
    #calculate range of split according to image indices
    start_img, start_idx, end_img, end_idx = split_img_idx(split, n_labels_per_fold)
    #number of indices for range
    range_size = end_img+1-start_img
    if rot==True:
        range_size *= n_rot
    range_idx = np.zeros((range_size,), dtype=np.int32)
    for i, idx in enumerate(range(start_img, end_img+1)):
        if idx>=testindex:
            idx +=1 
        if rot==False:
            range_idx[i] = idx
        else:
            range_idx[i*n_rot:(i+1)*n_rot] = range(idx*n_rot, (idx+1)*n_rot)
    return range_idx


"""
create file with probability maps of RF
"""
def create_probability_maps(i_stage, clf, n_images, images, n_classes, rot=False):
#create file with name: outputfilename: 'probability_map_stage_'+str(i_stage)+'.h5' with dataset 'data' containing probability maps of first n_images in array [img, x,y,c]
#uses hdf5 file selected_std_feat_rawdata_rotated_stage_0.h5 with dataset 'data' containing array of shape [xDim,yDim,n_folds*n_rot,n_feat]
#i_stage: stage of RF for calculating prob maps for naming of output file 
#clf: RF of i_stage
#n_images: number of images probability maps are computed on (=n_folds or n_folds*n_rot) 
#n_classes: number of different classes t
#rot: true means data augmentation through rotations

    #select augmented data if rot=true
    filename_rawdata = ''
    if rot==True:
        filename_rawdata = 'selected_std_feat_rawdata_rotated_stage_0.h5'
    else:
        filename_rawdata = 'selected_std_feat_rawdata_stage_0.h5'
    f = h5py.File(filename_rawdata,'r')
    xDim = f['data'].shape[0]
    yDim = f['data'].shape[1]
    n_features = f['data'].shape[3]
    std_features = np.zeros((xDim, yDim, n_features), dtype=np.float32)
    filename = 'probability_map_stage_'+str(i_stage)+'.h5'
    w = h5py.File(filename, 'w')
    w.create_dataset('data', (n_images, xDim, yDim, n_classes), dtype=np.float32, compression='gzip')
    if i_stage==0:
        for img in images:
            std_features[:,:,:] = f['data'][:,:,img,:]
            #shape of prob1D [xDim*yDim, n_classes]
            prob1D = clf.predict_proba(std_features.reshape((xDim*yDim,std_features.shape[2])))
            prob2D = prob1D.reshape((xDim, yDim,n_classes ))
            w['data'][img,:,:,:] = prob2D  
    else:
        addf = h5py.File('selected_std_feat_prob_stage_'+str(i_stage)+'.h5','r')
        n_feat_add = addf['data'].shape[3]
        add_features = np.zeros((xDim, yDim, n_feat_add), dtype=np.float32)
        for img in images:
            std_features[:,:,:] = f['data'][:,:,img,:]
            add_features[:,:,:] = addf['data'][:,:,img,:]
            all_features = np.zeros((xDim, yDim, std_features.shape[2]+add_features.shape[2]), dtype=np.float32)
            all_features[:,:,:std_features.shape[2]]= std_features[:,:,:]
            all_features[:,:,std_features.shape[2]:]= add_features[:,:,:]
            #shape of prob1D [xDim*yDim, n_classes]
            prob1D = clf.predict_proba(all_features.reshape((xDim*yDim,all_features.shape[2])))
            prob2D = prob1D.reshape((xDim, yDim, n_classes))
            w['data'][img,:,:,:] = prob2D 
        addf.close() 
    w.close()
    f.close()

"""
helper fct for predict_one_image fct, calculate selected convolutions without using hdf5 files, implemented for prediction fct
"""
def calc_selected_std_features_one_img(data,sigmasGaussian, sigmasLoG=[],sigmasGGM=[],sigmasSTE=[],sigmasHoGE=[]):
#return type feature_array [x,y,t,f]
#rawdata [x,y,c] 
#sigmasGaussian: Gaussian Smoothing
#sigmasLoG: Laplace of Gaussian
#sigmasGGM: Gaussian Gradient Magnitude
#sigmasSTE: Structure Tensor Eigenvalues
#sigmasHoGE: Hessian of Gaussian Eigenvalues

    rawdata = vg.taggedView(data.astype(np.float32), "xyc")

    #global parameters
    xDim = rawdata.shape[0]
    yDim = rawdata.shape[1]
    cDim = rawdata.shape[2] #c=0 for image data, c=nClassses for prob maps 
    nFeat_per_Class = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)
    nFeatures = cDim*nFeat_per_Class

    features = np.zeros(( xDim, yDim, nFeatures))
    features.astype(np.float32)

    for c in np.arange(cDim):
        for i,sigma in enumerate(sigmasGaussian):
            features[:,:,i+c*nFeat_per_Class] = vg.filters.gaussianSmoothing(rawdata[:,:,c], sigma)
        offset=len(sigmasGaussian)
        for i,sigma in enumerate(sigmasLoG):
            features[:,:,i+offset+c*nFeat_per_Class] = vg.filters.laplacianOfGaussian(rawdata[:,:,c], sigma)
        offset=len(sigmasGaussian)+len(sigmasLoG)
        for i,sigma in enumerate(sigmasGGM):
            features[:,:,i+offset+c*nFeat_per_Class] = vg.filters.gaussianGradientMagnitude(rawdata[:,:,c], sigma)
        offset=len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)
        for i,sigma in enumerate(sigmasSTE):
            data = vg.filters.structureTensorEigenvalues(rawdata[:,:,c], sigma, sigma )
            features[:,:,2*i+offset+c*nFeat_per_Class] = data[:,:,0]
            features[:,:,2*i+1+offset+c*nFeat_per_Class] = data[:,:,1]
        offset=len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+len(sigmasSTE)*2
        for i,sigma in enumerate(sigmasHoGE):
            data = vg.filters.hessianOfGaussianEigenvalues(rawdata[:,:,c], sigma)
            features[:,:,2*i+offset+c*nFeat_per_Class] = data[:,:,0]
            features[:,:,2*i+1+offset+c*nFeat_per_Class] = data[:,:,1]

    return features




"""
predict image with index img with given clf_list
"""
def predict_one_image(img, clf, n_classes, sigmasGaussian_prob, sigmasLoG_prob=[],sigmasGGM_prob=[],sigmasSTE_prob=[],sigmasHoGE_prob=[]):
#returns y2D_pred array of shape [xDim,yDim] of type np.int32
#img: index for frame that is predicted
#clf: list of clf for different stages

    f = h5py.File('selected_std_feat_rawdata_stage_0.h5','r')
    #parameters
    xDim = f['data'].shape[0]
    yDim = f['data'].shape[1]
    n_stages = len(clf)
    #determine number of features out of prob maps 
    n_feat_per_class = len(sigmasGaussian_prob)+len(sigmasLoG_prob)+len(sigmasGGM_prob)+2*len(sigmasSTE_prob)+2*len(sigmasHoGE_prob)
    n_add_features = n_classes*n_feat_per_class
    #allocate arrays
    y2D_pred = np.zeros((xDim, yDim), dtype=np.int32)            
    features = np.zeros((xDim,yDim,f['data'].shape[3]), dtype=np.float32)
    features[:,:,:] = f['data'][:,:,img,:]
    X_pred = features.reshape((xDim*yDim,f['data'].shape[3]))
    if n_stages==1:
        y2D_pred = clf[0].predict(X_pred).reshape((xDim, yDim))
        f.close()
        return y2D_pred
    #feature array for features out of rawdata and prob maps
    all_features = np.zeros((xDim,yDim,f['data'].shape[3]+n_add_features), dtype=np.float32)
    all_features[:,:,:f['data'].shape[3]] = features[:,:,:]
    for i_stage in np.arange(n_stages-1):
        i_stage +=1
        if i_stage-1==0:
            prob1D = clf[0].predict_proba(X_pred)
            prob_map = prob1D.reshape((xDim, yDim, n_classes))
            all_features[:,:,f['data'].shape[3]:]=calc_selected_std_features_one_img(prob_map, sigmasGaussian_prob, sigmasLoG_prob,sigmasGGM_prob, sigmasSTE_prob,sigmasHoGE_prob)     
            #if last classifier
            if i_stage==n_stages-1:
                X_pred_all = all_features.reshape((xDim*yDim,f['data'].shape[3]+n_add_features))
                y2D_pred = clf[i_stage].predict(X_pred_all).reshape((xDim, yDim))     
                f.close()
                return y2D_pred        
        else:
            X_pred_all = all_features.reshape((xDim*yDim,f['data'].shape[3]+n_add_features))
            prob1D = clf[i_stage-1].predict_proba(X_pred_all)
            prob_map = prob1D.reshape((xDim, yDim, n_classes))
            all_features[:,:,f['data'].shape[3]:]=calc_selected_std_features_one_img(prob_map, sigmasGaussian_prob, sigmasLoG_prob,sigmasGGM_prob, sigmasSTE_prob,sigmasHoGE_prob)     
            if i_stage==n_stages-1:
                X_pred_all = all_features.reshape((xDim*yDim,f['data'].shape[3]+n_add_features))
                y2D_pred = clf[i_stage].predict(X_pred_all).reshape((xDim, yDim))     
                f.close()
                return y2D_pred  
          

