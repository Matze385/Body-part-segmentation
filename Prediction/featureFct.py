import numpy as np
import vigra as vg
from vigra import blockwise as bw
import h5py 
import matplotlib.pyplot as plt

"""
calculate specified convolutions as features on given images of first n_images of data, used later for calculation of features on probability maps to calculate only needed features
"""
def calc_selected_std_features(data_filename, output_filename, n_images, images, sigmasGaussian, sigmasLoG=[],sigmasGGM=[],sigmasSTE=[],sigmasHoGE=[]):
#return type: none but produces file with name output_filename containing feature_array [x,y,t,c, type_feature, sigmas]
#data_filename: name of file with dataset 'data' in shape [t,x,y,c] to calculate features on
#output_filename: name of file with dataset 'data' containing feature_array [x,y,t,f]
#n_images: compute convolutions on first n_images of data_filename
#images: array of img indices features are calculated on
#sigmasGaussian: Gaussian Smoothing
#sigmasLoG: Laplace of Gaussian
#sigmasGGM: Gaussian Gradient Magnitude
#sigmasSTE: Structure Tensor Eigenvalues
#sigmasHoGE: Hessian of Gaussian Eigenvalues
    
    #open data file
    rawdataF = h5py.File(data_filename,'r')
    xDim = rawdataF['data'].shape[1]
    yDim = rawdataF['data'].shape[2]
    cDim = rawdataF['data'].shape[3]

    outF = h5py.File(output_filename ,'w')
    #number of features types: 7 because STE and HoGE consists of two eigenvalues
    nFeat_per_Class = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)
    nFeatures = cDim*nFeat_per_Class
    outF.create_dataset('data', (xDim, yDim, n_images, nFeatures), dtype=np.float32, compression='gzip')

    for t in images:
        data = np.zeros((xDim, yDim, cDim), dtype=np.float32)
        data = rawdataF['data'][t,:,:,:]
        #rawdata = rawdata.swapaxes(0,2).swapaxes(0,1)
        rawdata = vg.taggedView(data.astype(np.float32), "xyc")
        features = np.zeros((xDim, yDim,nFeatures), dtype=np.float32)

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
  	outF['data'][:,:,t,:] = features[:,:,:]
    outF.close()
    rawdataF.close()
    return features

"""
calculate convolutions of all std features on first n_images of data
"""
def calc_std_features(data_filename, output_filename, n_images, sigmas):
#return type: none but produces file with name output_filename containing feature_array [x,y,t,c, type_feature, sigmas]
#data_filename: name of file with dataset 'data' in shape [t,x,y,c] to calculate features on
#output_filename: name of file with dataset 'data' in shape [x,y,n_images, cDim, 7, len(sigmas)]
#n_images: compute convolutions on first n_images of data_filename
#sigmas: sigmas for following features: sigmasGaussian, sigmasLoG, sigmasGGM, sigmasSTE, sigmasHoGE
    
    #open data file
    rawdataF = h5py.File(data_filename,'r')
    xDim = rawdataF['data'].shape[1]
    yDim = rawdataF['data'].shape[2]
    cDim = rawdataF['data'].shape[3]

    outF = h5py.File(output_filename ,'w')
    #number of features types: 7 because STE and HoGE consists of two eigenvalues
    n_feat_types = 7
    outF.create_dataset("data", (xDim, yDim, n_images, cDim, n_feat_types, len(sigmas)), dtype=np.float32, compression='gzip')
    for t in np.arange(n_images):
        data = np.zeros((xDim, yDim, cDim), dtype=np.float32)
        data = rawdataF['data'][t,:,:,:]
        print data.shape
        #rawdata = rawdata.swapaxes(0,2).swapaxes(0,1)
        rawdata = vg.taggedView(data.astype(np.float32), "xyc")
        features = np.zeros((xDim, yDim,cDim, n_feat_types, len(sigmas)), dtype=np.float32)
        for c in np.arange(cDim):
            for i,sigma in enumerate(sigmas):
                features[:,:,c,0,i] = vg.filters.gaussianSmoothing(rawdata[:,:,c], sigma)
                features[:,:,c,1,i] = vg.filters.laplacianOfGaussian(rawdata[:,:,c], sigma)
                features[:,:,c,2,i] = vg.filters.gaussianGradientMagnitude(rawdata[:,:,c], sigma)
                data1 = vg.filters.structureTensorEigenvalues(rawdata[:,:,c], sigma, sigma )
                features[:,:,c,3,i] = data1[:,:,0]
                features[:,:,c,4,i] = data1[:,:,1]
                data2 = vg.filters.hessianOfGaussianEigenvalues(rawdata[:,:,c], sigma)
                features[:,:,c,5,i] = data2[:,:,0]
                features[:,:,c,6,i] = data2[:,:,1]
        outF['data'][:,:,t,:,:,:] = features[:,:,:,:,:]
    outF.close()
    rawdataF.close()
    return features



"""
select features out of all standard features out of given hdf5 file and write results again in hdf5 file 
"""
def select_std_features(i_stage, output_filename_extension, all_features_filename,  sigmasGaussian, sigmasLoG=[], sigmasGGM=[], sigmasSTE=[], sigmasHoGE=[]):
#i_stage: int number of current stage, for outout file name
#output_filename_extension: eg rawdata or prob output filename: 'selected_std_feat_'+output_filename_extension+'_stage_'+str(i_stage)+ '.h5' 
#all_features_filename: string of name of all std feature hdf5 file containing feature array [x,y,t,c,f,sigmas], f: all features, c=classes
#sigmas: selected sigmas, must be out of [0.3,1.0,1.6,3.5,5.0,10.0]
#produce array with selected feature array [x,y,t,sel_f] sel_f: selected features

    #test if sigmas are in correct range:
    sigmas = [0.3,1.0,1.6,3.5,5.0,10.0]
    for sigma in sigmasGaussian:
        #sigmas.index(sigma) throw an error if sigma is not in sigmas
        sigmas.index(sigma)
    for sigma in sigmasLoG:
        sigmas.index(sigma) 
    for sigma in sigmasGGM:
        sigmas.index(sigma)
    for sigma in sigmasSTE:
        sigmas.index(sigma)
    for sigma in sigmasHoGE:
        sigmas.index(sigma)
    
    #open all_features    
    all_features = h5py.File(all_features_filename,'r')

    #parameters
    xDim = all_features['data'].shape[0] 
    yDim = all_features['data'].shape[1]
    cDim = all_features['data'].shape[3]
    tDim = all_features['data'].shape[2]
    n_feat_type = all_features['data'].shape[4]
    n_sigmas = all_features['data'].shape[5]
    n_feat_per_class = len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+2*len(sigmasSTE)+2*len(sigmasHoGE)
    n_feat = cDim*n_feat_per_class

     
    #create hdf5 file with selected features
    filename = 'selected_std_feat_'+output_filename_extension+'_stage_'+str(i_stage)+ '.h5'
    f = h5py.File(filename,'w')
    f.create_dataset('data', (xDim,yDim,tDim,n_feat), dtype=np.float32, compression='gzip')
    #select features
    for t in np.arange(tDim):
        sel_feat = np.zeros((xDim,yDim,n_feat), dtype=np.float32)
        features = np.zeros((xDim,yDim,cDim,n_feat_type,n_sigmas), dtype=np.float32)
        features[:,:,:,:,:] = all_features['data'][:,:,t,:,:,:]
        for c in np.arange(cDim):
            for i,sigma in enumerate(sigmas):
                if sigma in sigmasGaussian:
                    sel_feat[:,:,sigmasGaussian.index(sigma)+c*n_feat_per_class] = features[:,:,c,0,i]
                offset=len(sigmasGaussian)  
                if sigma in sigmasLoG:
                    sel_feat[:,:,sigmasLoG.index(sigma)+c*n_feat_per_class+offset] = features[:,:,c,1,i]
                offset=len(sigmasGaussian)+len(sigmasLoG)  
                if sigma in sigmasGGM:
                    sel_feat[:,:,sigmasGGM.index(sigma)+c*n_feat_per_class+offset] = features[:,:,c,2,i]
                offset=len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)
                if sigma in sigmasSTE:
                    sel_feat[:,:,2*sigmasSTE.index(sigma)+c*n_feat_per_class+offset] = features[:,:,c,3,i]
                    sel_feat[:,:,2*sigmasSTE.index(sigma)+1+c*n_feat_per_class+offset] = features[:,:,c,4,i]
                offset=len(sigmasGaussian)+len(sigmasLoG)+len(sigmasGGM)+len(sigmasSTE)*2
                if sigma in sigmasHoGE:
                    sel_feat[:,:,2*sigmasHoGE.index(sigma)+c*n_feat_per_class+offset] = features[:,:,c,5,i]
                    sel_feat[:,:,2*sigmasHoGE.index(sigma)+1+c*n_feat_per_class+offset] = features[:,:,c,6,i]
        f['data'][:,:,t,:] = sel_feat[:,:,:]
    all_features.close()
    f.close()
















