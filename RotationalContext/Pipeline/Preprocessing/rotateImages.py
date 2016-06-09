import numpy as np
import h5py
import vigra as vg
import skimage as skim

from skimage.transform import rotate 


N_ROT = 16 		#step number for rotations
N_IMG = 10 		#number of trainingsimages without rotations
yDim = 0
xDim = 0
cDim = 0
xCenter = 510		#center coordinates for rotation
yCenter = 514


with h5py.File('FlyBowlMovie.h5','r') as rawF:
    yDim = rawF['data'].shape[1]
    xDim = rawF['data'].shape[2]
    cDim = rawF['data'].shape[3]
    with h5py.File('trainingsImages.h5','w') as trainF: 
        trainF.create_dataset("data", (N_IMG, yDim, xDim, cDim) )
        trainF['data'][0,:,:,0] = rawF['data'][0,:,:,0]
        trainF['data'][1,:,:,0] = rawF['data'][239,:,:,0]
        trainF['data'][2,:,:,0] = rawF['data'][354,:,:,0]
        trainF['data'][3,:,:,0] = rawF['data'][407,:,:,0]
        trainF['data'][4,:,:,0] = rawF['data'][461,:,:,0]
        trainF['data'][5,:,:,0] = rawF['data'][517,:,:,0]
        trainF['data'][6,:,:,0] = rawF['data'][531,:,:,0]
        trainF['data'][7,:,:,0] = rawF['data'][556,:,:,0]
        trainF['data'][8,:,:,0] = rawF['data'][562,:,:,0]
        trainF['data'][9,:,:,0] = rawF['data'][1000,:,:,0]


with h5py.File('trainingsImages.h5','r') as trainF:
    with h5py.File('trainingsImagesRot.h5','w') as trainRotF:
        trainRotF.create_dataset("data", (N_ROT*N_IMG, yDim, xDim, cDim)) 
        for i in np.arange(N_IMG):
            cpImg = np.array( trainF['data'][i,:,:,0], copy=True, dtype=np.uint8)	
            #print type( cpImg[2,2])
            #cpImgFloat = skim.img_as_float(cpImg)
            angles = np.linspace(0., 360., num=N_ROT, endpoint=False )
            for j,angle in enumerate(angles):
                cpRot = rotate(cpImg, angle, resize = False, center = (yCenter,xCenter), mode = 'reflect' )
                trainRotF['data'][j+i*N_ROT,:,:,0] = skim.img_as_ubyte(cpRot ) #convert 01float image to 0255 image
        #print  np.all(trainF['data'][0,:,:,0]== trainRotF['data'][0,:,:,0])








