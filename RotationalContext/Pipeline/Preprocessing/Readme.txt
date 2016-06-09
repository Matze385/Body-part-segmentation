rotateImage.py:

input:
-rawdata: FlyBowlMovie.h5

output:
-selected images: trainingsImages.h5
-rotated selected images for data augmentation: trainingsImagesRot.h5


crossValSets.py

need functions from:
-sampling.py 

input:
-rotated images: trainingsImagesRot.h5
-labeled selected images: 

output:
-crossValFolds.h5
