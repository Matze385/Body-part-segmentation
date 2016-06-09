#!/bin/bash

#array of number of trainingssize for the learningcurve
#trainsizes=(500)
trainsizes=(100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 5500 6000 7000 8000 9000 10000)
#trainsizes=(2000 2500 3000 3500 4000 4500 5000 5500 6000 7000 8000 9000 10000)

#create hdf5 file for storing learning results
python init_learning.py
for trainsize in ${trainsizes[*]}; do
    #remove old sampled trainingspixel
    rm CrossValFolds.h5
    cd Preprocessing
    #sample with special number of trainingspixel
    python crossValSets.py -n $trainsize
    cp CrossValFolds.h5 ../
    cd ..
    python 1_Stage.py
done
