# -*- coding: utf-8 -*-

#  packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf # designed with 2.1.0 /!\ models output changes with tf>2.2.0
import tensorflow.keras as keras

from lib import despawn


# Load a toy time series data to run DeSPAWN
signal = pd.read_csv("monthly-sunspots.csv")
lTrain = 2000 # length of the training section
signalT = ((signal['Sunspots']-signal['Sunspots'].mean())/signal['Sunspots'].std()).values[np.newaxis,:,np.newaxis,np.newaxis]
signal = signalT[:,:lTrain,:,:]

# Number of decomposition level is max log2 of input TS
level = np.floor(np.log2(signal.shape[1])).astype(int)
# Train hard thresholding (HT) coefficient?
trainHT = True
# Initialise HT value
initHT = 0.3
# Which loss to consider for wavelet coeffs ('l1' or None)
lossCoeff='l1'
# Weight for sparsity loss versus residual?
lossFactor = 1.0
# Train wavelets? (Trainable kernels)
kernTrainable = True
# Which training mode?
# cf (https://arxiv.org/pdf/2105.00899.pdf -- https://doi.org/10.1073/pnas.2106598119) [Section 4.4 Ablation Study]
#   CQF => learn wavelet 0 infer all other kernels from the network
#   PerLayer => learn one wavelet per level, infer others
#   PerFilter => learn wavelet + scaling function per level + infer other
#   Free => learn everything
mode = 'PerLayer' 

# Initialise wavelet kernel (here db-4)
kernelInit = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                           -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965])


epochs = 1000
verbose = 2

# Set sparsity (dummy) loss:
def coeffLoss(yTrue,yPred):
    return lossFactor*tf.reduce_mean(yPred,keepdims=True)
# Set residual loss:
def recLoss(yTrue,yPred):
    return tf.math.abs(yTrue-yPred)

keras.backend.clear_session()
# generates two models: 
#      model1 outputs the reconstructed signals and the loss on the wavelet coefficients
#      model2 outputs the reconstructed signals and wavelet coefficients
model1,model2 = despawn.createDeSpaWN(inputSize=None, kernelInit=kernelInit, kernTrainable=kernTrainable, level=level, lossCoeff=lossCoeff, kernelsConstraint=mode, initHT=initHT, trainHT=trainHT)
opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
# For the training we only use model1
model1.compile(optimizer=opt, loss=[recLoss, coeffLoss])
# the sparsity term has no ground truth => just input an empty numpy array as ground truth (anything would do, in coeffLoss, yTrue is not called)
H = model1.fit(signal,[signal,np.empty((signal.shape[0]))], epochs=epochs, verbose=verbose)


