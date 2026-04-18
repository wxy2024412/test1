# -*- coding: utf-8 -*-

# /!\ Designed for tensorflow 2.1.X
import tensorflow as tf
import tensorflow.keras as keras
from lib import despawnLayers as impLay

def createDeSpaWN(inputSize=None, kernelInit=8, kernTrainable=True, level=1, lossCoeff='l1', kernelsConstraint='QMF', initHT=1.0, trainHT=True):
    
    
    input_shape = (inputSize,1,1)
    inputSig = keras.layers.Input(shape=input_shape, name='input_Raw')
    g = inputSig
    if kernelsConstraint=='CQF':
        kern = impLay.Kernel(kernelInit, trainKern=kernTrainable)(g)
        kernelsG  = [kern for lev in range(level)]
        kernelsH  = kernelsG
        kernelsGT = kernelsG
        kernelsHT = kernelsG
    elif kernelsConstraint=='PerLayer':
        kernelsG  = [impLay.Kernel(kernelInit, trainKern=kernTrainable)(g) for lev in range(level)]
        kernelsH  = kernelsG
        kernelsGT = kernelsG
        kernelsHT = kernelsG
    hl = []
    inSizel = []
    # Decomposition
    for lev in range(level):
        inSizel.append(tf.shape(g))
        hl.append(impLay.HardThresholdAssym(init=initHT,trainBias=trainHT)(impLay.HighPassWave()([g,kernelsH[lev]])))
        g  = impLay.LowPassWave()([g,kernelsG[lev]])
    g = impLay.HardThresholdAssym(init=initHT,trainBias=trainHT)(g)
    # save intermediate coefficients to output them
    gint = g
    # Reconstruction
    for lev in range(level-1,-1,-1):
        h = impLay.HighPassTrans()([hl[lev],kernelsHT[lev],inSizel[lev]])
        g = impLay.LowPassTrans()([g,kernelsGT[lev],inSizel[lev]])
        g = keras.layers.Add()([g,h])
    
    # Compute specified loss on coefficients
    if not lossCoeff:
        vLossCoeff = tf.zeros((1,1,1,1))
    elif lossCoeff=='l1':
        # L1-Sum
        vLossCoeff = tf.math.reduce_mean(tf.math.abs(tf.concat([gint]+hl,axis=1)),axis=1,keepdims=True)
    else:
        raise ValueError('Could not understand value in \'lossCoeff\'. It should be either \'l1\' or \'None\'')
    return keras.models.Model(inputSig,[g,vLossCoeff]), keras.models.Model(inputSig,[g,gint,hl[::-1]])
    
