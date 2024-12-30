#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# In[107]:


def readPoiSpikes(fileName, Fs):
    '''
    The fileName is the name of the file to be read,
    If the file is in the directory of the script, the path is not needed
    otherwise, the path needs to be full 
    '''

    data = loadmat(fileName)
    rawspike = np.array(data['spikes'])
    rawspike = rawspike.flatten()
    
    totalTime = int(np.max(rawspike))
    if rawspike.size == 0:
        raise ValueError("The file is empty")
    
    bins = np.linspace(0,totalTime,totalTime+1)
    spikeTrain,_= np.histogram(rawspike, bins=bins)
    spikeTrain = (spikeTrain>0).astype(int)
    return spikeTrain


# In[2]:


def generatePoiSpikes(r, dt, totalSize):

    rand = np.random.rand(totalSize)
    spikeTrain = np.zeros(totalSize)
    spikeTrain[rand<r*dt] = 1
    

    return spikeTrain


# In[3]:


def calcFF(spikeTrain):

    FF = np.var(spikeTrain)/np.mean(spikeTrain)
    
    return FF

def calcCV(spikeTrain):
    spike = np.where(spikeTrain==1)[0]
    isi = np.diff(spike)
    CV = np.std(isi)/np.mean(isi)
    
    return CV


# In[4]:


def calcRate(spikeTrain, window, dt):

    if window == 0:
        spike = np.where(spikeTrain==1)[0]
        r = np.mean(np.diff(spike))*dt ## in seconds
        return 1/r
    window_bins = int(window/dt)
    rateOfFire = []
    
    for i in range(0,len(spikeTrain),window_bins):
        window_c = np.sum(spikeTrain[i:i+window_bins])
        rate = (window_c/window) 
        rateOfFire.append(rate)
    return rateOfFire

