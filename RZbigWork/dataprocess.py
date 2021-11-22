from matplotlib import markers
import pandas as pd
import numpy as np
import librosa
import glob
import os
import torch
from network2 import SoundClassifyNet2D
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def extract_mfcc(audio_data,n_mfcc):
    mfccs = librosa.feature.mfcc(audio_data,n_mfcc=n_mfcc,lifter=0) #(n_mfcc,t)
    return np.mean(mfccs.T,axis=0)
def extract_mfcc2D(audio_data,n_mfcc):
    mfccs = librosa.feature.mfcc(audio_data,n_mfcc=n_mfcc,lifter=0)
    return mfccs
def getdata(file):
    filter_size = 30000
    stride = 10000
    audio_data = file.values[:,0]
    data_sum = audio_data.shape[0]
    seg_num = int((data_sum - filter_size)/stride+1)
    data_seg = list()
    for i in range(seg_num):
        first_idx = stride*i
        last_idx = first_idx + filter_size
        data_seg.append(np.array(audio_data[first_idx:last_idx]))
        #extract mfcc feature
    feature = []
    for j in data_seg:
        mfccs_dim = 128
        mfccs = extract_mfcc(j,n_mfcc=mfccs_dim)
        feature.append(mfccs)
    feature = np.array(feature)
    feature = torch.tensor(feature)
    print("finish")
    return feature

def predict(data,model):
    data = data.float()
    data = data.reshape(1, 1, 13, 128)
    data = torch.cat((data, data), dim=0)
    res,out = model(data)
    out = torch.argmax(out, dim=1)
    if(out[0]==1):
        return res[0],1
    else:
        return res[0],0


def Ad():
    data = torch.load('../model/features_mfcc_float.pt')
    label = torch.load('../model/labels.pt')
    label = torch.unsqueeze(label,dim=1)
    model = SoundClassifyNet2D(2)
    data = torch.unsqueeze(data,dim=1)
    checkpoint1 = torch.load('../model/streamlit-64.pth', map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint1.items()})
    model.eval()
    res,out = model(data)
    out = torch.argmax(out,dim=1)
    np.save('64_feature',res.detach().numpy())
    np.save('64_label',out.detach().numpy())
    print('finish')
def get3dimpfea():
    data = torch.load('../model/3w.pt')
    label = torch.load('../model/labels.pt')
    label = torch.unsqueeze(label,dim=1)
    all = torch.cat((data,label),dim=1)
    print(data.shape,label.shape,all.shape)
    all = all.detach().numpy()
    np.save('4w.ny',all)
    print("finish")
def sigmoid(data):
    data = (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    return 1 / (1 + np.exp(-data))
def color(co):
    c = []
    for i in range(len(co)):
        if co[i]==1:
            c.append('g')
        else:
            c.append('b')
    return c
def display(fig,ax):
    data = np.load('64_feature.npy')
    label = np.load("64_label.npy")
#    min, max = np.min(data), np.max(data)
#    data = (max - data) / (max - min)
#    data = sigmoid(data)
#    fig = plt.figure()
#   ax = Axes3D(fig)
#    print(data)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(data[:,8],data[:,0],data[:,11],c=color(label),s=10,alpha=0.5)
    plt.show()
def display1():
    data = np.load("4w.npy")
    fig = plt.figure(12,8)
    ax1 = Axes3D(fig)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    ax1.set_xlim([min(x), max(x)])
    ax1.set_ylim([min(y), max(y)])
    ax1.set_zlim([min(z), max(z)])
    ax1.scatter(x,y,z)