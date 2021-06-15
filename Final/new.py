from matplotlib import markers
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from dataprocess import getdata
from network2 import SoundClassifyNet2D
from dataprocess import predict,display
from train2D import start
import os
import pickle
import base64
def get_model():
    model = SoundClassifyNet2D(2)
    checkpoint1 = torch.load('streamlit-64.pth', map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint1.items()})
    return model
def display_app():
    st.sidebar.markdown('### 第一步：选择样例文件或者上传本地文件(csv)')
    choice = st.sidebar.selectbox("选择文件",("样例一","样例二","本地文件"),key="display")
    if choice=="本地文件":
        file = st.sidebar.file_uploader("选择本地文件")
    elif choice=="样例一":
        file = pd.read_csv("good.csv",header=None)
    else:
        file = pd.read_csv("bad.csv",header=None)
    st.sidebar.markdown("### 第二步：点击查看结果")
    left,mid,right = st.sidebar.beta_columns(3)
    st.header("马达声音测试")
    st.subheader("使用方法")
    '''
    - 第一步：选择本地文件
    - 第二步：点击诊断查看结果
    '''
    model = get_model()
    model.eval()
    if mid.button("诊断"):
        if choice=="本地文件":
            csv_data = pd.read_csv(file, header=None)
            show(csv_data,model)
        else:
            show(file,model)
def show(csv_data,model):
    '''
    原始数据波形图
    '''
    st.subheader("原始波形图")
    fig , ax = plt.subplots()
    ax.plot(csv_data)
    st.pyplot(fig)
    mfcc_data = getdata(csv_data)
    st.subheader("mfcc")
    mfcc_da = np.array(mfcc_data)
    st.line_chart(mfcc_da.T)
    fig = plt.figure()
    plt.imshow(mfcc_data, cmap=plt.cm.jet, extent=[0, mfcc_data.shape[1], 0, mfcc_data.shape[0]], aspect='auto')
    st.pyplot(fig)
    res,Bool = predict(mfcc_data,model)
    res = res.detach().numpy()
    st.subheader("三维特征图")
    fig = plt.figure()
    ax = Axes3D(fig)
    display(fig,ax)
    ax.scatter(res[8],res[0],res[11],marker='v',c='r',s=50)
    st.pyplot(fig)
    st.header("诊断结果")
    if Bool:
        st.success("正常")
    else:
        st.error("故障")
def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="mymodel.pth">Download Trained Model .pth File</a>'
    st.markdown(href, unsafe_allow_html=True)
def process_data():
    st.sidebar.markdown("请选择数据集所在的文件夹")
def Train():
    st.sidebar.markdown("### 第一步：请上传训练数据")
    mfcc=st.sidebar.file_uploader("请上传训练集(N,C,W,H)-(.pt)")
    label = st.sidebar.file_uploader("请上传labels(.pt)")
    model = SoundClassifyNet2D(2)
    st.sidebar.markdown("### 第二步：训练模型")
    left,mid,right = st.sidebar.beta_columns(3)
    st.subheader("简单配置")
    col1,col2,col3 = st.beta_columns(3)
    with col1:
        epoch = st.number_input("epoch",min_value=1,max_value=1000,value=80)
    with col2:
        lr = st.number_input("learning rate(e10-4)",min_value=1,max_value=1000,value=100)
        lr = lr*0.0001
    with col3:
        option = st.selectbox("请选择优化方法",("Adam",'SGD','RMSprop'))
    tag = 0
    if mid.button("训练"):        
        data = torch.load(mfcc)
        val = torch.load(label)
        modeldata = start(epoch,lr,option,data,val)
        st.subheader("训练结束")
        st.write("点击下方链接下载模型")
        download_model(model)


def main():
    '''
    st.sidebar.markdown("# 请选择需求")
    Choose = st.sidebar.selectbox("选择步骤",['数据处理','模型训练','效果展示'],key='mian')
    if Choose == '效果展示':
        display_app()
    elif Choose == '数据处理':
        process_data()
    else:
        Train()
    '''
    display_app()
main()