## Others lib
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import timedelta

## ML lib
# import tensorflow as tf
# import keras.backend as K

## Env setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config) 
# K.set_session(sess)

date = '20200527_0000'
# save_path = 'CSI_PICTURE/Compare_{}/'.format(date)
save_path = 'Compare_{}/'.format(date)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

# csi_predrnn_loss_atten = np.loadtxt(open(save_path+"202005270000to6_predrnn_loss_atten.csv","rb"), delimiter=",", skiprows=0)
csi_CREF = np.loadtxt(open(save_path+"/60&120min/202005270000_to12(06)_CREF.csv","rb"), delimiter=",", skiprows=0)
csi_new_weight2 = np.loadtxt(open(save_path+"/60&120min/202005270000_01to06(mse).csv","rb"), delimiter=",", skiprows=0)
# csi_new_weight2 = np.loadtxt(open(save_path+"/60&120min/202005270000_to12_hprnn_smooth.csv","rb"), delimiter=",", skiprows=0)
csi_new_weight3 = np.loadtxt(open(save_path+"/60&120min/202005270000_01to06(bmse).csv","rb"), delimiter=",", skiprows=0)
csi_new_weight4 = np.loadtxt(open(save_path+"/60&120min/202005270000_01to06(bmse+cbam).csv","rb"), delimiter=",", skiprows=0)
csi_new_weight5 = np.loadtxt(open(save_path+"/60&120min/202005270000_01to06(mse+cbam).csv","rb"), delimiter=",", skiprows=0)
csi_new_weight6 = np.loadtxt(open(save_path+"/60&120min/202005270000to12_10to60(bmse+cbam)w1.csv","rb"), delimiter=",", skiprows=0)
csi_new_weight7 = np.loadtxt(open(save_path+"/60&120min/202005270000to12_10to60(bmse+cbam)w2.csv","rb"), delimiter=",", skiprows=0)
csi_PredRNN = np.loadtxt(open(save_path+"202005270000to6_predrnn_loss_atten.csv","rb"), delimiter=",", skiprows=0)

# Draw thesholds AVG CSI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, 60)
plt.ylim(-0.05, 1.0)
plt.xlabel('Threshold')
plt.ylabel('CSI')
plt.title('{}(10~60mins)'.format(date))
plt.grid(True)

all_csi = []
plt.plot(np.arange(csi_CREF.shape[1]), [np.nan] + np.mean(csi_CREF[:, 1:], 0).tolist(), c='red', linewidth=2.0, label='OP')
# plt.plot(np.arange(csi_PredRNN.shape[1]), [np.nan] + np.mean(csi_PredRNN[:, 1:], 0).tolist(), c='dodgerblue', linewidth=2.0, label='DA-LSTM(bMSE)')

# plt.plot(np.arange(csi_predrnn_loss_atten.shape[1]), [np.nan] + np.mean(csi_predrnn_loss_atten[:, 1:], 0).tolist(), c='b', linewidth=2.0, label='test_1')
# plt.plot(np.arange(csi_new_weight.shape[1]), [np.nan] + np.mean(csi_new_weight[:, 1:], 0).tolist(), c='limegreen', linewidth=2.0, label='mse')
# plt.plot(np.arange(csi_new_weight2.shape[1]), [np.nan] + np.mean(csi_new_weight2[:, 1:], 0).tolist(), c='mediumseagreen', linewidth=2.0, label='HPRNN(MSE)')
# plt.plot(np.arange(csi_new_weight3.shape[1]), [np.nan] + np.mean(csi_new_weight3[:, 1:], 0).tolist(), c='darkorange', linewidth=2.0, label='HPRNN(bMSE)')
# plt.plot(np.arange(csi_new_weight5.shape[1]), [np.nan] + np.mean(csi_new_weight5[:, 1:], 0).tolist(), c='darkorange', linewidth=2.0, label='HPRNN_CBAM(MSE)')
plt.plot(np.arange(csi_new_weight4.shape[1]), [np.nan] + np.mean(csi_new_weight4[:, 1:], 0).tolist(), c='m', linewidth=2.0, label='HPRNN_CBAM(b-MSE)')
plt.plot(np.arange(csi_new_weight6.shape[1]), [np.nan] + np.mean(csi_new_weight6[:, 1:], 0).tolist(), c='olivedrab', linewidth=2.0, label='HPRNN_CBAM(b-mse 2 Groups)')
plt.plot(np.arange(csi_new_weight7.shape[1]), [np.nan] + np.mean(csi_new_weight7[:, 1:], 0).tolist(), c='dodgerblue', linewidth=2.0, label='HPRNN_CBAM(b-mse 3 Groups)')


# plt.plot(np.arange(csi_ConvLSTM.shape[1]), [np.nan] + np.mean(csi_ConvLSTM[:, 1:], 0).tolist(), c='pink', linewidth=2.0, label='ConvLSTM')
# plt.plot(np.arange(csi_ConvLSTM_loss.shape[1]), [np.nan] + np.mean(csi_ConvLSTM_loss[:, 1:], 0).tolist(), c='m', linewidth=2.0, label='ConvLSTM_weightedloss')

# plt.plot(np.arange(csi_PredRNN_loss.shape[1]), [np.nan] + np.mean(csi_PredRNN_loss[:, 1:], 0).tolist(), c='c', linewidth=2.0, label='PredRNN_weightedloss')
# plt.plot(np.arange(csi_MLC.shape[1]), [np.nan] + np.mean(csi_MLC[:, 1:], 0).tolist(), c='y', linewidth=3.0, label='MLC-LSTM CSI')

plt.legend(loc='upper right')

fig.savefig(fname=save_path+'/60&120min/Thresholds_AVG_CSI_0608_to6(test3).png', format='png')
plt.clf()