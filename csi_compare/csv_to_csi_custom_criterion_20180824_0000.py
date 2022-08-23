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

date = '20180824_0000'
# save_path = 'CSI_PICTURE/Compare_{}/'.format(date)
save_path = 'Compare_{}/'.format(date)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

csi_predrnn_loss_atten = np.loadtxt(open(save_path+"60&120min/201808240000to6_predrnn_loss_atten.csv","rb"), delimiter=",", skiprows=0)
csi_CREF = np.loadtxt(open(save_path+"60&120min/201808240000_07to12_cref.csv","rb"), delimiter=",", skiprows=0)
csi_new_weight2 = np.loadtxt(open(save_path+"60&120min/201808240000_07to12(mse).csv","rb"), delimiter=",", skiprows=0)
csi_new_weight3 = np.loadtxt(open(save_path+"60&120min/201808240000_07to12(bmse).csv","rb"), delimiter=",", skiprows=0)
csi_new_weight4 = np.loadtxt(open(save_path+"60&120min/201808240000_07to12(bmse+cbam).csv","rb"), delimiter=",", skiprows=0)
csi_new_weight5 = np.loadtxt(open(save_path+"60&120min/201808240000_07to12(mse+cbam).csv","rb"), delimiter=",", skiprows=0)
csi_new_weight6 = np.loadtxt(open(save_path+"60&120min/201808240000to12_70to120(bmse+cbam)w1.csv","rb"), delimiter=",", skiprows=0)
csi_new_weight7 = np.loadtxt(open(save_path+"60&120min/201808240000to12_70to120(bmse+cbam)w2.csv","rb"), delimiter=",", skiprows=0)

# csi_new_weight_2 = np.loadtxt(open(save_path+"201808240000to6_weight_4.881.csv","rb"), delimiter=",", skiprows=0)
# csi_MLC = np.loadtxt(open("MLC_{}/csi.csv","rb"), delimiter=",", skiprows=0)
# csi_PredRNN = np.loadtxt(open(save_path+"201808240000to6_predrnn_loss_atten_v2.csv","rb"), delimiter=",", skiprows=0)
# csi_PredRNN_loss = np.loadtxt(open(save_path+"201808240000to6_predrnn_loss.csv","rb"), delimiter=",", skiprows=0)
# csi_predrnn_atten = np.loadtxt(open(save_path+"201808240000to6_predrnn_atten.csv","rb"), delimiter=",", skiprows=0)

# csi_ConvLSTM = np.loadtxt(open(save_path+"201808240000to6_convlstm.csv","rb"), delimiter=",", skiprows=0)

# csi_ConvLSTM_loss = np.loadtxt(open(save_path+"201808240000to6_convlstm_loss.csv","rb"), delimiter=",", skiprows=0)

# print("np.array(csi_predrnn_loss_atten).shape",np.array(csi_predrnn_loss_atten).shape)#np.array(csi).shape (6, 60)
# print("np.array(csi_predrnn_loss_atten_GAN).shape",np.array(csi_predrnn_loss_atten_GAN).shape)#np.array(csi).shape (6, 60)
# print("np.array(csi_CREF).shape",np.array(csi_CREF).shape)
# print("np.array(csi_MLC).shape",np.array(csi_MLC).shape)
# print("np.array(csi_PredRNN).shape",np.array(csi_PredRNN).shape)
# print("np.array(csi_PredRNN_loss).shape",np.array(csi_PredRNN_loss).shape)

# Draw thesholds AVG CSI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, 60)
plt.ylim(-0.05, 1.0)
plt.xlabel('Threshold')
plt.ylabel('CSI')
plt.title('{}(70~120mins)'.format(date))
plt.grid(True)

all_csi = []
plt.plot(np.arange(csi_CREF.shape[1]), [np.nan] + np.mean(csi_CREF[:, 1:], 0).tolist(), c='red', linewidth=2.0, label='OP')
# plt.plot(np.arange(csi_predrnn_loss_atten.shape[1]), [np.nan] + np.mean(csi_predrnn_loss_atten[:, 1:], 0).tolist(), c='limegreen', linewidth=2.0, label='DA-LSTM')
# plt.plot(np.arange(csi_new_weight2.shape[1]), [np.nan] + np.mean(csi_new_weight2[:, 1:], 0).tolist(), c='indianred', linewidth=2.0, label='mse')
# plt.plot(np.arange(csi_new_weight3.shape[1]), [np.nan] + np.mean(csi_new_weight3[:, 1:], 0).tolist(), c='m', linewidth=2.0, label='b-mse')
plt.plot(np.arange(csi_new_weight4.shape[1]), [np.nan] + np.mean(csi_new_weight4[:, 1:], 0).tolist(), c='dodgerblue', linewidth=2.0, label='b-mse+cbam')
plt.plot(np.arange(csi_new_weight5.shape[1]), [np.nan] + np.mean(csi_new_weight5[:, 1:], 0).tolist(), c='olivedrab', linewidth=2.0, label='mse+cbam')
plt.plot(np.arange(csi_new_weight6.shape[1]), [np.nan] + np.mean(csi_new_weight6[:, 1:], 0).tolist(), c='orange', linewidth=2.0, label='b-mse+cbam(w1)')
plt.plot(np.arange(csi_new_weight7.shape[1]), [np.nan] + np.mean(csi_new_weight7[:, 1:], 0).tolist(), c='mediumorchid', linewidth=2.0, label='b-mse+cbam(w2)')
# plt.plot(np.arange(csi_ConvLSTM.shape[1]), [np.nan] + np.mean(csi_ConvLSTM[:, 1:], 0).tolist(), c='pink', linewidth=2.0, label='ConvLSTM')

# plt.plot(np.arange(csi_predrnn_loss_atten_GAN.shape[1]), [np.nan] + np.mean(csi_predrnn_loss_atten_GAN[:, 1:], 0).tolist(), c='m', linewidth=2.0, label='PredRNN_bmse_atten_GAN1000')
# plt.plot(np.arange(csi_predrnn_loss_atten_GAN_500.shape[1]), [np.nan] + np.mean(csi_predrnn_loss_atten_GAN_500[:, 1:], 0).tolist(), c='c', linewidth=2.0, label='PredRNN_bmse_atten_GAN500')
# plt.plot(np.arange(csi_PredRNN.shape[1]), [np.nan] + np.mean(csi_PredRNN[:, 1:], 0).tolist(), c='chocolate', linewidth=2.0, label='PredRNN_weightedloss_atten_Transfer')
# plt.plot(np.arange(csi_predrnn_atten.shape[1]), [np.nan] + np.mean(csi_predrnn_atten[:, 1:], 0).tolist(), c='chocolate', linewidth=2.0, label='PredRNN_atten')
# plt.plot(np.arange(csi_ConvLSTM_loss.shape[1]), [np.nan] + np.mean(csi_ConvLSTM_loss[:, 1:], 0).tolist(), c='m', linewidth=2.0, label='ConvLSTM_weightedloss')

# plt.plot(np.arange(csi_PredRNN_loss.shape[1]), [np.nan] + np.mean(csi_PredRNN_loss[:, 1:], 0).tolist(), c='c', linewidth=2.0, label='PredRNN_weightedloss')

plt.legend(loc='upper right')

fig.savefig(fname=save_path+'60&120min/Thresholds_CSI_0608_to12.png', format='png')
plt.clf()
