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

date = '20210806_1300'
# save_path = 'CSI_PICTURE/Compare_{}/'.format(date)
save_path = 'Compare_{}_128/'.format(date)

csi_CREF = np.loadtxt(open(save_path+"/202108061300to12(Tengzhi)_128_07to12_CREF.csv","rb"), delimiter=",", skiprows=0)
csi_HPRNN = np.loadtxt(open(save_path+"/202108061300to12(Tengzhi)_128_07to12.csv","rb"), delimiter=",", skiprows=0)

# Draw thesholds AVG CSI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, 60)
plt.ylim(-0.05, 1.0)
plt.xlabel('Threshold')
plt.ylabel('CSI')
plt.title('{}(60~120mins)'.format(date))
plt.grid(True)

all_csi = []
plt.plot(np.arange(csi_CREF.shape[1]), [np.nan] + np.mean(csi_CREF[:, 1:], 0).tolist(), c='red', linewidth=2.0, label='OP')
plt.plot(np.arange(csi_HPRNN.shape[1]), [np.nan] + np.mean(csi_HPRNN[:, 1:], 0).tolist(), c='b', linewidth=2.0, label='HPRNN')

plt.legend(loc='upper right')

fig.savefig(fname=save_path+'/'+date+'_AVG_CSI_to12.png', format='png')
plt.clf()
