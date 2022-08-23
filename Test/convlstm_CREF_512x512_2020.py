## Others lib
from http.client import ImproperConnectionState
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pickle as pkl
# from datetime import timedelta

## ML lib
# import tensorflow as tf
# import keras.backend as K
# from keras.models import load_model
# from sklearn.externals import joblib

## Custom lib
# from visualize.Verification_CREF import Verification
# 
from visualize.Verification import Verification 
#  
#from model.CRNN.ConvLSTM_v2 import ConvLSTM
# from data.radar_echo_NWP import load_data
'''
from data.radar_echo_CREF_output512_010_p1 import load_data_CREF
import time
from data.radar_echo_k3_p20_060_drop1800 import load_data
#python -m Test.convlstm_CREF
from CustomUtils_v2 import SaveSummary, generator_getClassifiedItems
'''
from visualize.Verification import Verification 

#from model.CRNN.ConvLSTM_v2 import ConvLSTM
#from data.radar_echo_NWP import load_data
# from data.radar_echo_CREF_out315_5m9d import load_data_CREF
from data.radar_echo_CREF_out512_5m9d_2hr import load_data_CREF

import time
# radar_echo_p20_muti_sample_drop_08241800_load_512x512
# from data.radar_echo_p20_muti_sample_drop_08241800_load_512x512 import load_data
from data.radar_echo_p20_muti_sample_drop_08241800_load_512x512_v2 import load_data

# from data.radar_echo_p20_drop_5m9d import load_data#!
#python -m Test.convlstm_CREF
from CustomUtils_v2 import SaveSummary, generator_getClassifiedItems
# from visualize.visualized_CREF_v2 import visualized_area_with_map
from visualize.visualized_pred import visualized_area_with_map

# # alls = time.clock()

# ## Env setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## paramater setting
model_parameter = {"input_shape": [128, 128],
                   "output_shape": [128, 128],
                   "period": 6,
                   "predict_period": 12,
                   "filter": 36,
                   "kernel_size": [1, 1]}
                   
data_name = '202106060800to12(Sun_Moon_Lake)'
date_date=[['2021-06-06 08:10','2021-06-06 08:11']]
test_date=[['2021-06-06 08:10','2021-06-06 08:11']]
save_path = 'Result/CREF/T6tT6/{}_64x64/'.format(data_name)
if not os.path.isdir(save_path):
   os.makedirs(save_path)
places=['Sun_Moon_Lake']
        
radar_echo_storage_path= 'I:/radar/20210606and0806data/'
load_radar_echo_df_path=None
# load_radar_echo_df_path='E:/radar/201808240to12_512.pkl'
test_y=[]

radar = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                load_radar_echo_df_path=load_radar_echo_df_path,
                input_shape=model_parameter['input_shape'],
                output_shape=model_parameter['output_shape'],
                period=model_parameter['period'],
                predict_period=model_parameter['predict_period'],
                places=places,
                random=False,
                date_range=date_date,
                val_split=0,
                test_date=test_date)

if not load_radar_echo_df_path:    
    radar.saveRadarEchoDataFrame(path = save_path, load_name_pkl = 'test_{}_512x512'.format(data_name))

# save_path = 'Result/research_CREF/research_new/Result/ConvLSTM_CREF/T6tT6/5m9d_512x512/79x79/'
if not os.path.isdir(save_path):
   os.makedirs(save_path)
test_generator = radar.generator('test', batch_size=1)
num_of_month = test_generator.step_per_epoch
print("hour_of_month_total=",num_of_month)
test_x = []
test_y = []

# print("place =",places)
test_x, test_y= generator_getClassifiedItems(test_generator,  places=places) 
print("batch_X.shape, batch_y.shape",np.array(test_x).shape, np.array(test_y).shape)
# batch_X=batch_X.tolist()
# batch_y=batch_y.tolist()
# test_x.append(batch_X)
# test_y.append(batch_y) 

# print("test_x.shape, test_y.shape",np.array(test_x).shape, np.array(test_y).shape)
# # test_y.to_excel("test_y.xlsx'")

# test_x = np.concatenate(test_x, 0) #合併所有批次成一個資料集 32+32.....=1152
# test_y = np.concatenate(test_y, 0)
# test_y2=test_y.reshape(-1)
# test_y2=test_y.reshape(-1)
# np.savetxt(save_path+'test_{}_512x512.csv'.format(data_name), np.array(test_y2), delimiter = ',')
print('test_x.shape = ', test_x.shape)
print('test_y.shape = ', test_y.shape)
# test_x.shape =  (1, 6, 512, 512, 1)
# test_y.shape =  (1, 595350)
# test_x.shape, test_y.shape (1, 6, 6, 3, 3, 1) (1, 6, 108)
# test_x.shape =  (6, 6, 3, 3, 1)
# test_y.shape =  (6, 108)

print("=============================================")
print("=============================================")
print("=============================================")
radar_echo_storage_path = 'I:/radar/20210606and0806data/'
# load_radar_echo_df_path ='I:/yu_ting/Seq2Seq_Radar_Echo_Evaluation-main/Result/CREF/CREF_pkl_2hr/202005270000to12/CREF_202005270000to12.pkl'
load_radar_echo_df_path=None
places=['Sun_Moon_Lake']
model_parameter = {"input_shape": [640, 640],
                   "output_shape": [640, 640],
                   "period": 6,
                   "predict_period": 12,
                   "filter": 36,
                   "kernel_size": [1, 1]}
  
radar1 = load_data_CREF(radar_echo_storage_path=radar_echo_storage_path, 
                        load_radar_echo_df_path=load_radar_echo_df_path,
                        input_shape=model_parameter['input_shape'],
                        output_shape=model_parameter['output_shape'],
                        period=model_parameter['period'],
                        predict_period=model_parameter['predict_period'],
                        places=places,
                        date_range=date_date,
                        test_date=test_date,
                        val_split=0,                        
                        radar_echo_name_format=['CREF_010min.%Y%m%d.%H%M%S', 'CREF_020min.%Y%m%d.%H%M%S', 
                                                'CREF_030min.%Y%m%d.%H%M%S', 'CREF_040min.%Y%m%d.%H%M%S', 
                                                'CREF_050min.%Y%m%d.%H%M%S', 'CREF_060min.%Y%m%d.%H%M%S',
                                                'CREF_070min.%Y%m%d.%H%M%S', 'CREF_080min.%Y%m%d.%H%M%S',
                                                'CREF_090min.%Y%m%d.%H%M%S', 'CREF_100min.%Y%m%d.%H%M%S',
                                                'CREF_110min.%Y%m%d.%H%M%S', 'CREF_120min.%Y%m%d.%H%M%S'
                                                ])
save_path="Result/CREF/CREF_pkl_2hr/"+data_name+"/"
if not os.path.isdir(save_path):
   os.makedirs(save_path)
if not load_radar_echo_df_path:
    print("save CREF pkl...")
    radar1.saveRadarEchoDataFrame(path=save_path,load_name_pkl='CREF_{}'.format(data_name))
# save_path = 'Result/research_CREF/research_new/Result/ConvLSTM_CREF/T6tT6/5m9d_512x512/'
# if not os.path.isdir(save_path):
#    os.makedirs(save_path)
test_generator_CREF = radar1.generator('test', batch_size=1)

pred_x = []
pred_y = []

    # radar.saveRadarEchoDataFrame(load_name_pkl='CREF_5m9d_512x512')
pred_x = pd.read_pickle(save_path+'CREF_{}.pkl'.format(data_name))
# save_path = 'Result/research_CREF/research_new/Result/ConvLSTM_CREF/T6tT6/9_12/'
if not os.path.isdir(save_path):
   os.makedirs(save_path)
print("pred_x")
print(pred_x)
pred_x_pkl=np.array(pred_x)
print("pred_x_pkl")# (1, 6)
# print(pred_x_pkl)
print("pred_x_pkl",np.array(pred_x_pkl).shape)
# np.savetxt('pred_x_512x512_CREF.csv', pred_x_pkl, delimiter = ',')
# pred_x.to_excel('pred_x_p20.xlsx')
# p20_CREF.xlsx
pred_y = pred_x
# pred_y = pred_x.reset_index()
# np.savetxt('pred_y_512x512_CREF.csv', pred_y, delimiter = ',')
# pred_y.to_excel(save_path+'9_12_pred_y_p1_512x512_0824_0010.txt')
# np.savetxt('9_12_pred_y_p1_512x512_0824_0010_file.txt', np.array(pred_y), delimiter = ',')
# print(pred_y[['010','020','030','040']])
#pred_y = pred_y.loc[1:6, 3:8]
print('before pred_y.shape = ', pred_y.shape)# (1144, 8)
# pred_y = pred_y.loc[1:6, ['010','020','030','040','050','060']]#(6,6)
# pred_y = pred_y.tolist()
# print("pred_y=",pred_y[:][])
print("pred_y.shape",np.array(pred_y).shape)# (1, 6)
# print(pred_y['010'].size)
# print(pred_y['010'][0].size)
# print(pred_y['010'][0][0].size)
# print(pred_y['020'][0].size)
# print(pred_y['030'][0].size)
# print(pred_y['040'][0].size)
# print(pred_y['050'][0].size)
# print(pred_y['020'][0].size)

pred_y=np.array(pred_y)
# pred_list=pred_list.tolist()
# time_range=['010']
pred_list=[]


print("len(pred_y)=",len(pred_y))
for i in range(len(pred_y)):
    for j in range(12):#time_range:
        pred_list.append(pred_y[i][j][:][:])
        print("pred_list.shape",np.array(pred_list).shape)

        # print("list pred_y[i][j].shape",np.array(pred_y[i][time_range[j]]).shape)
        
print("pred_list.shape",np.array(pred_list).shape)

pred_list = np.array(pred_list).reshape(-1,12,640,640)#!
print("pred_list.shape=",pred_list.shape)
# pred_y=np.array(pred_list).reshape(1, 6,512*512)
# test_y=np.array(test_y[0]).reshape(1, 6,512*512)

# pic = []
# for i in range(12):
#     pic.append(pred_list[:,i,:,:].reshape(640,640))

'''
save_path_640 = save_path +"640/"
if not os.path.isdir(save_path_640):
   os.makedirs(save_path_640)
for i in range(6):
    visualized_area_with_map(pic[i], 'Sun_Moon_Lake', shape_size=[640,640], title='pred_list_{}_640'.format(i), savepath=save_path_640)
'''
# real_frm1 = test_y.reshape(-1)
# pred_frm1 = pred_list.reshape(-1)
#if not os.path.isdir(save_path):
#    os.makedirs(save_path)
# np.savetxt(save_path+'real_5m9d_p1_512x512_0824.txt', real_frm1)
# np.savetxt(save_path+'pred_5m9d_p1_512x512_0824.txt', pred_frm1)

# print('test_y.shape = ', test_y.shape)
# print('real_frm1.shape = ', real_frm1.shape)

print('----------------------test_y------------------')
print(test_y)
print('type(test_y) = ', type(test_y))

test_y = np.array(test_y).reshape(-1,12,128,128)
print('test_y.shape = ', test_y.shape)

'''
## predcit
#y是全連接層一維陣列，批次(二維打直)，模型是接收第二維度資料
for i in range(11):
    for j in range(11):
        print(test_x[:, :, i:i+21, j:j+21].shape) #133, 12
        print(test_y.reshape(-1, 12, 11, 11)[:, :, i, j].shape)#合併批次 = 133, 12, 11, 11
        exit()
''' 
#print(pred_x.shape)
print('pred_list.shape = ', pred_list.shape)
print('test_y.shape = ', test_y.shape)
# test_y.shape =  (1, 6, 512, 512)
# pred_list.shape =  (1, 6, 512, 512)
# pred_list=pred_list.tolist()
# test_y = test_y.tolist()
# loadCe = time.clock()

pred_list_0=pred_list[:,0,:,:].reshape(-1)
pred_list_1=pred_list[:,1,:,:].reshape(-1)
pred_list_2=pred_list[:,2,:,:].reshape(-1)
pred_list_3=pred_list[:,3,:,:].reshape(-1)
pred_list_4=pred_list[:,4,:,:].reshape(-1)
pred_list_5=pred_list[:,5,:,:].reshape(-1)
pred_list_6=pred_list[:,6,:,:].reshape(-1)
pred_list_7=pred_list[:,7,:,:].reshape(-1)
pred_list_8=pred_list[:,8,:,:].reshape(-1)
pred_list_9=pred_list[:,9,:,:].reshape(-1)
pred_list_10=pred_list[:,10,:,:].reshape(-1)
pred_list_11=pred_list[:,11,:,:].reshape(-1)

from scipy import interpolate
x =np.linspace(0,1,640)
y =np.linspace(0,1,640)

f0 = interpolate.interp2d(x, y, pred_list_0, kind='linear')
f1 = interpolate.interp2d(x, y, pred_list_1, kind='linear')#linear
f2 = interpolate.interp2d(x, y, pred_list_2, kind='linear')
f3 = interpolate.interp2d(x, y, pred_list_3, kind='linear')
f4 = interpolate.interp2d(x, y, pred_list_4, kind='linear')
f5 = interpolate.interp2d(x, y, pred_list_5, kind='linear')
f6 = interpolate.interp2d(x, y, pred_list_6, kind='linear')
f7 = interpolate.interp2d(x, y, pred_list_7, kind='linear')
f8 = interpolate.interp2d(x, y, pred_list_8, kind='linear')
f9 = interpolate.interp2d(x, y, pred_list_9, kind='linear')
f10 = interpolate.interp2d(x, y, pred_list_10, kind='linear')
f11 = interpolate.interp2d(x, y, pred_list_11, kind='linear')

x1 =np.linspace(0,1,512)
y2 =np.linspace(0,1,512)
pred_list_0 = f0(x1, y2)
pred_list_1 = f1(x1, y2)
pred_list_2 = f2(x1, y2)
pred_list_3 = f3(x1, y2)
pred_list_4 = f4(x1, y2)
pred_list_5 = f5(x1, y2)
pred_list_6 = f6(x1, y2)
pred_list_7 = f7(x1, y2)
pred_list_8 = f8(x1, y2)
pred_list_9 = f9(x1, y2)
pred_list_10 = f10(x1, y2)
pred_list_11 = f11(x1, y2)

pred_list_0 = np.array(pred_list_0).reshape(512,512)
print("pred_list_0.shape=",pred_list_0.shape)
pred_list_1 = np.array(pred_list_1).reshape(512,512)
pred_list_2 = np.array(pred_list_2).reshape(512,512)
pred_list_3 = np.array(pred_list_3).reshape(512,512)
pred_list_4 = np.array(pred_list_4).reshape(512,512)
pred_list_5 = np.array(pred_list_5).reshape(512,512)
pred_list_6 = np.array(pred_list_6).reshape(512,512)
pred_list_7 = np.array(pred_list_7).reshape(512,512)
pred_list_8 = np.array(pred_list_8).reshape(512,512)
pred_list_9 = np.array(pred_list_9).reshape(512,512)
pred_list_10 = np.array(pred_list_10).reshape(512,512)
pred_list_11 = np.array(pred_list_11).reshape(512,512)

# pred_list_0=np.transpose((1,0))
# pred_list_1 = np.transpose((1,0))
# pred_list_2 = np.transpose((1,0))
# pred_list_3 = np.transpose((1,0))
# pred_list_4 = np.transpose((1,0))
# pred_list_5 = np.transpose((1,0))
# pred_list_2 = np.transpose(1,0)

a=int(512/2-128/2)
b=int(512/2+128/2)
pred_list_0 = pred_list_0[a:b,a:b]
pred_list_1 = pred_list_1[a:b,a:b]
pred_list_2 = pred_list_2[a:b,a:b]
pred_list_3 = pred_list_3[a:b,a:b]
pred_list_4 = pred_list_4[a:b,a:b]
pred_list_5 = pred_list_5[a:b,a:b]
pred_list_6 = pred_list_6[a:b,a:b]
pred_list_7 = pred_list_7[a:b,a:b]
pred_list_8 = pred_list_8[a:b,a:b]
pred_list_9 = pred_list_9[a:b,a:b]
pred_list_10 = pred_list_10[a:b,a:b]
pred_list_11 = pred_list_11[a:b,a:b]
print("pred_list_0.shape",pred_list_0.shape)

# sys.exit()
# print(" N val loss", np.mean((pred_list-test_y)**2))
# sys.exit()

# mse010=np.mean((pred_list[:][0][:][:]-test_y[:][0][:][:])**2)
# print("補nan前 all mse010 =",mse010)#!!一天結果
# picture_mse = (np.square(pred_list - test_y).sum())/(512*512)
# print("picture_mse=",picture_mse)
# nan_data=np.isnan(np.array(pred_list))
# pred_list=np.array(pred_list)
# pred_list[nan_data] = 0


nan_data_test_y=np.isnan(np.array(test_y))
test_y=np.array(test_y)
test_y[nan_data_test_y] = 0
'''
real_frm1 = test_y.reshape(-1)
pred_frm1 = pred_list.reshape(-1)
#if not os.path.isdir(save_path):
#    os.makedirs(save_path)
np.savetxt('real{}.txt'.format("9_12_p1_512x512_0824"), real_frm1)
np.savetxt('pred_frm{}.txt'.format("9_12_p1_512x512_0824"), pred_frm1)
'''
pered_all=[]
pered_all.append(pred_list_0.T)
pered_all.append(pred_list_1.T)
pered_all.append(pred_list_2.T)
pered_all.append(pred_list_3.T)
pered_all.append(pred_list_4.T)
pered_all.append(pred_list_5.T)
pered_all.append(pred_list_6.T)
pered_all.append(pred_list_7.T)
pered_all.append(pred_list_8.T)
pered_all.append(pred_list_9.T)
pered_all.append(pred_list_10.T)
pered_all.append(pred_list_11.T)

mse010=np.mean((pred_list_0.T-test_y[:,0,:,:])**2)
print("mse010=",mse010)

mse020=np.mean((pred_list_1.T-test_y[:,1,:,:])**2)
print("mse020=",mse020)

mse030=np.mean((pred_list_2.T-test_y[:,2,:,:])**2)
print("mse030=",mse030)

mse040=np.mean((pred_list_3.T-test_y[:,3,:,:])**2)
print("mse040=",mse040)

mse050=np.mean((pred_list_4.T-test_y[:,4,:,:])**2)
print("mse050=",mse050)

mse060=np.mean((pred_list_5.T-test_y[:,5,:,:])**2)
print("mse060=",mse060)

mse070=np.mean((pred_list_6.T-test_y[:,6,:,:])**2)
print("mse070=",mse070)

mse080=np.mean((pred_list_7.T-test_y[:,7,:,:])**2)
print("mse080=",mse080)

mse090=np.mean((pred_list_8.T-test_y[:,8,:,:])**2)
print("mse090=",mse090)

mse100=np.mean((pred_list_9.T-test_y[:,9,:,:])**2)
print("mse100=",mse100)

mse110=np.mean((pred_list_10.T-test_y[:,10,:,:])**2)
print("mse110=",mse110)

mse120=np.mean((pred_list_11.T-test_y[:,11,:,:])**2)
print("mse120=",mse120)

save_path_512 = save_path +"/512/"
if not os.path.isdir(save_path_512):
   os.makedirs(save_path_512)
# import time
# time.sleep(60000)

# for i in range(12):
#     test = np.squeeze(test_y[:,i,:,:])
#     visualized_area_with_map(test,'Sun_Moon_Lake', shape_size=[128,128], title='GT{}'.format(i), savepath=save_path_512)

visualized_area_with_map(pred_list_0.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_0', savepath=save_path_512)
visualized_area_with_map(pred_list_1.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_1', savepath=save_path_512)
visualized_area_with_map(pred_list_2.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_2', savepath=save_path_512)
visualized_area_with_map(pred_list_3.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_3', savepath=save_path_512)
visualized_area_with_map(pred_list_4.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_4', savepath=save_path_512)
visualized_area_with_map(pred_list_5.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_5', savepath=save_path_512)
visualized_area_with_map(pred_list_6.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_6', savepath=save_path_512)
visualized_area_with_map(pred_list_7.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_7', savepath=save_path_512)
visualized_area_with_map(pred_list_8.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_8', savepath=save_path_512)
visualized_area_with_map(pred_list_9.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_9', savepath=save_path_512)
visualized_area_with_map(pred_list_10.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_10', savepath=save_path_512)
visualized_area_with_map(pred_list_11.T, 'Sun_Moon_Lake', shape_size=[128,128], title='CREF_pred_11', savepath=save_path_512)

# sys.exit()
# mse020=np.mean((pred_list[:][1][:][:]-test_y[:][1][:][:])**2)
# mse030=np.mean((pred_list[:][2][:][:]-test_y[:][2][:][:])**2)
# mse040=np.mean((pred_list[:][3][:][:]-test_y[:][3][:][:])**2)
# mse050=np.mean((pred_list[:][4][:][:]-test_y[:][4][:][:])**2)
# mse060=np.mean((pred_list[:][5][:][:]-test_y[:][5][:][:])**2)
# print("test_date=[['2018-08-24 00:10', '2018-08-25 00:00']]")

# print("00預測第一筆 mse",np.mean((pred_list[:][0][:][:]-test_y[:][0][:][:])**2))
# print("00預測第二筆 mse",np.mean((pred_list[:][1][:][:]-test_y[:][1][:][:])**2))
# print("00預測第三筆 mse",np.mean((pred_list[:][2][:][:]-test_y[:][2][:][:])**2))
# print("00預測第四筆 mse",np.mean((pred_list[:][3][:][:]-test_y[:][3][:][:])**2))
# print("00預測第五筆 mse",np.mean((pred_list[:][4][:][:]-test_y[:][4][:][:])**2))
# print("00預測第六筆 mse",np.mean((pred_list[:][5][:][:]-test_y[:][5][:][:])**2))
# # from visualize.Verification import Verification 
# radar_pred = np.array(pred_list[0][0][:][:]).reshape(128,128)
# print("radar_pred=",radar_pred)
# radar_test_y = np.array(test_y[0][0][:][:]).reshape(128,128)
# print("radar_test_y=",radar_test_y)

img_mse=[]
img_mse_picture=[]
pred_np=[]
for i in range(12):
   img_mse.append(0)
   img_mse_picture.append(0)
   

print(" ")


mse_1th =(mse010+mse020+mse030+mse040+mse050+mse060)/6
mse_2th = (mse070+mse080+mse090+mse100+mse110+mse120)/6
mse_all=(mse_1th+mse_2th)/2
fn = save_path_512 + '{}_mse.txt'.format(data_name)
with open(fn,'a') as file_obj:
    file_obj.write('mse=' + str(mse_all)+'\n')
    # file_obj.write('mse_picture=' + str(mse_picture)+'\n')

rmse=np.sqrt(mse_all)
rmse1=np.sqrt(mse_1th)
rmse2=np.sqrt(mse_2th)
fn = save_path_512 + '{}_rmse.txt'.format(data_name)
with open(fn,'a') as file_obj:
    file_obj.write('rmse=' + str(rmse)+'\n')
    file_obj.write('rmse1=' + str(rmse1)+'\n')
    file_obj.write('rmse2=' + str(rmse2)+'\n')
# print("len(pred_list)=",len(pred_list))
# for i in range(len(pred_list)):
#    save_path_picture =save_path+'i{}'.format(i) 
#    if not os.path.isdir(save_path_picture):
#       os.makedirs(save_path_picture)
#    for j in range(6):
#       radar_test_y  = np.array(test_y[i,j,2::4,1::4]).reshape(-1)

#     # radar_test_y  = np.array(test_y[i][j][2::4][1::4]).reshape(-1)
#       print("radar_test_y=",radar_test_y.shape)
#       radar_test_y = radar_test_y[:6241].reshape(79,79)
#       print("radar_test_y[:6241]=",radar_test_y.shape)
#       radar_pred= np.array(pred_list[i,j,2::5,1::5]).reshape(-1)

#     #   radar_pred= np.array(pred_list[i][j][2::5][1::5]).reshape(-1)
#       print("radar_pred=",radar_pred.shape)

#       radar_pred = radar_pred[:6241].reshape(79,79)
      
#       print("radar_pred[:6241]=",radar_pred.shape)
    
#       mse = np.mean((radar_pred-radar_test_y)**2)
#       img_mse[j]+=mse
    #   from visualize.visualized_pred import visualized_area_with_map

    #   visualized_area_with_map(test_y[i][j][:][:], 'Sun_Moon_Lake', shape_size=[128,128], title='radar_test_y_i{}.j{}'.format(i,j), savepath=save_path_picture)
      
#     #   from visualize.visualized_CREF_v2 import visualized_area_with_map2
#       print("pred_list[i][j][:][:].shape",pred_list[i][j][:][:].shape)
#       visualized_area_with_map(pred_list[i][j][:][:], 'Sun_Moon_Lake', shape_size=[393,393], title='radar_pred_y_i{}.j{}'.format(i,j), savepath=save_path_picture)

#       mse_picture = (np.square(radar_pred- radar_test_y).sum())/(79*79)
#       img_mse_picture[j]+=mse_picture
#       fn = save_path_picture + 'div.txt'
#       with open(fn,'a') as file_obj:
#          file_obj.write('mse=' + str(mse)+'\n')
#          file_obj.write('mse_picture=' + str(mse_picture)+'\n')


# fn = save_path + 'T05270000_mse_6.txt'
# with open(fn,'a') as file_obj:
#    for i in range(6):
#       avg_mse = img_mse[i]/len(pred_list)
#       avg_mse_picture = img_mse_picture[i]/len(pred_list)

#       file_obj.write('mse[{}]='.format(i) + str(img_mse[i])+'\n')
#       file_obj.write('avg_mse[{}]='.format(i) + str(avg_mse)+'\n')
#       file_obj.write('avg_mse_picture[{}]='.format(i) + str(avg_mse_picture)+'\n')

# print("預測第一筆 picture_mse",(np.square(pred_list[:][0][:][:] - test_y[:][0][:][:]).sum())/(512*512))
# print("預測第二筆 picture_mse",(np.square(pred_list[:][1][:][:] - test_y[:][1][:][:]).sum())/(512*512))
# print("預測第三筆 picture_mse",(np.square(pred_list[:][2][:][:] - test_y[:][2][:][:]).sum())/(512*512))
# print("預測第四筆 picture_mse",(np.square(pred_list[:][3][:][:] - test_y[:][3][:][:]).sum())/(512*512))
# print("預測第五筆 picture_mse",(np.square(pred_list[:][4][:][:] - test_y[:][4][:][:]).sum())/(512*512))
# print("預測第六筆 picture_mse",(np.square(pred_list[:][5][:][:] - test_y[:][5][:][:].sum())/(512*512)))
# print("一小時 N val loss", np.mean((pred_list[:6][:][:][:]-test_y[:6][:][:][:])**2))
# print("預測第一筆 mse",np.mean((pred_list[:6][0][:][:]-test_y[:6][0][:][:])**2))
# print("預測第二筆 mse",np.mean((pred_list[:6][1][:][:]-test_y[:6][1][:][:])**2))
# print("預測第三筆 mse",np.mean((pred_list[:6][2][:][:]-test_y[:6][2][:][:])**2))
# print("預測第四筆 mse",np.mean((pred_list[:6][3][:][:]-test_y[:6][3][:][:])**2))
# print("預測第五筆 mse",np.mean((pred_list[:6][4][:][:]-test_y[:6][4][:][:])**2))
# print("預測第六筆 mse",np.mean((pred_list[:6][5][:][:]-test_y[:6][5][:][:])**2))


# print("all mse010 =",mse010)#!!一天結果

# csis = time.clock()
# ----CSI----

# Color = ['#00FFFF', '#4169E1', '#0000CD', '#ADFF2F', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#9932CC']
#     ## CSI comput
csi = []
    # save_path = 'Result/ConvLSTM_CREF/output512/8_240000_0059_v3/{}_T6toT{}/'.format(str(places),model_parameter['predict_period'])


    # save_path = 'Result/ConvLSTM_CREF/T6tT6/11_10/output_512x512/8_240010_8_240100_TEST/'

    # if not os.path.isdir(save_path):
        # os.makedirs(save_path)
    # print("radar_test_y_all=",np.array(radar_test_y_all).shape)

for period in range(model_parameter['predict_period']):
    #    print('pred_list[:, period] = ', np.array(pred_list[:, period]).shape)
    #    print('test_y[:, period] = ', np.array(test_y[:, period]).shape)
    #    csi_eva = Verification(pred=pred_list[:, :].reshape(-1, 1), target=test_y[:, :].reshape(-1, 1), threshold=60, datetime='')
    csi_eva = Verification(pred=pered_all[period].reshape(-1, 1), target=test_y[:, period,:,:].reshape(-1, 1), threshold=60, datetime='')
    print("csi_eva.csi shape = ",np.array(csi_eva.csi).shape)
    print("np.nanmean(csi_eva.csi, axis=1) shape = ",np.nanmean(csi_eva.csi, axis=1).shape)
    csi.append(np.nanmean(csi_eva.csi, axis=1))
            # pred_list[:, period] =  (6, 11, 11)
            # test_y[:, period] =  (6, 11, 11)
            # csi_eva.csi shape =  (60, 726)
            # np.nanmean(csi_eva.csi, axis=1) shape =  (60,)
            # csi shape =  (6, 60)
csi = np.array(csi).reshape(12,60)
print("csi shape = ",csi.shape)
np.savetxt(save_path+'{}.csv'.format(data_name), csi, delimiter = ',')
np.savetxt(save_path+'{}_01to06_CREF.csv'.format(data_name), csi[:6,], delimiter = ',')
np.savetxt(save_path+'{}_07to12_CREF.csv'.format(data_name), csi[6:,], delimiter = ',')
# np.savetxt('2darray.csv', csi, delimiter=',', fmt='%d')
# csi.tofile('foo.csv',sep=',')
# csi = np.genfromtxt("I:/yu_ting/Seq2Seq_Radar_Echo_Evaluation-main/Result/CREF/CREF_pkl_2hr/202005270000to12/202005270000to06csi.csv",delimiter=',')

## Draw thesholds AVG CSI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, 60)
plt.ylim(-0.05, 1.0)
plt.xlabel('Threshold')
plt.ylabel('CSI')
plt.title('{}\nThresholds CSI'.format(data_name))
plt.grid(True)

all_csi = []
plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), linewidth=2.0, label='AVG CSI')

plt.legend(loc='upper right')

fig.savefig(fname=save_path_512+'Thresholds_AVG_CSI.png', format='png')
plt.clf()
print("ok")


#csie = time.clock()
#
#alle = time.clock()
#
#print("load NWP time = ", loadNe - loadNs)
#print("load CREF time = ", loadCe - loadCs)
#print("All time = ", alle - alls)
