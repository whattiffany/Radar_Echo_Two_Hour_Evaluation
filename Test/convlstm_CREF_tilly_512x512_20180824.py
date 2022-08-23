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
# from keras.models import load_model
# from sklearn.externals import joblib

## Custom lib
# from visualize.Verification_CREF import Verification
# 
# from visualize.Verification import Verification 
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
from data.radar_echo_CREF_out512_5m9d import load_data_CREF

import time
# radar_echo_p20_muti_sample_drop_08241800_load_512x512
from data.radar_echo_p20_muti_sample_drop_08241800_load_512x512 import load_data

# from data.radar_echo_p20_drop_5m9d import load_data#!
#python -m Test.convlstm_CREF
from CustomUtils_v2 import SaveSummary, generator_getClassifiedItems
# from visualize.visualized_CREF_v2 import visualized_area_with_map
from visualize.visualized_pred import visualized_area_with_map

# alls = time.clock()

## Env setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config) 
# K.set_session(sess)

#area_20={
#        'Banqiao':area_20_466880,
#        'Keelung':area_20_466940,
#        'Taipei':area_20_466920,
#        'New_House':area_20_467050,
#        'Chiayi':area_20_467480,
#        'Dawu':area_20_467540,
#        'Hengchun':area_20_467590,
#        'Success':area_20_467610,
#        'Sun_Moon_Lake':area_20_467650,
#        'Taitung':area_20_467660,
#        'Yuxi':area_20_467770,
#        'Hualien':area_20_466990,
#        'Beidou':area_20_C0G840,
#        'Bao_Zhong':area_20_C0K430,
#        'Chaozhou':area_20_C0R220,
#        'News':area_20_C0R550,
#        'Member_Hill':area_20_C0U990,
#        'Yuli':area_20_C0Z061,
#        'Snow_Ridge':area_20_C1F941,
#        'Shangdewen':area_20_C1R120,
#        }

## paramater setting
model_parameter = {"input_shape": [512, 512],
                   "output_shape": [512, 512],
                   "period": 6,
                   "predict_period": 6,
                   "filter": 36,
                   "kernel_size": [1, 1]}
                   
data_name = '20180824000to6'
csi_T ='20180824000'
save_path = 'Result/CREF/T6tT6/{}_512x512mse/'.format(data_name)
if not os.path.isdir(save_path):
   os.makedirs(save_path)
# date_date=[['2020-05-22 04:10', '2020-05-22 04:10']]
# date_date=[['2020-05-22 04:10', '2020-05-22 04:10']]
date_date=[['2018-08-24 00:10', '2018-08-24 00:10']]
test_date=[['2018-08-24 00:10', '2018-08-24 00:10']]

# date_date=[['2020-05-16 08:10', '2020-05-16 08:11']]
# test_date=[['2020-05-16 08:10', '2020-05-16 08:11']]
# date_date=[['2020-05-27 00:10', '2020-05-27 00:19']]
# test_date=[['2020-05-27 00:10', '2020-05-27 00:19']]

# date_date=[['2018-08-23 22:00', '2018-08-24 01:30']]
# test_date=[['2018-08-24 00:10', '2018-08-24 00:11']]
# date_date=[['2020-05-16 00:10','2020-05-16 01:00']]#,
# test_date=[['2020-05-16 00:10','2020-05-16 01:00']]#,

# date_date=[['2020-05-16 00:10','2020-05-17 00:00']]#,
            # ['2020-05-19 00:10','2020-05-20 00:00'],
            # ['2020-05-21 00:10','2020-05-22 00:00'],
            # ['2020-05-22 00:10','2020-05-23 00:00'],
            # ['2020-05-23 00:10','2020-05-24 00:00'],
            # ['2020-05-26 00:10','2020-05-27 00:00'],
            # ['2020-05-27 00:10','2020-05-28 00:00'],
            # ['2020-05-28 00:10','2020-05-29 00:00'],
            # ['2020-05-29 00:10','2020-05-30 00:00']]

# test_date=[['2020-05-16 00:10','2020-05-17 00:00']]#,
            # ['2020-05-19 00:10','2020-05-20 00:00'],
            # ['2020-05-21 00:10','2020-05-22 00:00'],
            # ['2020-05-22 00:10','2020-05-23 00:00'],
            # ['2020-05-23 00:10','2020-05-24 00:00'],
            # ['2020-05-26 00:10','2020-05-27 00:00'],
            # ['2020-05-27 00:10','2020-05-28 00:00'],
            # ['2020-05-28 00:10','2020-05-29 00:00'],
            # ['2020-05-29 00:10','2020-05-30 00:00']]
# date_date=[['2018-08-23 23:00', '2018-08-24 02:10']]
# # test_date=[['2018-08-24 01:00', '2018-08-24 17:59'],['2018-08-24 18:01', '2018-08-25 00:59']]
# test_date=[['2018-08-24 01:00', '2018-08-24 01:10']]
# places=['Sun_Moon_Lake','Taipei'
# places=['Sun_Moon_Lake']#,'Shangdewen']
# save_path = 'Result/ConvLSTM_nokmean_model_output512/CREF/8_240000_0059/{}_T6toT{}/'.format(places,model_parameter['predict_period'])
# places=['Banqiao','Keelung','Taipei','New_House','Chiayi',
#         'Dawu','Hengchun','Success','Sun_Moon_Lake','Taitung',
#         'Yuxi','Hualien','Beidou','Bao_Zhong','Chaozhou',
#         'News','Member_Hill','Yuli','Snow_Ridge','Shangdewen']
places=['Sun_Moon_Lake']
        
# places=['Banqiao','Keelung']
radar_echo_storage_path= 'NWP/'
load_radar_echo_df_path='data/201808240000to6_512x512.pkl'
radar = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                load_radar_echo_df_path=load_radar_echo_df_path,
                input_shape=model_parameter['input_shape'],
                output_shape=model_parameter['output_shape'],
                period=model_parameter['period'],
                predict_period=model_parameter['predict_period'],
                places=places,
                random=False,
                date_range=date_date,
                test_date=test_date)

if not load_radar_echo_df_path:
    # radar.exportRadarEchoFileList()
    radar.saveRadarEchoDataFrame(load_name_pkl='test_{}_512x512'.format(data_name))
    radar.saveRadarEchoDataFrame(path = save_path, load_name_pkl = 'test_{}_512x512'.format(data_name))

# save_path = 'Result/research_CREF/research_new/Result/ConvLSTM_CREF/T6tT6/5m9d_512x512/79x79/'
if not os.path.isdir(save_path):
   os.makedirs(save_path)
test_generator = radar.generator('test', batch_size=32)
test_x = []
test_y = []
for place in places:
    print("place =",place)
    # save_path = 'Result/ConvLSTM_nokmean_model_p20_try_cis_avg/input55_convlstm2_k7_filter36_pred060/3_13/{}_T6toT{}/'.format(place,model_parameter['predict_period'])
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)


    batch_X, batch_y= generator_getClassifiedItems( test_generator,  places=[place]) 
    print("batch_X.shape, batch_y.shape",np.array(batch_X).shape, np.array(batch_y).shape)
    batch_X=batch_X.tolist()
    batch_y=batch_y.tolist()
    test_x.append(batch_X)
    test_y.append(batch_y) 

    print("test_x.shape, test_y.shape",np.array(test_x).shape, np.array(test_y).shape)
# test_y.to_excel("test_y.xlsx'")

test_x = np.concatenate(test_x, 0) #合併所有批次成一個資料集 32+32.....=1152
test_y = np.concatenate(test_y, 0)
# test_y2=test_y.reshape(-1)
test_y2=test_y.reshape(-1)
np.savetxt(save_path+'test_{}_512x512.csv'.format(data_name), np.array(test_y2), delimiter = ',')
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
radar_echo_storage_path = 'CREF/'
load_radar_echo_df_path =None#'data/Sun_Moon_Lake_RadarEcho_512x512_CREF.pkl'
date_date=[['2018-08-24 00:00', '2018-08-24 00:09']]
test_date=[['2018-08-24 00:00', '2018-08-24 00:09']]
# date_date=[['2020-05-29 06:10', '2020-05-29 06:10']]
# test_date=[['2020-05-29 06:10', '2020-05-29 06:10']]

# date_date=[['2020-05-22 04:10', '2020-05-22 04:10']]
# test_date=[['2020-05-22 04:10', '2020-05-22 04:10']]
# date_date=[['2020-05-22 04:00', '2020-05-22 04:00']]
# test_date=[['2020-05-22 04:00', '2020-05-22 04:00']]
# date_date=[['2020-05-16 08:00', '2020-05-16 08:09']]
# test_date=[['2020-05-16 08:00', '2020-05-16 08:09']]
# date_date=[['2020-05-27 00:00', '2020-05-27 00:09']]
# test_date=[['2020-05-27 00:00', '2020-05-27 00:09']]
# date_date=[['2020-05-16 00:00','2020-05-16 00:59']]#,

# date_date=[['2020-05-16 00:00','2020-05-16 23:59']]#,
            # ['2020-05-19 00:00','2020-05-19 23:59'],
            # ['2020-05-21 00:00','2020-05-21 23:59'],
            # ['2020-05-22 00:00','2020-05-22 23:59'],
            # ['2020-05-23 00:00','2020-05-23 23:59'],
            # ['2020-05-26 00:00','2020-05-26 23:59'],
            # ['2020-05-27 00:00','2020-05-27 23:59'],
            # ['2020-05-28 00:00','2020-05-28 23:59'],
            # ['2020-05-29 00:00','2020-05-29 23:59']]
# test_date=[['2020-05-16 00:00','2020-05-16 00:59']]#,

# test_date=[['2020-05-16 00:00','2020-05-16 23:59']]#,
            # ['2020-05-19 00:00','2020-05-19 23:59'],
            # ['2020-05-21 00:00','2020-05-21 23:59'],
            # ['2020-05-22 00:00','2020-05-22 23:59'],
            # ['2020-05-23 00:00','2020-05-23 23:59'],
            # ['2020-05-26 00:00','2020-05-26 23:59'],
            # ['2020-05-27 00:00','2020-05-27 23:59'],
            # ['2020-05-28 00:00','2020-05-28 23:59'],
            # ['2020-05-29 00:00','2020-05-29 23:59']]
# places=['Sun_Moon_Lake']#,'Shangdewen']#,'New_House','Chiayi',
#         'Dawu','Hengchun','Success','Sun_Moon_Lake','Taitung',
#         'Yuxi','Hualien','Beidou','Bao_Zhong','Chaozhou',
#         'News','Member_Hill','Yuli','Snow_Ridge','Shangdewen']
# ## load data
# places=['Banqiao','Keelung','Taipei','New_House','Chiayi',
#         'Dawu','Hengchun','Success','Sun_Moon_Lake','Taitung',
#         'Yuxi','Hualien','Beidou','Bao_Zhong','Chaozhou',
#         'News','Member_Hill','Yuli','Snow_Ridge','Shangdewen']
places=['Sun_Moon_Lake']
model_parameter = {"input_shape": [640, 640],
                   "output_shape": [640, 640],
                   "period": 6,
                   "predict_period": 6,
                   "filter": 36,
                   "kernel_size": [1, 1]}
  
radar1 = load_data_CREF(radar_echo_storage_path=radar_echo_storage_path, 
                        load_radar_echo_df_path=load_radar_echo_df_path,
                        input_shape=model_parameter['input_shape'],
                        output_shape=model_parameter['output_shape'],
                        period=model_parameter['period'],
                        predict_period=model_parameter['predict_period'],
                        places=places,
                        date_range=test_date,
                        test_date=test_date,
                        radar_echo_name_format=['CREF_010min.%Y%m%d.%H%M%S', 'CREF_020min.%Y%m%d.%H%M%S', 
                                                'CREF_030min.%Y%m%d.%H%M%S', 'CREF_040min.%Y%m%d.%H%M%S', 
                                                'CREF_050min.%Y%m%d.%H%M%S', 'CREF_060min.%Y%m%d.%H%M%S'])

if not load_radar_echo_df_path:
    radar1.exportRadarEchoFileList()
    # radar1.saveRadarEchoDataFrame(load_name_pkl='CREF_{}_640x640'.format(data_name))
    radar1.saveRadarEchoDataFrame(path=save_path,load_name_pkl='CREF_{}_640x640'.format(data_name))
# save_path = 'Result/research_CREF/research_new/Result/ConvLSTM_CREF/T6tT6/5m9d_512x512/'
# if not os.path.isdir(save_path):
#    os.makedirs(save_path)
#test_generator_CREF = radar1.generator('test', batch_size=32)

pred_x = []
pred_y = []

    # radar.saveRadarEchoDataFrame(load_name_pkl='CREF_5m9d_512x512')
pred_x = pd.read_pickle(save_path+'CREF_{}_640x640.pkl'.format(data_name))
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
pred_x.to_excel('pred_x_p20.xlsx')
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
    for j in range(6):#time_range:
        pred_list.append(pred_y[i][j][:][:])
        print("pred_list.shape",np.array(pred_list).shape)

        # print("list pred_y[i][j].shape",np.array(pred_y[i][time_range[j]]).shape)
        
print("pred_list.shape",np.array(pred_list).shape)

pred_list = np.array(pred_list).reshape(-1,6,640,640)#!
print("pred_list.shape=",pred_list.shape)
# pred_y=np.array(pred_list).reshape(1, 6,512*512)
# test_y=np.array(test_y[0]).reshape(1, 6,512*512)

from visualize.visualized_pred import visualized_area_with_map
pic0=pred_list[:,0,:,:].reshape(640,640)
pic1=pred_list[:,1,:,:].reshape(640,640)

pic2=pred_list[:,2,:,:].reshape(640,640)
pic3=pred_list[:,3,:,:].reshape(640,640)
pic4=pred_list[:,4,:,:].reshape(640,640)
pic5=pred_list[:,5,:,:].reshape(640,640)
'''
save_path_640 = save_path +"640/"
if not os.path.isdir(save_path_640):
   os.makedirs(save_path_640)
visualized_area_with_map(pic0, 'Sun_Moon_Lake', shape_size=[640,640], title='pred_list_0_640', savepath=save_path_640)
visualized_area_with_map(pic1, 'Sun_Moon_Lake', shape_size=[640,640], title='pred_list_1_640', savepath=save_path_640)
visualized_area_with_map(pic2, 'Sun_Moon_Lake', shape_size=[640,640], title='pred_list_2_640', savepath=save_path_640)
visualized_area_with_map(pic3, 'Sun_Moon_Lake', shape_size=[640,640], title='pred_list_3_640', savepath=save_path_640)
visualized_area_with_map(pic4, 'Sun_Moon_Lake', shape_size=[640,640], title='pred_list_4_640', savepath=save_path_640)
visualized_area_with_map(pic5, 'Sun_Moon_Lake', shape_size=[640,640], title='pred_list_5_640', savepath=save_path_640)
'''
real_frm1 = test_y.reshape(-1)
pred_frm1 = pred_list.reshape(-1)
#if not os.path.isdir(save_path):
#    os.makedirs(save_path)
# np.savetxt(save_path+'real_5m9d_p1_512x512_0824.txt', real_frm1)
# np.savetxt(save_path+'pred_5m9d_p1_512x512_0824.txt', pred_frm1)

print('test_y.shape = ', test_y.shape)

print('----------------------test_y------------------')
print(test_y)
print('type(test_y) = ', type(test_y))

test_y = np.array(test_y).reshape(-1,6,512,512)
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

from scipy import interpolate
x =np.linspace(0,1,640)
y =np.linspace(0,1,640)

f0 = interpolate.interp2d(x, y, pred_list_0, kind='linear')
f1 = interpolate.interp2d(x, y, pred_list_1, kind='linear')#linear
f2 = interpolate.interp2d(x, y, pred_list_2, kind='linear')
f3 = interpolate.interp2d(x, y, pred_list_3, kind='linear')
f4 = interpolate.interp2d(x, y, pred_list_4, kind='linear')
f5 = interpolate.interp2d(x, y, pred_list_5, kind='linear')

x1 =np.linspace(0,1,512)
y2 =np.linspace(0,1,512)
pred_list_0 = f0(x1, y2)
pred_list_1 = f1(x1, y2)
pred_list_2 = f2(x1, y2)
pred_list_3 = f3(x1, y2)
pred_list_4 = f4(x1, y2)
pred_list_5 = f5(x1, y2)

pred_list_0 = np.array(pred_list_0).reshape(512,512)
print("pred_list_0.shape=",pred_list_0.shape)
pred_list_1 = np.array(pred_list_1).reshape(512,512)
pred_list_2 = np.array(pred_list_2).reshape(512,512)
pred_list_3 = np.array(pred_list_3).reshape(512,512)
pred_list_4 = np.array(pred_list_4).reshape(512,512)
pred_list_5 = np.array(pred_list_5).reshape(512,512)

# pred_list_0=np.transpose((1,0))
# pred_list_1 = np.transpose((1,0))
# pred_list_2 = np.transpose((1,0))
# pred_list_3 = np.transpose((1,0))
# pred_list_4 = np.transpose((1,0))
# pred_list_5 = np.transpose((1,0))
# pred_list_2 = np.transpose(1,0)



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


# nan_data_test_y=np.isnan(np.array(test_y))
# test_y=np.array(test_y)
# test_y[nan_data_test_y] = 0
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
print("mse050=",mse060)
save_path_512 = save_path +"512/"
if not os.path.isdir(save_path_512):
   os.makedirs(save_path_512)
# import time
# time.sleep(6000)
from visualize.visualized_pred import visualized_area_with_map

visualized_area_with_map(pred_list_0.T, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_list_0', savepath=save_path_512)
visualized_area_with_map(pred_list_1.T, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_list_1', savepath=save_path_512)
visualized_area_with_map(pred_list_2.T, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_list_2', savepath=save_path_512)
visualized_area_with_map(pred_list_3.T, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_list_3', savepath=save_path_512)
visualized_area_with_map(pred_list_4.T, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_list_4', savepath=save_path_512)
visualized_area_with_map(pred_list_5.T, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_list_5', savepath=save_path_512)
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
# radar_pred = np.array(pred_list[0][0][:][:]).reshape(512,512)
# print("radar_pred=",radar_pred)
# radar_test_y = np.array(test_y[0][0][:][:]).reshape(512,512)
# print("radar_test_y=",radar_test_y)

img_mse=[]
img_mse_picture=[]
pred_np=[]
for i in range(6):
   img_mse.append(0)
   img_mse_picture.append(0)
   
'''
radar_test_y = test_y[:,:,3::4,1::4]#.reshape(-1)
# print("radar_test_y=",radar_test_y)
print("radar_test_y=",np.array(radar_test_y).shape)
# '''
# radar_test_y_all = test_y[:,:,3::4,1::4]#.reshape(-1)
# print("radar_test_y_all=",np.array(radar_test_y_all).shape)
# import sys
# radar_test_y_all = radar_test_y_all[:,:,:77,:77]
# print("radar_test_y_all=",np.array(radar_test_y_all).shape)
# # sys.exit()
# radar_pred_all = np.array(pred_list[:,:,2::5,1::5])
# print("radar_pred_all=",radar_pred_all.shape)

# radar_pred_all = radar_pred_all[:,:,:77,:77]

# print("radar_pred_all=",radar_pred_all.shape)
# sys.exit()
print(" ")
'''
radar_test_y = radar_test_y[:,:,:,:78]
print("radar_test_y=",np.array(radar_test_y).shape)

radar_test_y=radar_test_y.reshape(1,6,-1)
radar_test_y = radar_test_y[:][:][:6241].reshape(len(radar_test_y), 6, 79, 79)

radar_pred = np.array(pred_list[0][:][2::5][1::5]).reshape(len(pred_list),6,6288)
print("radar_pred=",radar_pred.shape)
radar_pred = radar_pred[:][:][:6241].reshape(len(radar_pred), 6, 79, 79)
'''
# mse = np.mean((pred_list-test_y)**2)

mse_all =(mse010+mse020+mse030+mse040+mse050+mse060)/6
fn = save_path + '{}_mse.txt'.format(data_name)
with open(fn,'a') as file_obj:
    file_obj.write('mse=' + str(mse_all)+'\n')
    # file_obj.write('mse_picture=' + str(mse_picture)+'\n')

rmse=np.sqrt(mse_all)
fn = save_path + '{}_rmse.txt'.format(data_name)
with open(fn,'a') as file_obj:
    file_obj.write('rmse=' + str(rmse)+'\n')
# print("len(pred_list)=",len(pred_list))
# for i in range(len(pred_list)):
#    save_path_picture =save_path+'i{}'.format(i) 
#    if not os.path.isdir(save_path_picture):
#       os.makedirs(save_path_picture)
#    for j in range(6):
#       radar_test_y  = np.array(test_y[i,j,2::4,1::4]).reshape(-1)

#     #   radar_test_y  = np.array(test_y[i][j][2::4][1::4]).reshape(-1)
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

    #   visualized_area_with_map(test_y[i][j][:][:], 'Sun_Moon_Lake', shape_size=[512,512], title='radar_test_y_i{}.j{}'.format(i,j), savepath=save_path_picture)
      
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
#sys.exit()
# csis = time.clock()

Color = ['#00FFFF', '#4169E1', '#0000CD', '#ADFF2F', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#9932CC']

## CSI comput
csi = []
# save_path = 'Result/ConvLSTM_CREF/output512/8_240000_0059_v3/{}_T6toT{}/'.format(str(places),model_parameter['predict_period'])


# save_path = 'Result/ConvLSTM_CREF/T6tT6/11_10/output_512x512/8_240010_8_240100_TEST/'

# if not os.path.isdir(save_path):
    # os.makedirs(save_path)
# print("radar_test_y_all=",np.array(radar_test_y_all).shape)

for period in range(model_parameter['predict_period']):
#    print('pred_list[:, period] = ', np.array(pred_list[:, period]).shape)
#    print('test_y[:, period] = ', np.array(test_y[:, period]).shape)
#   csi_eva = Verification(pred=pred_list[:, :].reshape(-1, 1), target=test_y[:, :].reshape(-1, 1), threshold=60, datetime='')
   csi_eva = Verification(pred=pered_all[period].reshape(-1, 1), target=test_y[:, period,:,:].reshape(-1, 1), threshold=60, datetime='')
   print("csi_eva.csi shape = ",np.array(csi_eva.csi).shape)
   print("np.nanmean(csi_eva.csi, axis=1) shape = ",np.nanmean(csi_eva.csi, axis=1).shape)
   csi.append(np.nanmean(csi_eva.csi, axis=1))
        # pred_list[:, period] =  (6, 11, 11)
        # test_y[:, period] =  (6, 11, 11)
        # csi_eva.csi shape =  (60, 726)
        # np.nanmean(csi_eva.csi, axis=1) shape =  (60,)
        # csi shape =  (6, 60)
csi = np.array(csi).reshape(6,60)
print("csi shape = ",csi.shape)
np.savetxt(save_path+'{}csi.csv'.format(data_name), csi.reshape(6,60), delimiter = ' ')
# np.savetxt('2darray.csv', csi, delimiter=',', fmt='%d')
# csi.tofile('foo.csv',sep=',')
#
#
## Draw peiod CSI 
'''
for threshold in range(5, 56, 5):
   fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
   ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
   plt.xlim(0, model_parameter['predict_period']+1)
   plt.ylim(-0.05, 1.0)
   plt.xlabel('Time/10min')
   plt.ylabel('CSI')
   my_x_ticks = np.arange(0, model_parameter['predict_period']+1, 1)
   plt.xticks(my_x_ticks)
   plt.title('Threshold {} dBZ'.format(threshold))
   plt.grid(True)
   plt.plot(np.arange(len(csi)+1), [np.nan] + list(csi[:, threshold-1]), 'o--')

   fig.savefig(fname=save_path+'Period_CSI_th{}.png'.format(threshold), format='png')
   plt.clf()
save_path
'''
## Draw peiod ALL CSI 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, model_parameter['predict_period']+1)
plt.ylim(-0.05, 1.0)
plt.xlabel('Time/10min')
plt.ylabel('CSI')
my_x_ticks = np.arange(0, model_parameter['predict_period']+1, 1)
plt.xticks(my_x_ticks)
plt.title('Threshold 5-55 dBZ')
plt.grid(True)
i = 0
for threshold in range(5, 56, 5):
   plt.plot(np.arange(len(csi)+1), [np.nan] + list(csi[:, threshold-1]), 'o--', label='{} dBZ'.format(threshold), color=Color[i])
   i = i + 1
#plt.legend(loc='lower right')

fig.savefig(fname=save_path+'Period_CSI_ALL2.png', format='png')
plt.clf()


## Draw thesholds CSI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, 60)
plt.ylim(-0.05, 1.0)
plt.xlabel('Threshold')
plt.ylabel('CSI')
plt.title('{}\nThresholds CSI'.format(csi_T))
plt.grid(True)

all_csi = []
for period in range(model_parameter['predict_period']):
   plt.plot(np.arange(csi.shape[1]), [np.nan] + csi[period, 1:].tolist(), 'o--', label='{} min'.format((period+1)*10))

plt.legend(loc='upper right')

fig.savefig(fname=save_path+'Thresholds_CSI.png', format='png')
plt.clf()


## Draw thesholds AVG CSI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, 60)
plt.ylim(-0.05, 1.0)
plt.xlabel('Threshold')
plt.ylabel('CSI')
plt.title('{}\nThresholds CSI'.format(csi_T))
plt.grid(True)

all_csi = []
plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), 'o--', label='AVG CSI')
   
plt.legend(loc='upper right')

fig.savefig(fname=save_path+'Thresholds_AVG_CSI.png', format='png')
plt.clf()
print("ok")
#csie = time.clock()
#
#alle = time.clock()
#
#print("load NWP time = ", loadNe - loadNs)
#print("load CREF time = ", loadCe - loadCs)
#print("All time = ", alle - alls)