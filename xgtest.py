# -*- coding: utf-8 -*-
"""
###############################################################################
# 作者：wanglei5205
# 邮箱：wanglei5205@126.com
# 代码：http://github.com/wanglei5205
# 博客：http://cnblogs.com/wanglei5205
# 目的：xgboost基本用法
###############################################################################
"""
### load module
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import  pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from xgboost import plot_importance

attributes=['R1-PA1:VH', 'R1-PM1:V', 'R1-PA2:VH', 'R1-PM2:V', 'R1-PA3:VH', 'R1-PM3:V', 'R1-PA4:IH',
            'R1-PM4:I', 'R1-PA5:IH', 'R1-PM5:I', 'R1-PA6:IH', 'R1-PM6:I', 'R1-PA7:VH', 'R1-PM7:V',
            'R1-PA8:VH', 'R1-PM8:V', 'R1-PA9:VH', 'R1-PM9:V', 'R1-PA10:IH', 'R1-PM10:I', 'R1-PA11:IH',
            'R1-PM11:I', 'R1-PA12:IH', 'R1-PM12:I', 'R1:F', 'R1:DF', 'R1-PA:Z', 'R1-PA:ZH', 'R1:S',
            'R2-PA1:VH', 'R2-PM1:V', 'R2-PA2:VH', 'R2-PM2:V', 'R2-PA3:VH', 'R2-PM3:V', 'R2-PA4:IH',
            'R2-PM4:I', 'R2-PA5:IH', 'R2-PM5:I', 'R2-PA6:IH', 'R2-PM6:I', 'R2-PA7:VH', 'R2-PM7:V',
            'R2-PA8:VH', 'R2-PM8:V', 'R2-PA9:VH', 'R2-PM9:V', 'R2-PA10:IH', 'R2-PM10:I', 'R2-PA11:IH',
            'R2-PM11:I', 'R2-PA12:IH', 'R2-PM12:I', 'R2:F', 'R2:DF', 'R2-PA:Z', 'R2-PA:ZH', 'R2:S',
            'R3-PA1:VH', 'R3-PM1:V', 'R3-PA2:VH', 'R3-PM2:V', 'R3-PA3:VH', 'R3-PM3:V', 'R3-PA4:IH',
            'R3-PM4:I', 'R3-PA5:IH', 'R3-PM5:I', 'R3-PA6:IH', 'R3-PM6:I', 'R3-PA7:VH', 'R3-PM7:V',
            'R3-PA8:VH', 'R3-PM8:V', 'R3-PA9:VH', 'R3-PM9:V', 'R3-PA10:IH', 'R3-PM10:I', 'R3-PA11:IH',
            'R3-PM11:I', 'R3-PA12:IH', 'R3-PM12:I', 'R3:F', 'R3:DF', 'R3-PA:Z', 'R3-PA:ZH', 'R3:S', 'R4-PA1:VH',
            'R4-PM1:V', 'R4-PA2:VH', 'R4-PM2:V', 'R4-PA3:VH', 'R4-PM3:V', 'R4-PA4:IH', 'R4-PM4:I', 'R4-PA5:IH',
            'R4-PM5:I', 'R4-PA6:IH', 'R4-PM6:I', 'R4-PA7:VH', 'R4-PM7:V', 'R4-PA8:VH', 'R4-PM8:V', 'R4-PA9:VH',
            'R4-PM9:V', 'R4-PA10:IH', 'R4-PM10:I', 'R4-PA11:IH', 'R4-PM11:I', 'R4-PA12:IH', 'R4-PM12:I', 'R4:F',
            'R4:DF', 'R4-PA:Z', 'R4-PA:ZH', 'R4:S', 'control_panel_log1', 'control_panel_log2', 'control_panel_log3',
            'control_panel_log4', 'relay1_log', 'relay2_log', 'relay3_log', 'relay4_log', 'snort_log1', 'snort_log2',
            'snort_log3', 'snort_log4', 'label']
# 将data中的inf替换为该列的平均值
def op_inf(data):
    inf_df=np.isinf(data)
    data[inf_df]=np.nan
    data_mean=data.mean()
    for i in range(len(data_mean)):
        temp_ser=data.iloc[:,i].copy()  # .copy()为深拷贝，新建的对象元素不在指向源位置
        temp_ser.loc[np.isnan(temp_ser)]=data_mean[i]
        data.iloc[:, i] = temp_ser
    return data

def z_scoreNorm(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scoreNormed = (data - mean) / std
    #z_scoreNormed = z_scoreNormed.dropna(axis=1)
    return z_scoreNormed
#s=preprocessing.normalize()

#  min-max标准化
def min_maxNorm(data):
    save_M=data['label']
    #data = data.dropna(axis=0)
    scaler = preprocessing.MinMaxScaler()
    min_maxNorm = scaler.fit_transform(data)
    #print(min_maxNorm[:, -1].shape)
    min_maxNorm[:,-1]=save_M

    return min_maxNorm
# 将data归一化
def frames_norm(data):
    norm=min_maxNorm(data)
    normed_df=pd.DataFrame(norm,columns=attributes)
    return normed_df


### load datasets
train_path = "temp_train.csv"
data=pd.read_csv(train_path)
data = pd.read_csv(train_path)
data = op_inf(data)
data = frames_norm(data)
print(data)
Y = data.label
X = data.drop('label',axis=1)
### data split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state = 33)
### fit model for train data
model = XGBClassifier()
model.fit(x_train,y_train)
### make prediction for test data
y_pred = model.predict(x_test)
### model evaluate
accuracy = accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
print("precision: %.2f%%" % (precision*100.0))
print("recall: %.2f%%" % (recall*100.0))
print("accuarcy: %.2f%%" % (accuracy*100.0))
fig, ax = plt.subplots(figsize=(10, 15))
plot_importance(model, height=0.5, max_num_features=80, ax=ax)
plt.show()

