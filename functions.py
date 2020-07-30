import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing
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

#  场景分类
def scenarios():
    # Natural Events,No Events,
    # A1_Data Injection,A2_Remote Tripping,A3Relay Setting Change
    s1 = np.array(np.append(np.arange(1, 7), np.array([13, 14])))
    s2 = np.array([41])
    s3 = np.array(np.arange(7, 13))
    s4 = np.array(np.arange(15, 21))
    s5 = np.array(np.append(np.arange(21, 31), np.arange(35, 41)))
    scenes = np.array([s1, s2, s3, s4, s5])
    return scenes


#  Z-score标准化
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


# 将csv文件合并成data
def csv_joint(num_li):
    data = pd.DataFrame(data=None,columns=attributes)
    for i in num_li:
        name="../../data/morris/Muticlass_csv_"+str(i)+".csv"
        temp=pd.read_csv(name)
        temp.rename(columns={'marker': 'label'}, inplace=True)
        data=data.append(temp,ignore_index=True)
        data=data.astype(float)

    return data


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

# 将data归一化
def frames_norm(data):
    norm=min_maxNorm(data)
    normed_df=pd.DataFrame(norm,columns=attributes)
    return normed_df


# 创建空的frames[]
def create_frames(data):
    frames=[]
    frame = pd.DataFrame(data=None, columns=data.columns)
    for i in range(5):
        frames.append(frame)
    return frames



# 将data按照scenes分类
'''
def depart_set(data,scenes):
    frames=create_frames(data)
    for i in range(data.shape[0]):
        label=data['label'][i]
        print(i,end=" ")
        if label in scenes[0]:
            print("detected in 0")
            frames[0] = frames[0].append(data.loc[i])
        elif label in scenes[1]:
            print("detected in 1")
            frames[1] = frames[1].append(data.loc[i])
        elif label in scenes[2]:
            print("detected in 2")
            frames[2] = frames[2].append(data.loc[i])
        elif label in scenes[3]:
            print("detected in 3")
            frames[3] = frames[3].append(data.loc[i])
        elif label in scenes[4]:
            print("detected in 4")
            frames[4] = frames[4].append(data.loc[i])
    frames = [frames[0], frames[1], frames[2], frames[3], frames[4]]
    return frames


def cut_frames(frames,num=40):
    for i in range(len(frames)):
        group_name="./data/groups/group"+str(i)+"/"
        temp_data=frames[i]
        gap=temp_data.shape[0]//num
        print(gap)
        for j in range(num):
            start = j * gap
            temp_name = group_name + str(j) + ".csv"
            temp_frame = temp_data[start:start + gap]
            temp_frame.to_csv(temp_name, index=None)
'''

