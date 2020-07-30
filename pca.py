import pandas as pd
import numpy as np
#from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import preprocessing
import  matplotlib.pyplot as plt
# pd.set_option('display.max_row', None)
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

file_path = "temp_train.csv"
#pca_path = "D:/Users/oyyk/PycharmProjects/F_G_P/scripts/train/few_shot/results/pca_path.pt"
out_file_path = "train_afterpca.csv"
data = pd.read_csv(file_path)
data=op_inf(data)
data=frames_norm(data)
print([column for column in data])

x = data.drop(columns=['label'])
y = data.label
scaled_data = preprocessing.scale(x)
#scaled_data=frames_norm(data)
feature_num =65
pca = PCA(n_components=feature_num)
pca.fit(scaled_data)
#joblib.dump(pca,pca_path)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]
pca_df = pd.DataFrame(pca_data, columns=labels)
pca_df['label'] = y

pca_df.to_csv(out_file_path, index=None)
# pca_df = pd.DataFrame(pca_data, columns=labels)
print(pca_df)
plt.scatter(pca_df.PC1, pca_df.PC2)
print(per_var)
plt.title('My PCA Graph')
plt.xlabel('PC1- {0}%'.format(per_var[1]))
plt.ylabel('PC2- {0}%'.format(per_var[2]))
#
#     # 颜色集合，不同标记的样本染不同的颜色
# colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
#
# for label ,color in zip( np.unique(y),colors):
#             position=y==label
#             plt.scatter(pca_data[position,0],pca_data[position,1],
#             color=color)
# for i in range(pca_df.shape[0]):
#     plt.annotate(i, (pca_df.PC1.iloc[i],pca_df.PC2.iloc[i]))
plt.show()