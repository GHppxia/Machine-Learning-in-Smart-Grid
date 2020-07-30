import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import  precision_score
from sklearn.metrics import  recall_score
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
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

def pca(data,X):
    out_file_path = "train_afterpca.csv"
    scaled_data = preprocessing.scale(X)
    feature_num = 10
    pca = PCA(n_components=feature_num)
    pca.fit(scaled_data)
    # joblib.dump(pca,pca_path)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    pca_df = pd.DataFrame(pca_data, columns=labels)
    pca_df.to_csv(out_file_path, index=None)
    # pca_df = pd.DataFrame(pca_data, columns=labels)
    return  pca_df


if __name__ == '__main__':

    print("Start read data...")
    time_1 = time.time()

    data = pd.read_csv('temp_train.csv')
    data=op_inf(data)
    #data=frames_norm(data)
    features = data.drop('label',axis=1)
    labels = data.label
    data = pca(data,features)

    # 随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(data,labels, test_size=0.2, random_state=30)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))


    print('Start training...')
    # n_estimators表示要组合的弱分类器个数；
    # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
    clf = AdaBoostClassifier(n_estimators=4000,algorithm='SAMME.R')
    clf.fit(train_features,train_labels)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))


    print('Start predicting...')
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))
    accuracy= accuracy_score(test_labels, test_predict)
    precision=precision_score(test_labels,test_predict)
    recall=recall_score(test_labels,test_predict)
    print("precision: %.2f%%" % (precision*100.0))
    print("recall: %.2f%%" % (recall*100.0))
    print("accuarcy: %.2f%%" % (accuracy*100.0))