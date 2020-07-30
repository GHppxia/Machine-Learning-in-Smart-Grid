import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn import preprocessing
# Multinomial Naive Bayes Classifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import AdaBoostClassifier

def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.1)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=40)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=300)
    model.fit(train_x, train_y)
    return model
#XGboost Classfier
'''
def xgboot_classifier(train_x,train_y):
    from xgboost import XGBRegressor
    xgbt=XGBRegressor()
    xgbt.fit(train_x,train_y)
'''

# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

def xgboost_classifier(train_x,train_y):
    model = XGBClassifier()
    model.fit(train_x, train_y)
    return  model

def adaboost_classifier(train_x,train_y):
    model=AdaBoostClassifier(n_estimators=5000,algorithm='SAMME.R')
    model.fit(train_x,train_y)
    return model




def pca(data,X,Y):
    out_file_path = "train_afterpca.csv"
    scaled_data = preprocessing.scale(X)
    feature_num = 23
    pca = PCA(n_components=feature_num)
    pca.fit(scaled_data)
    # joblib.dump(pca,pca_path)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    pca_df = pd.DataFrame(pca_data, columns=labels)
    pca_df.to_csv(out_file_path, index=None)
    pca_df = pd.DataFrame(pca_data, columns=labels)
    pca_df['label'] = Y;
    return  pca_df

# SVM Classifier using cross validation
'''
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items())
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

# 处理csv文件,随机取出1000个样本并保存
#ef op_csv(set0,set1,output_name,size=1000):
    #   name0 = "./output/data_for_models/mean/results"+str(set0)+".csv"
    #   name1 = "./output/data_for_models/mean/results"+str(set1)+".csv"
    # data0 = pd.read_csv(name0);
    # data0.marker=0
    # data0=data0.sample(n=size).reset_index(drop=True)  # 随机采样
    # data1 = pd.read_csv(name1);data1.marker=1
    # data1=data1.sample(n=size).reset_index(drop=True)

    # data=data0.append(data1,ignore_index=True)
    #  data.rename(columns={'marker': 'label'}, inplace=True)
    #  data=data.sample(frac=1).reset_index(drop=True)
#  data.to_csv(output_name,index=None)
'''
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


#  Z-score标准化ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
def z_scoreNorm(data):
    save_M = data['label']
    # print(data)
    data=data.drop('label',axis=1)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scoreNormed = (data - mean) / std
    z_scoreNormed = z_scoreNormed.dropna(axis=1)
    z_scoreNormed=pd.DataFrame(z_scoreNormed)
    z_scoreNormed['label'] = save_M
    # print(z_scoreNormed)
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
    normed_df = pd.DataFrame(min_maxNorm, columns=attributes)
    # print(normed_df)

    return normed_df


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
    # data[inf_df] = 0
    return data


# 将data归一化
def frames_norm(data):
    norm=min_maxNorm(data)

    # print(normed_df)
    return norm

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


def op_data(data_path, train_frac, test_frac=1):

    data = pd.read_csv(train_path)
    data = op_inf(data)
    Y = data.label
    data=frames_norm(data)
    X = data.drop('label',axis=1)
    # data=pca(data,X,Y)
    # X = data.drop('label', axis=1)
    # print(X)
    # Y = data.label
    train_x, test_x, train_y, test_y=train_test_split(X, Y, test_size=0.1, random_state=42)

    '''
    train=train_data.loc[train_data['label'] == 0].head(train_num)
    train=train.append(train_data.loc[train_data['label'] == 1].head(train_num), ignore_index=True)
    train=train.sample(frac=1).reset_index(drop=True)
    test = test_data[:int(len(test_data) * test_frac)]

    print("训练集大小:", train.shape)
    print("测试集大小:", test.shape)
    train_y = train.label
    train_x = train.drop('label', axis=1)

    test_y = test.label
    test_x = test.drop('label', axis=1)
    '''
    return train_x, train_y, test_x, test_y


def ex_exp(result_path,rounds=3):
    data_path = "temp_train.csv"
    test_path = "temp_test.csv"
    test_classifiers = ['XGBT', 'RF', 'DT', 'SVM', 'GBDT']  # 去掉svmcv
    classifiers = {'XGBT':xgboost_classifier,
                   'AdaBT':adaboost_classifier,
                   'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'GBDT': gradient_boosting_classifier
                   }
    print('reading training and testing data...')

    av_result = pd.DataFrame(
        data=None, index=None, columns=['classifier', 'av_precision', 'av_recall', 'av_accuracy'])
    comp_result = pd.DataFrame(
        data=None, index=None, columns=['classifier', 'precision', 'recall', 'accuracy'])

    for j in range(len(test_classifiers)):
        classifier = test_classifiers[j]
        for i in range(rounds):
            train_x, train_y, test_x, test_y = op_data(data_path,1)
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)

            print('training took %fs!' % (time.time() - start_time))
            predict = model.predict(test_x)
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            accuracy = metrics.accuracy_score(test_y, predict)
            comp_result.loc[j * rounds + i] = [classifier, precision, recall, accuracy]
            print('precision: %.2f%%, recall: %.2f%%, '
                  'accuracy: %.2f%%' % (100 * precision, 100 * recall, 100 * accuracy))

    comp_result.to_csv(result_path)
    return comp_result

if __name__ == '__main__':
    train_path = "temp_train.csv"
    for e in range(4):
        comp_results_path = "results.csv"
        ex_exp(comp_results_path,rounds=3)
