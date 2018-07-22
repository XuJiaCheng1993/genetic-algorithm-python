# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:51:26 2018

@author: 徐嘉诚
"""
import numpy as np
import pandas as pd
from GeneticAlgorithm import Adaptive_Genetic_Algorithm
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def fitness_function(z,*args,**kwargs ):
    pos = np.where(z[0]== 1)[0].astype('int')
    feature = np.array(kwargs['feature'])
    feature = feature[pos].tolist()
    if len(pos) == 0:
        feature = kwargs['feature']
    score,pred = SVMCV(kwargs['train'], feature, kwargs['cv_nums'],kwargs['label_name'],
                C=z[-2],random_state=int(z[-1]))   
    true = kwargs['train'][kwargs['label_name']].as_matrix()
    acc = accuracy_score(true,pred)
    loss = 1- acc
    return loss

def SVMCV(dataframe,feature,n_kf,label_name,C=1,random_state=0):
    kf = KFold(n_splits=n_kf).split(dataframe)
    score = np.zeros([dataframe.shape[0]])
    pred = np.zeros_like(score)
    for i, (train_index, test_index) in enumerate(kf):
        lsvc = SVC(kernel='linear', C=C,random_state=random_state,probability=True)
        lsvc.fit(dataframe[feature].iloc[train_index, :],dataframe[label_name].iloc[train_index])
        score[test_index] = lsvc.predict_proba(dataframe[feature].iloc[test_index,:])[:,1]
        pred[test_index] = lsvc.predict(dataframe[feature].iloc[test_index,:])
    return score,pred

if __name__ == '__main__':
    ## 参数
    N_FEATURES = 10
    CV_NUMS = 3
    LABEL_NAME = 'label'
    GA_CHROME_RANGE = [None,[0.25,3.0],[0.0,5.0]]
    GA_GENERATION = 200
    GA_TOP_N_MODEL = 10
    
    ## 生成特征
    x, y = make_classification(n_samples=100, n_informative=5,n_redundant=1, random_state=223,n_features=N_FEATURES)
    feature_name = ['Fea'+str(f) for f in range(N_FEATURES)]
    data = pd.DataFrame(data = np.hstack((y.reshape(-1,1),x)),
                        columns = [LABEL_NAME,] + feature_name)
    
    # 自适应遗传算法
    fitness_kwargs = {'train':data,
                      'feature':feature_name,
                      'cv_nums':CV_NUMS,
                      'label_name':LABEL_NAME}
    
    ga = Adaptive_Genetic_Algorithm(chrome_size = len(feature_name), 
                                    chrome_num = len(GA_CHROME_RANGE),
                                    Range = GA_CHROME_RANGE,
                                    generation = GA_GENERATION, 
                                    top_n = GA_TOP_N_MODEL)
    
    result = ga.run(fitness_function,fitness_kwargs).results_
    
    print('best fitness:{0}'.format(result['minimal'][0]))
    print('best feature comb: {0}'.format(np.array(feature_name)[np.where((result['chrome'][0][0])==1)[0]].tolist() ) ) 
    print('SVC best C:{0} , SVC best randomstate:{1}'.format(result['chrome'][0][1],int(result['chrome'][0][2])) )