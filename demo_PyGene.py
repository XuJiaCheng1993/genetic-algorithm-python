#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'JiaChengXu'
__mtime__ = '2019/3/23'
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

from PyGene import GeneticAlgorithm, AnnealGeneticAlgorithm, AdaptiveGeneticAlgorithm

# 定义SVM的交叉验证函数
def SVMCV(estimator, X, y, n_kf):
    kf = KFold(n_splits=n_kf).split(y)
    score = np.zeros_like(y, dtype='float64')
    pred = np.zeros_like(score)
    for i, (train_index, test_index) in enumerate(kf):
        clf = clone(estimator)
        clf.fit(X[train_index, :], y[train_index])
        score[test_index] = clf.predict_proba(X[test_index, :])[:, 1]
        pred[test_index] = clf.predict(X[test_index, :])
    return score, pred

def fitness_function(chrome, *args, **kwargs ):
    # 第一号染色体用于挑选特征
    pos = np.where(chrome[0]== 1)[0].astype('int').tolist()
    if not pos:
        X = kwargs['X'].copy()
    else:
        X = kwargs['X'][:, pos].copy()

    # 第二、三、四、五号染色体分别对应 SVM的 kernel、C、gamma 和random_state参数
    kernel_choice = ['linear', 'rbf', 'poly']
    estimator_params = {'kernel': kernel_choice[int(chrome[1])],
                        'C': chrome[2],
                        'gamma': chrome[3],
                        'random_state': int(chrome[4]),}
    clf = SVC(probability=True).set_params(**estimator_params)

    score, _ = SVMCV(clf, X, kwargs['y'], kwargs['n_kf'])

    # 以1-AUC值作为适应度函数
    loss = 1 - roc_auc_score(kwargs['y'], score)
    return loss


n_features = 20
n_kf = 5
## 生成特征
X, y = make_classification(n_samples=100, n_informative=5, n_redundant=1, random_state=223, n_features=n_features)

chrome_paramts = [(n_features, None), # 第1号染色体, 挑选特征
                  (8, [0.0, 2.99]), # 第2号染色体， kernel
                  (12, [0.5, 5.0]), # 第3号染色体, C
                  (12, [0.5, 5.0]), # 第4号染色体, gamma
                  [8, [0, 10]]] # 第5号染色体, random_state

Aga = AdaptiveGeneticAlgorithm(population_num=20, chrome_kws=chrome_paramts, top_n=10)
Aga.set_evolute_params(dict(pm1=0.2, pm2=0.01))
fitness_kwargs = {'X': X, 'y': y, 'n_kf': n_kf} # fitness函数额外传入的参数
Aga.run(func=fitness_function, func_kwargs=fitness_kwargs, tol=-1, generation=10)
result = Aga.result

# 接着Aga的结果继续运行
ga = GeneticAlgorithm(population_num=20, chrome_kws=chrome_paramts, top_n=10)
ga.run(population=Aga.population, func=fitness_function, func_kwargs=fitness_kwargs, tol=-1, generation=10)
result2 = ga.result

import matplotlib.pyplot as plt
plt.figure()
plt.grid()
plt.plot(np.hstack((result['evolute_curve'], result2['evolute_curve'])), '-rs', linewidth=2)
plt.plot(result['evolute_curve'], '-gs', linewidth=2)
plt.xticks(range(0, 20, 4), [f + 1 for f in range(0, 20, 4)], fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Generation', fontsize=16)
plt.ylabel('Fitness Value (1 - AUC)', fontsize=16)
plt.tight_layout()
plt.show()