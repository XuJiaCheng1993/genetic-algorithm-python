# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:51:26 2018

__author__ = 'JiaChengXu'
__mtime__ = '2018/12/11'
"""
import numpy as np
import matplotlib.pyplot as plt
from Gentics import AdaptiveGeneticAlgorithm, GeneticAlgorithm
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.base import clone

# 定义KL散度函数
def kldiv(true, pred, class_label=None):
	if class_label is None:
		class_label = [0, 1]
	# 指标-KL散度
	kld = 0
	for i in class_label:
		p =  np.count_nonzero(true==i) / len(true)
		q = np.count_nonzero(pred[true==i]==i) / len(true)
		if q == 0:
			kld = np.inf
			break
		kld += p * np.log(p / q)
	return kld

# 定义适应度函数
def fitness_function(z, *args, **kwargs ):
    # 第一号染色体用于挑选特征
    pos = np.where(z[0]== 1)[0].astype('int').tolist()
    if not pos:
        X = kwargs['X'].copy()
    else:
        X = kwargs['X'][:, pos].copy()

    # 第二、三、四、五号染色体分别对应 SVM的 kernel、C、gamma 和random_state参数
    kernel_choice = ['linear', 'rbf', 'poly']
    estimator_params = {'kernel': kernel_choice[int(z[1])],
                        'C': z[2],
                        'gamma': z[4],
                        'random_state': int(z[3]),}
    clf = SVC(probability=True).set_params(**estimator_params)

    _, pred = SVMCV(clf, X, kwargs['y'], kwargs['n_kf'])

    # 以kl散度值作为适应度函数
    loss = kldiv(y, pred)
    return loss

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


## 参数
n_features = 16
n_kf = 5 # 交叉验证折数
chrome_num = 5  # 染色体个数，分别对5个该挑选的参数
chrome_range = [None , #1 染色体用于挑选特征
                [0.0, 1.99], #2 染色体用于挑选kerbel
                [0.25, 5.0], #2 染色体用于挑选C
                [0.25, 5.0], #3 染色体用于挑选ganma
                [0.0, 11.0] #4 染色体用于挑选random_state
                ]
population = 20  # 种群中的个体数
generation1 = 30  # 第一轮进化代数
generation2 = 10 # 第二轮进化代数
chrome_size = n_features  # 染色体的位数

## 生成特征
X, y = make_classification(n_samples=100, n_informative=5, n_redundant=1, random_state=223, n_features=n_features)

## 遗传算法求解
# 设置自适应遗传算法参数
aga = AdaptiveGeneticAlgorithm(chrome_num=chrome_num,
                              chrome_size=chrome_size,
                              population=population,
                              top_n=5,  # 结果给出的最优个体数
                              chrome_range=chrome_range,
                              tol=1e-4, # 停机精度
	                          elitism_radio=0.1, # 精英策略比例
                              cross_radio=0.8, # 交叉的个体比例
                              pc1=0.9, # 染色体交叉的最高概率
                              pc2=0.6, # 染色体交叉的最低概率
                              pm1=0.1, # 染色体变异的最高概率
                              pm2=0.001 # 染色体变异的最低概率
                              )

# 设置遗传算法参数
ga = GeneticAlgorithm(chrome_num=chrome_num,
                      chrome_size=chrome_size,
                      population=population,
                      top_n=5,  # 结果给出的最优个体数
                      chrome_range=chrome_range,
                      tol=1e-4, # 停机精度
	                  elitism_radio=0.1, # 精英策略比例
                      cross_radio=0.8, # 交叉的个体比例
                      mutation_prob=0.05 # 染色体变异概率
                      )

# 运行遗传算法
fitness_kwargs = {'X': X, 'y': y, 'n_kf': n_kf} # fitness函数额外传入的参数
aga.run(fitness_function=fitness_function,
        generation=generation1,
        function_params=fitness_kwargs)
result1 = aga.results_

# 运行结果不满意？热启动，接着上轮的结果继续进行迭代。
ga.run(fitness_function=fitness_function,
        generation=generation2,
        function_params=fitness_kwargs,
        chrome=aga.last_generation)

result2 = ga.results_

## 画图展示结果
chrome = result2['chrome'][0]
kernel_choice = ['linear', 'rbf', 'poly']

plt.figure()
plt.plot(np.hstack((result1['curve'], result2['curve'])), '-o', linewidth=2)
plt.plot(result1['curve'], '-s', linewidth=2)
plt.plot(0, result2['fitns'][0], 'w')
plt.plot(0, result2['fitns'][0], 'w')
plt.plot(0, result2['fitns'][0], 'w')
plt.plot(0, result2['fitns'][0], 'w')
plt.plot(0, result2['fitns'][0], 'w')
plt.legend(['Second iter', 'First iter','FeatureIndex:{0}'.format(np.where(chrome[0]==1)[0].astype(int)),
            'Kernel:{}'.format(kernel_choice[int(chrome[1])]), 'C:{}'.format(chrome[2]), 'gamma:{}'.format(chrome[3]),
            'random_state:{}'.format(int(chrome[4]))])
plt.xlabel('Generation', fontsize=16)
plt.ylabel('Fitness Value', fontsize=16)
plt.title('Evoluting Curve : Searching best params for SVM', fontsize=16)

plt.show()


