#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'JiaChengXu'
__mtime__ = '2018/12/11'
"""

from Gentics import GeneticAlgorithm, AdaptiveGeneticAlgorithm, AnnealGeneticAlgorithm
import matplotlib.pyplot as plt


# 求二元函数 x^2 + y^2 - 2x - 1的极小值, 其中x, y∈[-5, 5]

# 定义适应度函数：f(x, y) = x^2 + y^2 - 2x - 1
def fitness_function(z,*args,**kwargs):
    return z[0]**2 + z[1]**2 - 2*z[0] - 1

chrome_num = 2 #染色体个数，分别对应两个自变量x和y
chrome_range = [[-5.0,5.0], # x的取值范围
                [-5.0,5.0]] # y的取值范围
chrome_size = 12 # 染色体的位数
population = 20 # 种群中的个体数
generation = 50 # 进化代数


## 遗传算法求解
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
ga.run(fitness_function=fitness_function,
       generation=generation)

result_ga = ga.results_


## 自适应遗传算法求解
# 设置遗传算法参数
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
# 运行遗传算法
aga.run(fitness_function=fitness_function,
       generation=generation)

result_aga = aga.results_

## 退火遗传算法求解
# 设置遗传算法参数
anga = AnnealGeneticAlgorithm(chrome_num=chrome_num,
                            chrome_size=chrome_size,
                            population=population,
                            top_n=5,  # 结果给出的最优个体数
                            chrome_range=chrome_range,
                            tol=1e-4, # 停机精度
	                        elitism_radio=0.1, # 精英策略比例
                            cross_radio=0.8, # 交叉的个体比例
                            T0=90, # 退火初始温度
                            speed=0.99, # 退火速率
                            )
# 运行遗传算法
anga.run(fitness_function=fitness_function,
       generation=generation)

result_anga = anga.results_

## 展示结果
plt.figure()
plt.plot(result_ga['curve'], '-*', linewidth=2)
plt.plot(result_aga['curve'], '-s', linewidth=2)
plt.plot(result_anga['curve'], '-o', linewidth=2)
plt.legend([ga.alg_name, aga.alg_name, anga.alg_name], fontsize=16)
plt.xlabel('Generation', fontsize=16)
plt.ylabel('Fitness Value', fontsize=16)
plt.title('Evoluting Curve - function : x^2 + y^2 - 2x - 1', fontsize=16)
plt.show()