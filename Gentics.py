#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'JiaChengXu'
__mtime__ = '2018/12/26'
"""

import numpy as np
from tqdm import tqdm


class Genetics(object):
	def __init__(self, chrome_num=1, chrome_size=4, population=20, top_n=5, chrome_range=[None, ], tol=1e-4,
	             cross_radio=0.8, elitism_radio=0.1):
		self.__chrome_size = chrome_size
		self.__chrome_num = chrome_num
		self.__population = population
		self.__chrome_range = chrome_range
		self.__top_n = top_n
		self.__tol = tol
		self.__cross_radio = cross_radio
		self.__elitism_radio = elitism_radio

	def _initial_ga(self):
		## 初始化染色体
		chrome = np.random.randint(2, size=(self.chrome_num, self.population, self.chrome_size), dtype='int8')
		return chrome

	def _decode(self, chrome):
		## 染色体解码
		chrome_range = self.__chrome_range
		maxim = 2 ** self.chrome_size - 1

		chrome_values = []
		for j in range(self.population):
			tmp = []
			for i in range(self.chrome_num):
				# 根据每一号染色体的范围 解码到对应数值
				if chrome_range[i] is None:
					tmp.append(chrome[i, j, :])
				else:
					[Lw_va, Up_va] = chrome_range[i]
					tmp += [(Up_va - Lw_va) * np.dot(chrome[i, j, :], self.chrome_size_wi) / maxim + Lw_va]
			chrome_values.append(tmp)
		return chrome_values

	def _calculate_fitness(self, chrome_values, fitness_function, fitness_function_params):
		# 计算适应度值
		fitns = np.zeros([self.population])
		for i, chrome_value in enumerate(chrome_values):
			fitns[i] = fitness_function(chrome_value, **fitness_function_params)
		return fitns

	def _best_n_chromes(self, chrome_values, fitns, top_n):
		# 最好的 n个 个体
		best_values_index = np.argsort(fitns)[:top_n]
		individual_best = []
		for i in best_values_index:
			individual_best.append(chrome_values[i])
		return fitns[best_values_index].copy(), individual_best

	def _select(self, chrome, fitns):
		# 精英策略
		elitism_num = int(self.population * self.elitism_radio)  # 精英策略挑选的染色体数目
		el_index = np.argsort(fitns)[:elitism_num]

		# 轮盘赌法
		roulette_num = self.population - elitism_num  # 轮盘赌法挑选的染色体数目
		rl_index = np.zeros([roulette_num])
		fitns_normalized = (np.max(fitns) - fitns) / (np.max(fitns) - np.min(fitns))  # 适应值越小占比越大
		cumsum_prob = np.cumsum(fitns_normalized) / np.sum(fitns_normalized)
		for i in range(roulette_num):
			for j, j_value in enumerate(cumsum_prob):
				if j_value >= np.random.rand(1):
					rl_index[i] = j
					break

		#  整合两者策略挑选的的 个体
		select_index = np.hstack((el_index, rl_index)).astype(int)
		return chrome[:, select_index, :].copy(), fitns[select_index].copy()

	def _evolute(self, code, mutation_prob, cross_prob=1.0):
		## 染色体进化

		# 交叉操作
		cross_num = int(self.population * self.cross_radio / 2)  # 计算需要交叉的次数

		if isinstance(cross_prob, (int, float)):
			cross_prob = [cross_prob, ] * cross_num

		code_cross = code.copy()
		for i in range(self.chrome_num):
			for j in range(cross_num):
				# 交叉
				prob = np.random.rand(1)
				if prob <= cross_prob[j]:
					cross_position = np.random.randint(int(0.5 * self.chrome_size))  # 单点交叉的位置
					for k in range(cross_position + 1, self.chrome_size):
						temp = code_cross[i, 2 * j, k]
						code_cross.itemset((i, 2 * j, k), code_cross[i, 2 * j + 1, k])
						code_cross.itemset((i, 2 * j + 1, k), temp)

		# 变异操作
		if isinstance(mutation_prob, (int, float)):
			mutation_prob = [mutation_prob, ] * self.population

		code_mutation = code_cross.copy()
		for i in range(self.chrome_num):
			for j in range(self.population):
				for k in range(self.chrome_size):
					# 随机值小于变异概率则进行编译操作
					prob = np.random.rand(1)
					if prob <= mutation_prob[j]:
						temp = code_mutation[i, j, k]
						code_mutation.itemset((i, j, k), int(1 - temp))
		return code_mutation.copy()



	def _update(self, current_generation_fitns, current_generation_individual,
	             global_best_fitns, global_best_individual, top_n):
		## 更新当前种群

		## 整合当代中却与全局最优种群
		fitns = np.hstack((current_generation_fitns, global_best_fitns))
		individual = current_generation_individual + global_best_individual
		## 挑选出最优的 n组 个体与他们的适应度值
		index = np.argsort(fitns)[:top_n]
		return fitns[index].copy(), [individual[i] for i in index]

	@property
	def chrome_num(self):
		return self.__chrome_num

	@property
	def chrome_size(self):
		return self.__chrome_size

	@property
	def population(self):
		return self.__population

	@property
	def chrome_size_wi(self):
		wi = np.zeros([self.chrome_size])
		for i in range(self.chrome_size):
			wi[i] = pow(2, self.chrome_size - 1 - i)
		return wi

	@property
	def top_n(self):
		return self.__top_n

	@property
	def tol(self):
		return self.__tol

	@property
	def cross_radio(self):
		return self.__cross_radio

	@property
	def elitism_radio(self):
		return self.__elitism_radio


class GeneticAlgorithm(Genetics):
	def __init__(self, chrome_num=1, chrome_size=4, population=20, top_n=5, chrome_range=[None, ], tol=1e-4,
	             elitism_radio=0.1, cross_radio=0.8, mutation_prob=0.05):
		super(GeneticAlgorithm, self).__init__(chrome_num=chrome_num, chrome_size=chrome_size, population=population,
		                                      top_n=top_n, chrome_range=chrome_range, tol=tol, cross_radio=cross_radio,
		                                       elitism_radio=elitism_radio)
		self.__mutation_prob = mutation_prob
		self.__alg_name = 'GenticAlg'

	def run(self, fitness_function, generation, function_params={}, chrome=None):
		## 运行遗传算法

		evoluting_curve = []
		# 计算第一代种群
		if chrome is None:
			chrome = self._initial_ga()
		else:
			print('热启动开始, {0}算法将从上次的迭代结果继续进行'.format(self.alg_name))

		chrome_values = self._decode(chrome)
		fitns = self._calculate_fitness(chrome_values, fitness_function, function_params)
		global_best_fitns, global_best_individual = self._best_n_chromes(chrome_values, fitns, self.top_n)

		for i in tqdm(range(generation)):
			# step1 选择
			chrome_selected, _ = self._select(chrome, fitns)
			# step2 染色体遗传
			chrome_evoluted = self._evolute(chrome_selected, self.mutation_prob)
			# step3 染色体解码
			chrome_values = self._decode(chrome_evoluted)
			# step4 计算适应度函数
			fitns = self._calculate_fitness(chrome_values, fitness_function, function_params)
			# 统计top_n的个体的适应度
			current_best_fitns, current_best_individual = self._best_n_chromes(chrome_values, fitns, self.top_n)

			evoluting_curve.append(np.min(fitns))
			# 满足条件则停机
			if np.sum(np.abs(global_best_fitns - current_best_fitns)) < self.tol:
				print('iter {0} times, best value {1}'.format(i + 1, global_best_fitns[0]))
				break
			# 更新
			del chrome
			chrome = chrome_evoluted
			global_best_fitns, global_best_individual = self._update(current_best_fitns, current_best_individual,
			                                                          global_best_fitns, global_best_individual,
			                                                          self.top_n)
		else:
			print('iter {0} times, best value {1}'.format(generation, global_best_fitns[0]))

		self.results_ = {'fitns': global_best_fitns,
		                 'chrome': global_best_individual,
		                 'curve': np.array(evoluting_curve)}
		self.__last_generation = chrome
		return self

	@property
	def alg_name(self):
		return self.__alg_name

	@property
	def mutation_prob(self):
		return self.__mutation_prob

	@property
	def last_generation(self):
		return self.__last_generation

class AdaptiveGeneticAlgorithm(Genetics):
	def __init__(self, chrome_num=1, chrome_size=4, population=20, top_n=5, chrome_range=[None, ], tol=1e-4,
	             elitism_radio=0.1, cross_radio=0.8, pc1=0.9, pc2=0.6, pm1=0.1, pm2=0.001):
		super(AdaptiveGeneticAlgorithm, self).__init__(chrome_num=chrome_num, chrome_size=chrome_size, population=population,
		                                      top_n=top_n, chrome_range=chrome_range, tol=tol, cross_radio=cross_radio,
		                                       elitism_radio=elitism_radio)

		self.__adaptive_cof = [pc1, pc2, pm1, pm2]
		self.__alg_name = 'AdaptiveGA'

	def _Adaptive_fetch_cross_and_mutation_prob(self, fitns):
		## 自适应的计算交叉概率和变异概率
		fmin, fave = np.min(fitns), np.mean(fitns)
		[pc1, pc2, pm1, pm2] = self.adaptive_cof
		cross_num = int(self.population * self.cross_radio / 2)
		cross_prob = np.zeros([cross_num])
		for i in range(cross_num):
			f_tmp = np.min(fitns[2 * i:2 * i + 2])
			if f_tmp <= fave:
				cross_prob[i] = pc1 - (pc1 - pc2) * (fave - f_tmp) / (fave - fmin)
			else:
				cross_prob[i] = self.adaptive_cof[0]

		mutation_prob = np.zeros([self.population])
		for i in range(self.population):
			if fitns[i] <= fave:
				mutation_prob[i] = pm1 - (pm1 - pm2) * (fave - fitns[i]) / (fave - fmin)
			else:
				mutation_prob[i] = pm1

		return cross_prob, mutation_prob

	def run(self, fitness_function, generation, function_params={}, chrome=None):
		## 运行遗传算法

		evoluting_curve = []
		# 计算第一代种群
		if chrome is None:
			chrome = self._initial_ga()
		else:
			print('热启动开始, {0}算法将从上次的迭代结果继续进行'.format(self.alg_name))

		chrome_values = self._decode(chrome)
		fitns = self._calculate_fitness(chrome_values, fitness_function, function_params)
		global_best_fitns, global_best_individual = self._best_n_chromes(chrome_values, fitns, self.top_n)

		for i in tqdm(range(generation)):
			# step1 选择
			chrome_selected, fitns = self._select(chrome, fitns)
			# step2 染色体遗传
			cross_prob, mutation_prob = self._Adaptive_fetch_cross_and_mutation_prob(fitns)
			chrome_evoluted = self._evolute(chrome_selected, mutation_prob, cross_prob)
			# step3 染色体解码
			chrome_values = self._decode(chrome_evoluted)
			# step4 计算适应度函数
			fitns = self._calculate_fitness(chrome_values, fitness_function, function_params)
			# 统计top_n的个体的适应度
			current_best_fitns, current_best_individual = self._best_n_chromes(chrome_values, fitns, self.top_n)

			evoluting_curve.append(np.min(fitns))
			# 满足条件则停机
			if np.sum(np.abs(global_best_fitns - current_best_fitns)) < self.tol:
				print('iter {0} times, best value {1}'.format(i + 1, global_best_fitns[0]))
				break
			# 更新
			del chrome
			chrome = chrome_evoluted
			global_best_fitns, global_best_individual = self._update(current_best_fitns, current_best_individual,
			                                                          global_best_fitns, global_best_individual,
			                                                          self.top_n)
		else:
			print('iter {0} times, best value {1}'.format(generation, global_best_fitns[0]))

		self.results_ = {'fitns': global_best_fitns,
		                 'chrome': global_best_individual,
		                 'curve': np.array(evoluting_curve)}
		self.__last_generation = chrome
		return self

	@property
	def alg_name(self):
		return self.__alg_name

	@property
	def adaptive_cof(self):
		return self.__adaptive_cof

	@property
	def last_generation(self):
		return self.__last_generation

class AnnealGeneticAlgorithm(AdaptiveGeneticAlgorithm):
	def __init__(self, chrome_num=1, chrome_size=4, population=20, top_n=5, chrome_range=[None, ], tol=1e-4,
	             cross_radio=0.8, elitism_radio=0.1, T0=90, speed=0.99):
		super(AnnealGeneticAlgorithm, self).__init__(chrome_num=chrome_num, chrome_size=chrome_size,
		                                             population=population,top_n=top_n, chrome_range=chrome_range,
		                                             tol=tol, cross_radio=cross_radio, elitism_radio=elitism_radio)
		self.__T0 = T0
		self.__speed = speed
		self.__alg_name = 'Anneal_GA'

	def __stoffa_normalize(self, fitns, T):
		fitns = np.power(np.e, fitns / T)
		fitns = fitns / np.sum(fitns)
		return fitns.copy()

	def __annel_accept(self, chrome_after, chrome_before, fitn_after, fitn_before, T):
		chrome_choice = np.zeros_like(chrome_after, dtype='int8')
		fitns_choice = np.zeros_like(fitn_after, dtype='float64')

		for j in range(self.population):
			if fitn_after[j] <= fitn_before[j]:
				chrome_choice[:, j, :] = chrome_after[:, j, :]
				fitns_choice[j] = fitn_after[j]
			else:
				annel_accept_prob = 1 / (1 + np.e ** ((fitn_before[j] - fitn_after[j]) / T))
				prob = np.random.rand(1)
				if annel_accept_prob < prob:
					chrome_choice[:, j, :] = chrome_after[:, j, :]
					fitns_choice[j] = fitn_after[j]
				else:
					chrome_choice[:, j, :] = chrome_before[:, j, :]
					fitns_choice[j] = fitn_before[j]
		return chrome_choice, fitns_choice

	def run(self, fitness_function, generation, function_params={}, chrome=None):
		## 运行遗传算法

		evoluting_curve = []
		# 计算第一代种群
		if chrome is None:
			chrome = self._initial_ga()
		else:
			print('热启动开始, {0}算法将从上次的迭代结果继续进行'.format(self.alg_name))

		chrome_values = self._decode(chrome)
		fitns = self._calculate_fitness(chrome_values, fitness_function, function_params)
		global_best_fitns, global_best_individual = self._best_n_chromes(chrome_values, fitns, self.top_n)


		for i in tqdm(range(generation)):
			# step1 计算当前温度T
			temperature = self.__T0 * self.__speed ** i
			# step2 计算stoffa适应度值
			stoffa_fitns = self.__stoffa_normalize(fitns, temperature)
			# step3 选择
			chrome_selected, stoffa_fitns = self._select(chrome, stoffa_fitns)
			# step4 染色体遗传
			cross_prob, mutation_prob = self._Adaptive_fetch_cross_and_mutation_prob(stoffa_fitns)
			chrome_evoluted = self._evolute(chrome_selected, mutation_prob, cross_prob)
			# step5 染色体解码
			chrome_values = self._decode(chrome_evoluted)
			# 6 计算适应度函数
			fitns_evoluted = self._calculate_fitness(chrome_values, fitness_function, function_params)
			# step6 模拟退火的accept
			chrome_accept, fitns_accept = self.__annel_accept(chrome_evoluted, chrome, fitns_evoluted, fitns, temperature)
			chrome_values = self._decode(chrome)
			# 统计top_n的个体的适应度
			current_best_fitns, current_best_individual = self._best_n_chromes(chrome_values, fitns_accept, self.top_n)
			evoluting_curve.append(np.min(fitns_accept))
			# 满足条件则停机
			if np.sum(np.abs(global_best_fitns - current_best_fitns)) < self.tol:
				print('iter {0} times, best value {1}'.format(i + 1, global_best_fitns[0]))
				break
			# 更新
			del chrome, fitns
			chrome, fitns = chrome_evoluted, fitns_evoluted
			global_best_fitns, global_best_individual = self._update(current_best_fitns, current_best_individual,
			                                                         global_best_fitns, global_best_individual,
			                                                         self.top_n)
		else:
			print('iter {0} times, best value {1}'.format(generation, global_best_fitns[0]))

		self.results_ = {'fitns': global_best_fitns, 'chrome': global_best_individual,
		                 'curve': np.array(evoluting_curve)}
		self.__last_generation = chrome
		return self

	@property
	def alg_name(self):
		return self.__alg_name

	def annel_cof(self):
		return [self.__T0, self.__speed]

	@property
	def last_generation(self):
		return self.__last_generation

# def fitness(x,*args,**kwargs):
# 	return x[0]**2 + x[1]**2 + 1
#
# if __name__ == '__main__':
# 	GA_CHROME_RANGE = [[-5.0, 5.0], [-5.0, 5.0]]
# 	ga = AnnealGeneticAlgorithm(chrome_num=2, chrome_size=12, population=20, chrome_range=GA_CHROME_RANGE, tol=1e-2)
# 	res = ga.run(fitness, 200).results_
# 	res2 = ga.run(fitness, 200, chrome=ga.last_generation).results_
#
# 	import matplotlib.pyplot as plt
# 	plt.figure()
# 	plt.plot(np.hstack((res['curve'],res2['curve'])))
# 	plt.show()
#
# 	print('this is ga for python')