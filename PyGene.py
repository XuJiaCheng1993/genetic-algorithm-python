#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'JiaChengXu'
__mtime__ = '2019/3/7'
"""
import numpy as np
import copy
from tqdm import tqdm

class Chrome(object):
	def __init__(self, bits=8, range_=None, value=None):
		self.__bits = bits
		self.__bitweights = np.array([2 ** i for i in range(bits)])
		self.__maxvalue = 2 ** bits - 1
		self.__range = range_
		if range_ is None or value is None:
			self.__chrome = np.random.randint(2, size=(bits), dtype='bool')
		else:
			self.__chrome = self.code(range_, value)

	def code(self, range_, value):
		value = int((value - range_[0]) / (range_[1] - range_[0]) * self.__maxvalue)
		i, chrome = 0, np.zeros([self.__bits], dtype='bool')
		while value > 0:
			value, yu = divmod(value, 2)
			chrome[i] = bool(yu)
			i += 1
		return chrome

	def mutate(self, probability=0.05):
		values = np.random.random([self.__bits])
		for i, v in enumerate(values):
			if v < probability:
				self.__chrome[i] = ~self.__chrome[i]

	def decode(self, range_=None):
		if range_ is None:
			return self.__chrome
		else:
			value = self.__chrome.dot(self.__bitweights) / self.__maxvalue
		return  value * (range_[1] - range_[0]) + range_[0]

	@classmethod
	def cross(cls, chrome_1, chrome_2, probability=0.8, st=0.0, ed=1.0):
		assert 0.0 <= st and st <= ed and ed <= 1.0
		st, ed = int(st * chrome_1.bits), int(ed * chrome_1.bits)
		if np.random.random(1)[0] < probability:
			temp = copy.deepcopy(chrome_1.chrome)
			chrome_1.chrome[st:ed] = copy.deepcopy(chrome_2.chrome[st:ed])
			chrome_2.chrome[st:ed] = temp[st:ed]
		return chrome_1, chrome_2

	@property
	def chrome(self):
		return self.__chrome

	@property
	def bits(self):
		return self.__bits

	@property
	def chrome_value(self):
		return self.decode(self.__range)


class Gene(object):
	def __init__(self, chrome_kws=None,):
		if chrome_kws is None:
			self.__gene = [Chrome(), ]
		else:
			self.__gene = [Chrome(*ch_kws) for ch_kws in chrome_kws]
		self.__gene_score = 0

	def update_gene_score(self, func, func_args=(), func_kwargs={}):
		self.__gene_score = func([g.chrome_value for g in self.__gene], *func_args, **func_kwargs)

	def mutate(self, probability=0.05):
		for i, chrome in enumerate(self.__gene):
			chrome.mutate(probability)

	@classmethod
	def cross(cls, gene_1, gene_2, probability=0.8, st=0.0, ed=1.0):
		for i in range(gene_1.gene_num):
			gene_1.gene[i], gene_2.gene[i] = Chrome.cross(gene_1.gene[i], gene_2.gene[i], probability, st, ed)
		return gene_1, gene_2

	@property
	def gene(self):
		return self.__gene

	@property
	def gene_num(self):
		return self.__gene.__len__()

	@property
	def gene_score(self):
		return self.__gene_score


class Population(object):
	def __init__(self, population_num=20, chrome_kws=None, func=None, func_args=(), func_kwargs={}):
		self.__population_num = population_num
		self.__population = [Gene(chrome_kws) for i in range(population_num)]
		self.__func = func
		self.__func_args = func_args
		self.__func_kwargs = func_kwargs
		self.__value = np.zeros([population_num])
		if func is not None:
			self.update_fitness_value()

	def fitness_function(self, func, func_args=(), func_kwargs={}):
		self.__func = func
		self.__func_args = func_args
		self.__func_kwargs = func_kwargs
		self.update_fitness_value()

	def update_fitness_value(self, compute=True):
		for i, gene in enumerate(self.__population):
			if compute:
				gene.update_gene_score(self.__func, self.__func_args, self.__func_kwargs)
			self.__value[i] = gene.gene_score

	def mutate(self, probability=0.05, func=None, func_args=(), func_kwargs={}):
		for gene in self.__population:
			if func is not None:
				probability = func(gene.gene_score, *func_args, **func_kwargs)
			gene.mutate(probability)

	def cross(self,  probability=0.8, gene_st=0.0, gene_ed=1.0, chrome_st=0.0, chrome_ed=1.0, func=None, func_args=(), func_kwargs={}):
		assert 0.0 <= gene_st and gene_st <= gene_ed and gene_ed <= 1.0
		st, ed = int(gene_st * self.__population_num), int(gene_ed * self.__population_num)

		index = np.random.permutation(range(st, ed))
		num, flag = divmod(len(index), 2)
		temp = copy.deepcopy(self.__population[:st] + self.__population[:-ed])
		if flag == 1:
			temp += [copy.deepcopy(self.__population[index[-1]]), ]
		for i in range(num):
			g1o, g2o = self.__population[index[2 * i]], self.__population[index[2 * i + 1]]
			if func is not None:
				probability = func(g1o.gene_score, g2o.gene_score, *func_args, **func_kwargs)
			g1, g2 = Gene.cross(g1o, g2o, probability, chrome_st, chrome_ed)
			temp += [g1, g2]
		self.__population = temp

	def swap(self, i=0, j=0, index=None,):
		if index is not None:
			self.__population = np.array(self.__population)[index].tolist()
		else:
			self.__population[i], self.__population[j] = self.__population[j], self.__population[i]
		self.update_fitness_value(False)


	@property
	def population(self):
		return self.__population

	@population.setter
	def population(self, pop):
		if sum(1 for f in pop if isinstance(f, Gene)) == len(pop):
			self.__population = pop

	@property
	def population_score(self):
		return self.__value

	@property
	def population_num(self):
		return self.__population.__len__()

class Genetics(object):
	def __init__(self, population_num=20, chrome_kws=None, cross_radio=0.8, elitism_radio=0.2,mutate_prob=0.01,
	             top_n=5,):
		self.Pop = Population(population_num, chrome_kws)
		self.globalPop = Population(top_n, chrome_kws)
		self.__population_num = self.Pop.population_num
		self.evoluate_params = {}
		self.__top_n = top_n
		self._evolute_curve = np.array([])
		self.alg_name = 'GenticAlg'
		self.iter = 0
		self.set_evolute_params(dict(cross_radio=cross_radio, elitism_radio=elitism_radio, mutate_prob=mutate_prob))

	def set_evolute_params(self, params_dict):
		for key, value in params_dict.items():
			self.evoluate_params.update({key: value})

	def _select(self):
		fitns, pop_num, el_radio = self.Pop.population_score, self.__population_num, self.evoluate_params['elitism_radio']
		# 精英策略
		elitism_num = int(pop_num * el_radio)  # 精英策略挑选的染色体数目
		el_index = np.argsort(fitns)[:elitism_num]

		# 轮盘赌法
		roulette_num = pop_num - elitism_num  # 轮盘赌法挑选的染色体数目
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
		self.Pop.swap(index = select_index)

	def _evolute(self):
		pass

	def _update(self):
		gbl_gene = self.globalPop.population
		gbl_fitn = self.globalPop.population_score

		loc_fitn = self.Pop.population_score
		loc_gene = self.Pop.population

		fitn = np.hstack((gbl_fitn, loc_fitn))
		pop = gbl_gene + loc_gene
		index = np.argsort(fitn)[:self.__top_n]

		self.globalPop.population = np.array(pop)[index].tolist()
		self.globalPop.update_fitness_value(False)


	def _downtime(self, tol):
		gbl_fitn = self.globalPop.population_score
		loc_fitn =  np.sort(self.Pop.population_score)[:self.__top_n]
		if np.sum((loc_fitn - gbl_fitn) ** 2) < tol:
			return False
		else:
			return True

	def run(self, func, population=None, func_args=(), func_kwargs={}, generation=50, tol=1e-4):
		if population is not None:
			self.Pop = population
			evolute_curve_1 = self.evolute_curve
		else:
			evolute_curve_1 = np.array([])

		self.iter = 0
		self.Pop.fitness_function(func, func_args, func_kwargs)
		self.globalPop.fitness_function(func, func_args, func_kwargs)

		evolute_curve_2 = np.zeros([generation])
		self._update()
		pbar = tqdm(total=generation, desc=self.alg_name)
		while self._downtime(tol) and self.iter < generation:
			self._select()
			self._evolute()
			evolute_curve_2[self.iter] = np.mean(self.globalPop.population_score)
			self._update()
			self.iter += 1
			pbar.update(1)
		pbar.close()

		self._evolute_curve = np.hstack((evolute_curve_1, evolute_curve_2[:self.iter]))

	def __str__(self):
		return self.alg_name

	@property
	def population(self):
		return self.Pop

	@property
	def best_gene(self):
		return self.globalPop.population

	@property
	def evolute_curve(self):
		return self._evolute_curve

	@property
	def result(self):
		gene = [[[chrome.chrome, chrome.chrome_value] for chrome in gene.gene] for gene in self.best_gene ]
		res = dict(evolute_curve = self.evolute_curve,
		           gene_coding = [[c[0] for c in g] for g in gene],
		           gene_decoding = [[c[1] for c in g]for g in gene],
		           fitness = self.globalPop.population_score)
		return res

class GeneticAlgorithm(Genetics):
	def __init__(self, *args, **kwargs):
		super(GeneticAlgorithm, self).__init__(*args, **kwargs)
		self.alg_name = 'GenticAlg'

	def _evolute(self):
		cross_radio, mutate_prob = self.evoluate_params['cross_radio'], self.evoluate_params['mutate_prob']

		self.Pop.cross(probability=1,
		               gene_st=1 - cross_radio,
		               chrome_st=0.5 * np.random.random(1)[0],
		               chrome_ed=0.5 * np.random.random(1)[0] + 0.5)
		self.Pop.mutate(mutate_prob)
		self.Pop.update_fitness_value()


class AdaptiveGeneticAlgorithm(Genetics):
	def __init__(self, pc1=0.9, pc2=0.6, pm1=0.1, pm2=0.001, *args, **kwargs):
		super(AdaptiveGeneticAlgorithm, self).__init__(*args, **kwargs)
		self.alg_name = 'AdaptiveGA'
		self.set_evolute_params(dict(pc1=pc1, pc2=pc2, pm1=pm1, pm2=pm2))

	@staticmethod
	def get_cross_prob(fitn_1, fitn_2, f_ave, f_min, pc1, pc2):
		f_tmp = min(fitn_1, fitn_2)
		if f_tmp <= f_ave:
			cross_prob = pc1 - (pc1 - pc2) * (f_ave - f_tmp) / (f_ave - f_min)
		else:
			cross_prob = pc1
		return cross_prob

	@staticmethod
	def get_mutate_prob(fitn, f_ave, f_min, pm1, pm2):
		if fitn <= f_ave:
			mutate_prob = pm1 - (pm1 - pm2) * (f_ave - fitn) / (f_ave - f_min)
		else:
			mutate_prob = pm1
		return mutate_prob

	def _evolute(self):
		fitns, pop_num = self.Pop.population_score, self.Pop.population_num
		## 自适应的计算交叉概率和变异概率
		fmin, fave = np.min(fitns), np.mean(fitns)
		[cr, pc1, pc2, pm1, pm2] = [self.evoluate_params[key] for key in ['cross_radio', 'pc1', 'pc2', 'pm1', 'pm2']]

		self.Pop.cross(gene_st= 1 - cr,
		               chrome_st=0.5 * np.random.random(1)[0],
		               chrome_ed=0.5 * np.random.random(1)[0] + 0.5,
		               func= AdaptiveGeneticAlgorithm.get_cross_prob,
		               func_args=(fave, fmin, pc1, pc2),)

		self.Pop.mutate(func= AdaptiveGeneticAlgorithm.get_mutate_prob,
		                func_args=(fave, fmin, pm1, pm2), )

		self.Pop.update_fitness_value()

class AnnealGeneticAlgorithm(Genetics):
	def __init__(self, T0=90, speed=0.99, *args, **kwargs):
		super(AnnealGeneticAlgorithm, self).__init__(*args, **kwargs)
		self.alg_name = 'AnnealGA'
		self.set_evolute_params(dict(T0=T0, speed=speed,))

	def __stoffa_normalize(self, fitns, T):
		fitns = np.power(np.e, fitns / T)
		fitns = fitns / np.sum(fitns)
		return fitns.copy()

	def _evolute(self):
		T = self.evoluate_params['T0'] * self.evoluate_params['speed'] ** self.iter
		cross_radio, mutate_prob = self.evoluate_params['cross_radio'], self.evoluate_params['mutate_prob']

		self.Pop.cross(probability=1,
		               gene_st=1 - cross_radio,
		               chrome_st=0.5 * np.random.random(1)[0],
		               chrome_ed=0.5 * np.random.random(1)[0] + 0.5)
		self.Pop.update_fitness_value()

		beforePop = copy.deepcopy(self.Pop)
		self.Pop.mutate(mutate_prob)
		self.Pop.update_fitness_value()

		for i in range(beforePop.population_num):
			if self.Pop.population_score[i] > beforePop.population_score[i]:
				annel_accept_prob = 1 / (1 + np.e ** ((beforePop.population_score[i] - self.Pop.population_score[i]) / T))
				if annel_accept_prob >= np.random.rand(1)[0]:
					self.Pop.population[i] = beforePop.population[i]
		self.Pop.update_fitness_value(False)


	def _select(self):
		T = self.evoluate_params['T0'] * self.evoluate_params['speed'] ** self.iter
		fitns, pop_num, el_radio = self.Pop.population_score, self.Pop.population_num, self.evoluate_params['elitism_radio']
		fitns = self.__stoffa_normalize(fitns, T)


		# 精英策略
		elitism_num = int(pop_num * el_radio)  # 精英策略挑选的染色体数目
		el_index = np.argsort(fitns)[:elitism_num]

		# 轮盘赌法
		roulette_num = pop_num - elitism_num  # 轮盘赌法挑选的染色体数目
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
		self.Pop.swap(index = select_index)


# def fitness(chrome, *args, **kwargs):
# 	[x, y,] = [f for f in chrome]
# 	return x** 2 + y ** 2 + 1 - 2 * x * y  + 2 *x + 3 * y
#
# chrome_kws = [(16, [-1.0, 1.0]), (16, [-1.0, 1.0]),]
#
# ga = GeneticAlgorithm(population_num=20, chrome_kws=chrome_kws, top_n=10)
# # ga.set_evolute_params( dict(cross_radio=0.4, elitism_radio=0.5, mutate_radio=0.5))
# ga.run(fitness, tol=1e-6)
# ga.run(fitness, population = ga.population)
# res = ga.result
#
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(ga.evolute_curve)
# plt.show()

