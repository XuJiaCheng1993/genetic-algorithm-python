# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:51:18 2018

@author: JiachengXu
"""

import numpy as np
from progressbar import Percentage,Bar,Timer,ETA,FileTransferSpeed,ProgressBar

class _genetic_base(object):
    def __init__(self,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,Range =[None,], tol = 1e-4,verbose = 0):
        self.chrome_size = chrome_size
        self.chrome_num = chrome_num
        self.population = population    
        self.generation = generation
        self.top_n = top_n
        self.tol = tol
        self.Range = Range
        self.verbose = verbose
        self.iter_ = generation

    def initial_ga(self):
        chrome_num,population,chrome_size = self.chrome_num,self.population,self.chrome_size
        chrome = np.random.randint( 2 , size=(chrome_num,population,chrome_size),dtype = 'int8')
        return chrome

    def decode(self,chrome):

        chrome_num, population, chrome_size = self.chrome_num, self.population, self.chrome_size
        Range = self.Range
        maxim = 2 **chrome_size - 1

        wi = np.zeros([chrome_size])
        for i in range(chrome_size):
            wi[i] = pow(2, chrome_size - 1 - i)

        Out = []
        for j in range(population):
            tmp = []
            for i in range(chrome_num):
                if Range[i] is None:
                    tmp.append(chrome[i,j,:])
                else:
                    [Lw_va, Up_va] = Range[i]
                    tmp.append( (Up_va - Lw_va) * np.dot(chrome[i, j, :], wi) / maxim + Lw_va )
            Out.append(tmp)

        return Out

    def compute_fitness(self,chrome_values,function,function_params ):
        # 计算适应度值
        fitns = []
        for i,chrome_value in enumerate(chrome_values):
            fitns.append(function(chrome_value,**function_params))
        fitns = np.array(fitns).reshape([-1])
        return fitns

    def sts_minimal_chrome(self,chrome_values,fitns,n):
        # 最好的n各个体
        best_values_idx = np.argsort(fitns)[:n]
        best_values = np.sort(fitns)[:n]
        individual_best = []
        for i in best_values_idx:
            individual_best.append( chrome_values[i] )
        return best_values,individual_best

    def select(self,chrome,fitns):
        # 精英策略
        elitism_rate = self.elitism_rate # 精英策略选择的比例
        elitism_num = int(self.population*elitism_rate)
        el_index = np.argsort(fitns)[:elitism_num]
        
        # 轮盘赌法
        roulette_num = self.population - elitism_num
        rl_index = np.zeros([roulette_num])
        fitns_normalized = 1 - (fitns - np.min(fitns) )/ np.max(fitns - np.min(fitns) ) 
        cumsum_prob = np.cumsum( fitns_normalized / np.sum(fitns_normalized) )
        for i in range(roulette_num):
            for j,j_value in enumerate(cumsum_prob):
                if j_value >= np.random.rand(1):
                    rl_index[i] = j
                    break
        
        #  整合两者策略
        select_index = np.hstack((el_index,rl_index)).astype(int)
        return chrome[:,select_index,:].copy(), fitns[select_index]

    def update(self,minimal,individual,minimal_global,individual_global):
        temp_minimal = np.hstack((minimal, minimal_global))
        temp_individual = individual + individual_global
        ind = np.argsort(temp_minimal)[:self.top_n]
        best_minimal = temp_minimal[ind]
        best_individual = []
        [best_individual.append(temp_individual[i]) for i in ind]
        return best_minimal,best_individual

class Genetic_Algorithm(_genetic_base):
    def __init__(self,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,
                 elitism_rate = 0.1,cross_rate=0.8,mutation_prob=0.05,Range =[[-1,1],], tol = 1e-4,verbose=0):
        super(Genetic_Algorithm,self).__init__(chrome_num=chrome_num,
             chrome_size=chrome_size,population=population,generation=generation,top_n=top_n ,Range=Range , tol= tol,verbose = verbose)
        self.elitism_rate = elitism_rate
        self.cross_rate = cross_rate
        self.mutation_prob = mutation_prob
        self.alg_name = 'GA'
        widgets = [self.alg_name+': ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
        self.pbar = ProgressBar(widgets=widgets,maxval= 10*generation)

    def _genetic_operation(self,code):
        # 交叉与变异
        cross_rate,chrome_num,chrome_size,population = self.cross_rate,self.chrome_num,self.chrome_size,self.population
        cross_num = int( population * cross_rate / 2)
        
        code_cross = code.copy()          
        for j in range( cross_num ):  
            for i in range(chrome_num):
                cross_position = np.random.randint( 0.5*chrome_size )  # 单点交叉  
                for k in np.arange(cross_position+1,chrome_size):
                    temp = code_cross[i,2*j,k]
                    code_cross.itemset( (i,2*j,k), code_cross[i,2*j+1,k] )
                    code_cross.itemset( (i,2*j+1,k), temp)      
                    
        code_mutation = code_cross.copy()
        mutation_prob = self.mutation_prob # 变异概率
        for i in range(chrome_num):
            for j in range( population ):    
                for k in range( chrome_size ):
                    prob = np.random.rand(1)
                    if prob <= mutation_prob:
                        temp = code_mutation[i,j,k]
                        code_mutation.itemset( (i,j,k) , int(1-temp) ) 
        return code_mutation.copy()
    
    def run(self,fitness_function , function_extra_params = dict(default = 'None')):
         # 初始化
        genetic_curve = []
        chrome = self.initial_ga()
        chrome_values = self.decode(chrome)
        fitns = self.compute_fitness(chrome_values,fitness_function,function_extra_params)
        minimal_global,individual_best_global = self.sts_minimal_chrome(chrome_values,fitns,self.top_n)
        self.pbar.start()
        for i in range(self.generation):
            self.pbar.update(10 * i + 1)
            # step1 选择
            chrome_selected, _ = self.select(chrome,fitns)
            # step2 染色体遗传
            code_cromut = self._genetic_operation(chrome_selected)
            # step3 染色体解码
            chrome_values = self.decode(code_cromut)
            # step4 计算适应度函数
            fitns = self.compute_fitness(chrome_values,fitness_function,function_extra_params)
            # 统计top_n的个体的适应度
            minimal,individual = self.sts_minimal_chrome(chrome_values,fitns,self.top_n)
            genetic_curve.append(np.min(fitns))
            # 满足条件则停机
            if np.sum(np.abs(minimal_global-minimal)) < self.tol:
                print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
                break
            # 更新
            minimal_global, individual_best_global = self.update(minimal,individual,minimal_global,individual_best_global)
        else:
            print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
        self.iter_ = i+1
        self.pbar.finish()
        self.results_ = {'minimal':minimal_global,'chrome':individual_best_global,'curve':np.array(genetic_curve)}
        return self
                
class Adaptive_Genetic_Algorithm(_genetic_base):
    def __init__(self,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,cross_rate = 0.8,
                 elitism_rate = 0.1,Range =[None,], tol = 1e-4,verbose=0):
        super(Adaptive_Genetic_Algorithm,self).__init__(chrome_num=chrome_num,
             chrome_size=chrome_size,population=population,generation=generation,top_n=top_n ,Range=Range, tol= tol,verbose = verbose)
        self.elitism_rate = elitism_rate
        self.cross_rate = cross_rate
        self.alg_name = 'Adaptive_GA'
        widgets = [self.alg_name+': ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
        self.pbar = ProgressBar(widgets=widgets,maxval= 10*generation)
        
    def run(self,fitness_function , function_extra_params = dict(default = 'None')):
         # 初始化
        genetic_curve = []
        chrome = self.initial_ga()
        chrome_values = self.decode(chrome)
        fitns = self.compute_fitness(chrome_values,fitness_function,function_extra_params)
        minimal_global,individual_best_global = self.sts_minimal_chrome(chrome_values,fitns,self.top_n)
        self.pbar.start()
        for i in range(self.generation):
            self.pbar.update(10 * i + 1)
            # step1 选择
            chrome_selected, fitns_selected = self.select(chrome, fitns)
            # step3 染色体遗传
            code_cromut = self._genetic_operation(chrome_selected,fitns_selected)
            # step4 染色体解码
            chrome_values = self.decode(code_cromut)
            # step5 计算适应度函数
            fitns = self.compute_fitness(chrome_values, fitness_function, function_extra_params)
            # 统计top_5的个体的适应度
            minimal,individual = self.sts_minimal_chrome(chrome_values,fitns,self.top_n)
            genetic_curve.append(np.min(fitns))
            # 满足条件则停机
            if np.sum(np.abs(minimal_global-minimal)) < self.tol:
                print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
                break
            # 更新
            minimal_global, individual_best_global = self.update(minimal, individual, minimal_global,
                                                                 individual_best_global)
        else:
            print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
        self.iter_ = i+1
        self.pbar.finish()
        self.results_ = {'minimal':minimal_global,'chrome':individual_best_global,'curve':np.array(genetic_curve)}
        return self

    def _genetic_operation(self,code,fitns):
        fmin = np.min(fitns)
        fave = np.mean(fitns)
                        
        # 交叉
        cross_rate,chrome_num,chrome_size,population = self.cross_rate,self.chrome_num,self.chrome_size,self.population
        cross_num = int( population * cross_rate / 2)
        
        code_cross = code.copy()   
        for i in range(chrome_num):
            for j in range( cross_num ):  
                if np.min( fitns[2*j:2*j+2] ) <= fave:
                    cross_prob = 0.9 - 0.6*(fave - np.min( fitns[2*j:2*j+2] ))/(fmin * fave)
                else:
                    cross_prob = 0.9
                prob = np.random.rand(1)
                if prob >= cross_prob:
                    cross_position = np.random.randint( 0.5*chrome_size )  # 单点交叉  
                    for k in np.arange(cross_position+1,chrome_size):
                        temp = code_cross[i,2*j,k]
                        code_cross.itemset( (i,2*j,k), code_cross[i,2*j+1,k] )
                        code_cross.itemset( (i,2*j+1,k), temp)      
        
        # 变异            
        code_mutation = code_cross.copy()
        for j in range( population ): 
            if fitns[j] > fave:
                mutation_prob = 0.1
            else:
                mutation_prob = 0.1 - 0.009 * fmin * fitns[j] / (fmin * fave)
            for i in range(chrome_num):    
                for k in range( chrome_size ):
                    prob = np.random.rand(1)
                    if prob <= mutation_prob:
                        temp = code_mutation[i,j,k]
                        code_mutation.itemset( (i,j,k) , int(1-temp) ) 
        return code_mutation.copy()
    
class Anneal_Genetic_Algorithm(_genetic_base):
    def __init__(self,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,cross_rate = 0.8,T0 = 90,speed = 0.99,
                 elitism_rate = 0.1,Range =[None,], tol = 1e-4,verbose=0):
        super(Anneal_Genetic_Algorithm,self).__init__(chrome_num=chrome_num,
             chrome_size=chrome_size,population=population,generation=generation,top_n=top_n ,Range=Range , tol= tol,verbose = verbose)
        self.elitism_rate = elitism_rate
        self.cross_rate = cross_rate    
        self.T0 = T0
        self.speed = speed
        self.alg_name = 'Anneal_GA'
        widgets = [self.alg_name+': ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
        self.pbar = ProgressBar(widgets=widgets,maxval= 10*generation)

    def _stoffa_method(self,fitns,T):
        fitns = np.power(np.e,fitns/T)
        fsum = np.sum(fitns)
        fitns = fitns / fsum
        return fitns.copy()
    
    def _accept(self,chrome1,chrome2,fitn1,fitn2,T):
        chrome_choose = np.zeros_like(chrome1,dtype='int8')
        fitns_choose = np.zeros_like(fitn1,dtype='float64')
        for i in range(len(fitn1)):
            if fitn2[i] < fitn1[1]:
                chrome_choose[:,i,:] = chrome2[:,i,:]
                fitns_choose[i] = fitn2[i]
            else:
                accept_prob = np.e ** ( (fitn1[1] - fitn2[1])/T )
                prob = np.random.rand(1)
                if prob < accept_prob:
                    chrome_choose[:,i] = chrome2[:,i,:]
                    fitns_choose[i] = fitn2[i]
                else:
                    chrome_choose[:,i] = chrome1[:,i,:]
                    fitns_choose[i] = fitn1[i]
        return chrome_choose, fitns_choose
           
    def run(self,fitness_function , function_extra_params = dict(default = 'None')):
         # 初始化
        genetic_curve = []
        chrome = self.initial_ga()
        chrome_values = self.decode(chrome)
        fitns = self.compute_fitness(chrome_values,fitness_function,function_extra_params)
        minimal_global,individual_best_global = self.sts_minimal_chrome(chrome_values,fitns,self.top_n)
        self.pbar.start()
        for i in range(self.generation):
            self.pbar.update(10*i+1)
            temperature = self.T0 * self.speed ** i
            stoffa_fitns = self._stoffa_method(fitns,temperature)
            # step1 选择
            chrome_selected, fitns_selected = self.select(chrome, stoffa_fitns)
            # step2 染色体编码
            code_cromut = self._genetic_operation(chrome_selected,fitns_selected)
            # step4 染色体解码
            chrome_values = self.decode(code_cromut)
            # step5 计算适应度函数
            fitns = self.compute_fitness(chrome_values, fitness_function, function_extra_params)
            # step6 模拟退火的accept
            chrome,fitns = self._accept(chrome_selected,chrome,fitns_selected,fitns,temperature)
            chrome_values = self.decode(chrome)
            # 统计top_5的个体的适应度
            minimal,individual = self.sts_minimal_chrome(chrome_values,fitns,self.top_n)
            genetic_curve.append(np.min(fitns))
            # 满足条件则停机
            if np.sum(np.abs(minimal_global-minimal)) < self.tol:
                print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
                break
            # 更新
            minimal_global, individual_best_global = self.update(minimal, individual, minimal_global,
                                                                 individual_best_global)
        else:
            print('iter {0} times, best value {1}'.format(i + 1, minimal_global[0]))

        self.pbar.finish()
        self.iter_ = i+1

        self.results_ = {'minimal':minimal_global,'chrome':individual_best_global,'curve':np.array(genetic_curve)}
        return self

    def _genetic_operation(self,code,fitns):
        fmin = np.min(fitns)
        fave = np.mean(fitns)                        
        # 交叉
        cross_rate,chrome_num,chrome_size,population = self.cross_rate,self.chrome_num,self.chrome_size,self.population
        cross_num = int( population * cross_rate / 2)
        
        code_cross = code.copy()   
        for i in range(chrome_num):
            for j in range( cross_num ):  
                if np.min( fitns[2*j:2*j+2] ) <= fave:
                    cross_prob = 0.9 - 0.6*(fave - np.min( fitns[2*j:2*j+2] ))/(fmin * fave)
                else:
                    cross_prob = 0.9
                prob = np.random.rand(1)
                if prob >= cross_prob:
                    cross_position = np.random.randint( 0.5*chrome_size )  # 单点交叉  
                    for k in np.arange(cross_position+1,chrome_size):
                        temp = code_cross[i,2*j,k]
                        code_cross.itemset( (i,2*j,k), code_cross[i,2*j+1,k] )
                        code_cross.itemset( (i,2*j+1,k), temp)      
        
        # 变异            
        code_mutation = code_cross.copy()
        for j in range( population ): 
            if fitns[j] > fave:
                mutation_prob = 0.1
            else:
                mutation_prob = 0.1 - 0.009 * fmin * fitns[j] / (fmin * fave)
            for i in range(chrome_num):    
                for k in range( chrome_size ):
                    prob = np.random.rand(1)
                    if prob <= mutation_prob:
                        temp = code_mutation[i,j,k]
                        code_mutation.itemset( (i,j,k) , int(1-temp) ) 
        return code_mutation.copy()

if __name__ == '__main__':
    print('this is ga for python')
    