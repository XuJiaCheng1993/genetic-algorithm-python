# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:51:18 2018

@author: 徐嘉诚
"""

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
import matplotlib.pyplot as plt
    
class _genetic_base(object):
    def __init__(self,fitness_function,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,Range =None,n_jobs = 1, tol = 1e-4,verbose = 0):
        self.fitness_function = fitness_function
        self.chrome_size = chrome_size
        self.chrome_num = chrome_num
        self.population = population    
        self.generation = generation
        self.top_n = top_n
        self.tol = tol
        self.Range = Range
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.iter_ = generation
        
    def trans(self,chrome):
        Range = self.Range
        if Range is None:
            dataout = chrome
        else:        
            dataout = np.zeros([chrome.shape[0],chrome.shape[1]])
            maxim = pow(2,self.chrome_size)-1
            for i in range(self.chrome_num):
                [Lw_va,Up_va] = Range[i]          
                dataout[i,:] = (Up_va - Lw_va) * chrome[i,:] / maxim + Lw_va
                
        return dataout

    def fitness(self,x):
        return self.fitness_function(x)

    def coding(self,orin):
        chrome_num,population,chrome_size = self.chrome_num,self.population,self.chrome_size
        # 染色体编码
        code = np.zeros([chrome_num,population,chrome_size],dtype = 'int8')
        for i in range(chrome_num):
            for j in range(population):
                temp = orin[i,j]
                for k in range(chrome_size):
                    if temp >= pow(2,chrome_size-1-k):
                        code[i,j,k] = 1
                        temp -= pow(2, chrome_size-1-k )
                    else:
                        code[i,j,k] = 0
        return code

    def decoding(self,code):
        chrome_num,population,chrome_size = self.chrome_num,self.population,self.chrome_size
        wi = np.zeros([chrome_size])
        S_de = np.zeros([chrome_num,population])
        for i in range(chrome_size):
            wi[i] = pow(2 , chrome_size-1-i )    
        for i in range(chrome_num ):
            for j in range(population):
                S_de[i,j] = np.dot( code[i,j,:] ,wi )   
        return S_de

    def compute_fitness(self,chrome,function):
        chrome = self.trans(chrome)
        # 并行计算计算每个染色体的 fitnesss
        fitns = Parallel(n_jobs = self.n_jobs,verbose = self.verbose)( delayed(function) ( x = chrome[:,i] ) for i in range(self.population) )  
        fitns = np.array(fitns).reshape([-1])
        return fitns

    def initial_ga(self):
        chrome_num,population,chrome_size = self.chrome_num,self.population,self.chrome_size
        # 随机生成 一批染色体
        chrome = np.random.randint( pow(2,chrome_size) , size=(chrome_num ,population))   
        # 计算适应度值
        fitns = self.compute_fitness(chrome,self.fitness)
        return chrome,fitns
    
    def sts_minimal_chrome(self,chrome,fitns,n):
        # 最好的n各个体
        individual_best = chrome[:,np.argsort(fitns)[:n]]
        return np.sort(fitns)[:n],individual_best

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
        return select_index
                  
    def fit(self):
        return self._fit()

class Genetic_Algorithm(_genetic_base):
    def __init__(self,fitness_function,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,
                 elitism_rate = 0.1,cross_rate=0.8,mutation_prob=0.05,Range =[[-1,1],],n_jobs = 1, tol = 1e-4,verbose=0):
        super(Genetic_Algorithm,self).__init__(fitness_function=fitness_function,chrome_num=chrome_num,
             chrome_size=chrome_size,population=population,generation=generation,top_n=top_n ,Range=Range ,n_jobs=n_jobs, tol= tol,verbose = verbose)
        self.elitism_rate = elitism_rate
        self.cross_rate = cross_rate
        self.mutation_prob = mutation_prob

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
    
    def _fit(self):
         # 初始化
        best_fitness = []
        chrome,fitns = self.initial_ga()  
        minimal_global,individual_best_global = self.sts_minimal_chrome(chrome,fitns,self.top_n)
        for i in range(self.generation):        
            # step1 选择
            index_selected = self.select(chrome,fitns)
            chrome_selected = chrome[:,index_selected].copy()
            # step2 染色体编码
            code = self.coding(chrome_selected)
            # step3 染色体遗传
            code_cromut = self._genetic_operation(code) 
            # step4 染色体解码
            chrome = self.decoding(code_cromut)
            # step5 计算适应度函数
            fitns = self.compute_fitness(chrome,self.fitness)
            # 统计top_5的个体的适应度
            minimal,individual = self.sts_minimal_chrome(chrome,fitns,self.top_n)
            best_fitness.append(np.min(fitns))
            # 满足条件则停机
            if np.sum(np.abs(minimal_global-minimal)) < self.tol:
                print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
                break
            # 更新
            temp_minimal = np.hstack((minimal,minimal_global))
            temp_individual = np.hstack((individual,individual_best_global))
            ind = np.argsort(temp_minimal)[:self.top_n]
            minimal_global = temp_minimal[ind]
            individual_best_global = temp_individual[:,ind]
        else:
            print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
        self.iter_ = i+1 
        return minimal_global,self.trans(individual_best_global),np.array(best_fitness) 
                
class Adaptive_Genetic_Algorithm(_genetic_base):
    def __init__(self,fitness_function,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,cross_rate = 0.8,
                 elitism_rate = 0.1,Range =[[-1,1],],n_jobs = 1, tol = 1e-4,verbose=0):
        super(Adaptive_Genetic_Algorithm,self).__init__(fitness_function=fitness_function,chrome_num=chrome_num,
             chrome_size=chrome_size,population=population,generation=generation,top_n=top_n ,Range=Range ,n_jobs=n_jobs, tol= tol,verbose = verbose)
        self.elitism_rate = elitism_rate
        self.cross_rate = cross_rate
        
    def _fit(self):
         # 初始化
        best_fitness = []
        chrome,fitns = self.initial_ga()  
        minimal_global,individual_best_global = self.sts_minimal_chrome(chrome,fitns,self.top_n)
        for i in range(self.generation):        
            # step1 选择
            index_selected = self.select(chrome,fitns)
            chrome_selected = chrome[:,index_selected].copy()
            fitns_selected = fitns[index_selected].copy()
            # step2 染色体编码
            code = self.coding(chrome_selected)
            # step3 染色体遗传
            code_cromut = self._genetic_operation(code,fitns_selected) 
            # step4 染色体解码
            chrome = self.decoding(code_cromut)
            # step5 计算适应度函数
            fitns = self.compute_fitness(chrome,self.fitness)
            # 统计top_5的个体的适应度
            minimal,individual = self.sts_minimal_chrome(chrome,fitns,self.top_n)
            best_fitness.append(np.min(fitns))
            # 满足条件则停机
            if np.sum(np.abs(minimal_global-minimal)) < self.tol:
                print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
                break
            # 更新
            temp_minimal = np.hstack((minimal,minimal_global))
            temp_individual = np.hstack((individual,individual_best_global))
            ind = np.argsort(temp_minimal)[:self.top_n]
            minimal_global = temp_minimal[ind]
            individual_best_global = temp_individual[:,ind]
        else:
            print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
        self.iter_ = i+1 
        return minimal_global,self.trans(individual_best_global),np.array(best_fitness)    

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
    def __init__(self,fitness_function,chrome_num = 1,chrome_size = 4,population = 20,generation = 10,top_n = 5,cross_rate = 0.8,T0 = 90,speed = 0.99,
                 elitism_rate = 0.1,Range =[[-1,1],],n_jobs = 1, tol = 1e-4,verbose=0):
        super(Anneal_Genetic_Algorithm,self).__init__(fitness_function=fitness_function,chrome_num=chrome_num,
             chrome_size=chrome_size,population=population,generation=generation,top_n=top_n ,Range=Range ,n_jobs=n_jobs, tol= tol,verbose = verbose)
        self.elitism_rate = elitism_rate
        self.cross_rate = cross_rate    
        self.T0 = T0
        self.speed = speed

    def _stoffa_method(self,fitns,T):
        fitns = np.power(np.e,fitns/T)
        fsum = np.sum(fitns)
        fitns = fitns / fsum
        return fitns.copy()
    
    def _accept(self,chrome1,chrome2,fitn1,fitn2,T):
        chrome = np.zeros([chrome1.shape[0],chrome1.shape[1]])
        fitns = np.zeros([len(fitn1)])
        for i in range(len(fitn1)):
            if fitn2[i] < fitn1[1]:
                chrome[:,i] = chrome2[:,i]
                fitns[i] = fitn2[i]
            else:
                accept_prob = np.e ** ( (fitn1[1] - fitn2[1])/T )
                prob = np.random.rand(1)
                if prob < accept_prob:
                    chrome[:,i] = chrome2[:,i]
                    fitns[i] = fitn2[i]
                else:
                    chrome[:,i] = chrome1[:,i]
                    fitns[i] = fitn1[i]
        return chrome.copy(),fitns.copy()
           
    def _fit(self):
         # 初始化
        best_fitness = []
        chrome,fitns = self.initial_ga()  
        minimal_global,individual_best_global = self.sts_minimal_chrome(chrome,fitns,self.top_n)
        for i in range(self.generation): 
            temperature = self.T0 * self.speed ** i
            stoffa_fitns = self._stoffa_method(fitns,temperature)
            # step1 选择
            index_selected = self.select(chrome,stoffa_fitns)
            chrome_selected = chrome[:,index_selected].copy()
            fitns_selected = fitns[index_selected].copy()
            # step2 染色体编码
            code = self.coding(chrome_selected)
            # step3 染色体遗传
            code_cromut = self._genetic_operation(code,fitns_selected) 
            # step4 染色体解码
            chrome = self.decoding(code_cromut)
            # step5 计算适应度函数
            fitns = self.compute_fitness(chrome,self.fitness)
            # step6 模拟退火的accept
            chrome,fitns = self._accept(chrome_selected,chrome,fitns_selected,fitns,temperature)
            # 统计top_5的个体的适应度
            minimal,individual = self.sts_minimal_chrome(chrome,fitns,self.top_n)
            best_fitness.append(np.min(fitns))
            # 满足条件则停机
            if np.sum(np.abs(minimal_global-minimal)) < self.tol:
                print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
                break
            # 更新
            temp_minimal = np.hstack((minimal,minimal_global))
            temp_individual = np.hstack((individual,individual_best_global))
            ind = np.argsort(temp_minimal)[:self.top_n]
            minimal_global = temp_minimal[ind]
            individual_best_global = temp_individual[:,ind]
        else:
            print('iter {0} times, best value {1}'.format(i+1,minimal_global[0]))
        self.iter_ = i+1 
        return minimal_global,self.trans(individual_best_global),np.array(best_fitness)    

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


#def fitn(x):
#    return x*np.sin(4*x**3)+2
#    
#if __name__ == '__main__':
#    __spec__ = None
#    Rane = [[-1,2],]
#    ga = Adaptive_Genetic_Algorithm(fitn,chrome_size=10,chrome_num=1,Range=Rane,population=20,n_jobs=1,tol= 0.01,generation=1000)   
#    minimal_global,individual_best_global,curve = ga.fit()
#    minimal_global = np.round(minimal_global,3)
#    individual_best_global = np.round(individual_best_global,3)
#    print('function: x*np.sin(4*np.pi*x)+2 x∈[-1,2], min={0},x={1}'.format(minimal_global[0],individual_best_global[0,0])  )
#    plt.figure()
#    plt.subplot(312)
#    x = np.linspace(-1,2,1000)
#    plt.plot(x,fitn(x))
#    plt.scatter(individual_best_global[0,0],minimal_global[0],s = 200,marker='*',c = 'r'  )
#    plt.text(-1,3.5,'function: x*np.sin(4*np.pi*x)+2 x∈[-1,2], min={0},x={1}'.format(minimal_global[0],individual_best_global[0,0]) )
#    plt.title('Adaptive_Genetic_Algorithm iter {0}'.format(ga.iter_) )
#    
#    ga = Genetic_Algorithm(fitn,chrome_size=10,chrome_num=1,Range=Rane,population=20,n_jobs=1,tol= 0.01,generation=1000)   
#    minimal_global,individual_best_global,curve = ga.fit()
#    minimal_global = np.round(minimal_global,3)
#    individual_best_global = np.round(individual_best_global,3)
#    print('function: x*np.sin(4*np.pi*x)+2 x∈[-1,2], min={0},x={1}'.format(minimal_global[0],individual_best_global[0,0])  )
#    plt.subplot(311)
#    x = np.linspace(-1,2,1000)
#    plt.plot(x,fitn(x))
#    plt.scatter(individual_best_global[0,0],minimal_global[0],s = 200,marker='*',c = 'r'  )
#    plt.text(-1,3.5,'function: x*np.sin(4*np.pi*x)+2 x∈[-1,2], min={0},x={1}'.format(minimal_global[0],individual_best_global[0,0]) )
#    plt.title('Genetic_Algorithm iter {0}'.format(ga.iter_) )
#    
#    ga = Anneal_Genetic_Algorithm(fitn,chrome_size=10,chrome_num=1,Range=Rane,population=20,n_jobs=1,tol= 0.01,generation=1000)   
#    minimal_global,individual_best_global,curve = ga.fit()
#    minimal_global = np.round(minimal_global,3)
#    individual_best_global = np.round(individual_best_global,3)
#    print('function: x*np.sin(4*np.pi*x)+2 x∈[-1,2], min={0},x={1}'.format(minimal_global[0],individual_best_global[0,0])  )
#    plt.subplot(313)
#    x = np.linspace(-1,2,1000)
#    plt.plot(x,fitn(x))
#    plt.scatter(individual_best_global[0,0],minimal_global[0],s = 200,marker='*',c = 'r'  )
#    plt.text(-1,3.5,'function: x*np.sin(4*np.pi*x)+2 x∈[-1,2], min={0},x={1}'.format(minimal_global[0],individual_best_global[0,0]) )
#    plt.title('Anneal_Algorithm iter {0}'.format(ga.iter_) )
    


        

        






















