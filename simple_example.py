# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 19:10:15 2018

@author: 徐嘉诚
"""

from GeneticAlgorithm import Genetic_Algorithm
import matplotlib.pyplot as plt

def fitness_function(z,*args,**kwargs):
    return z[0]**2 + z[1]**2 - 2*z[0]-1 

GA_CHROME_RANGE = [[-5.0,5.0],[-5.0,5.0]]
GA_GENERATION = 300
CHROME_SIZE = 12
GA_TOP_N_MODEL = 10
GA_POPULATION = 20

ga = Genetic_Algorithm(chrome_size = CHROME_SIZE,
                       chrome_num = len(GA_CHROME_RANGE),
                       population = GA_POPULATION,
                       Range = GA_CHROME_RANGE,
                       generation = GA_GENERATION, 
                       top_n = GA_TOP_N_MODEL)

result = ga.run(fitness_function).results_
print( 'function: x**2 - 2x + y**2 - 1; best x = {0}, best y = {1}, best f(x,y) = {2}'.format(
        result['chrome'][0][0],result['chrome'][0][1],result['minimal'][0]) )