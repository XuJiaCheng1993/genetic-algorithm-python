## 最新更新
2019-03-23 万物皆对象的思路重写了遗传算法，新的遗传算法为PyGene.py, 基本的接口与之前一致, 以及更新了一个GA-SVM的demo函数[demo_PyGene.py](https://github.com/XuJiaCheng1993/genetic-algorithm-python/blob/master/demo_PyGene.py)。


# genetic-algorithm-python

- python 版本的遗传算法
- 本代码实现了三种形式的遗传算法：1.精英选择的改进遗传算法；2.基于1改进的自适应遗传算法；3.基于2改进的退火自适应遗传算法

## Usage

```python
from GeneticAlgorithm import Adaptive_Genetic_Algorithm
ga = Adaptive_Genetic_Algorithm()
ga.run()
res = ga.results_
```

## Example

- 二元函数: $\ f(x,y) = x^{2}-2x+y^{2}-1$  求最小值。[点击样例](https://github.com/XuJiaCheng1993/genetic-algorithm-python/blob/master/demo_simple_ga.py)
- 用于SVM的特征选择。[点击样例](https://github.com/XuJiaCheng1993/genetic-algorithm-python/blob/master/demo_modelparam_ga.py)

## Installation

- 下载
- 将```GeneticAlgorithm.py``` 放到工程目录下即可

