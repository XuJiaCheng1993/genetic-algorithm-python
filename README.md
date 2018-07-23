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

- 二元函数: $\ f(x,y) = x^{2}-2x+y^{2}-1$  求最小值。[点击样例](https://github.com/XuJiaCheng1993/genetic-algorithm-python/blob/master/simple_example.py)
- 用于SVM的特征选择。[点击样例](https://github.com/XuJiaCheng1993/genetic-algorithm-python/blob/master/example_for_feature_selection.py)

## Installation

- 下载
- 将```GeneticAlgorithm.py``` 放到工程目录下即可

