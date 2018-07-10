# genetic-algorithm-python

- python 版本的遗传算法
- 可实现fitness的并行计算，但是实际的并行效果不如不并行，其中问题尚待解决
- 本版本共实现了三种形式的遗传算法：1.精英选择的改进遗传算法那；2.自适应遗传算法；3.退火自适应遗传算法

#### Part 1 ``` Genetic_Algorithm```

- 主要对于基因选择策略中，在原始的轮盘赌法中加入了 精英策略，以保证那些优秀个体的基因总能遗传到下一代。

#### Part 2 ```Adaptive_Genetic_Algorithm```

- 自适应遗传算法中，主要在```Genetic_Algorithm```的基础上，在交叉与变异操作中进行了优化。
- ```Genetic_Algorithm```的交叉概率和变异概率随着代数的增加是保持不变的
- ```Adaptive_Genetic_Algorithm``` 的交叉和遗传概率会随着遗传代数的增加，而自适应的增加。
- 更新公式如下：(待补充)

#### Part 3 ```Anneal_Genetic_Algorithm```

- 退火遗传算法，在```Adaptive_Genetic_Algorithm```的基础上，增加了一步```accept```操作，该操作取自模拟退火算法
- ```accept```操作主要应用于交叉和变异操作，每一步的交叉和变异之后，都要判断是否满足一定条件而接受该步变异。若不满足条件，则回退到之前的情形。