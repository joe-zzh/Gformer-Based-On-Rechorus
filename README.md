# GFormer-masteer model Reproduction Based on ReChorus Framework

本项目旨在复现基于ReChorus框架的Gformer-master模型.
ReChorus是一个开源的推荐系统框架，用于处理多种推荐算法的研究和复现工作
ReChorus框架的github地址为：https://github.com/THUwangcy/ReChorus
GFormer-master模型是ReChorus框架中用于处理文本序列的模型，其结构与BERT模型类似，但采用了更复杂的注意力机制。
GFormer-master模型的github地址为：https://github.com/HKUDS/GFormer
本项目将基于ReChorus框架复现Gformer-master模型，并在多个推荐任务上进行验证。


## Requirements

本人在windows11系统上进行复现，依赖的库与ReChorus框架有些许不同:
- numpy==1.23.5
- scipy==1.14.1
- yaml==0.2.5

## Project Structure
- scr/: 包含模型实现代码
    - main.py: 复现任务的主程序
    - models/: 各个模型的定义（Gformer-master,以及作为对比的各个模型）
    - BaseReader/：数据读取和预处理，使用Rechorus提供的基础框架
    - BaseReader/： 模型训练与评估，使用Rechorus提供的基础框架
    - src\models\general\GFormer.py:复现的GFormer模型
- data/: 包含数据集
    - Grocery_and_Gourmet_Food/:amazon数据集
    - MIND_Large/：MIND_small数据集(受限于设备故未选择Large数据集)
    - MoviesLens/: MoviesLens-1M数据集
- results/: 保存模型训练结果
- readme.md: 项目说明文档

其中，由于部分模型的不兼容，故手动保存了各个模型的输出结果到Result中。

## Data Preparation
1.下载并且解压数据集
2.将数据放入对应的‘data/'文件下
3.运行相应的xxxx.ipynb文件，进行数据预处理
4.处理完成数据

## Usage
运行下面的命令：
```
python src/main.py --model_name BPR --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset dataset_name
```

- `--emb_size`： 批处理大小
- `--dataset`： 数据集名称
- `--lr`： 学习率
- `--l2`： 优化器的权重衰减
- `--seed`： 随机种子
- 可以根据具体需求添加其他参数

## Result
| Data                     | Metric               | GFormer_master         | BPRMF                  | BUIR                   |CFKG                    |LightGCN                |POP                     |
|:-------------------------|:---------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Grocery_and_Gourmet_Food | HR@5</br><br/>NDCG@5 | 0.4109</br><br/>0.3117 | 0.3690</br><br/>0.2722 | 0.4109</br><br/>0.3117 | 0.3409</br><br/>0.2606 | 0.4109</br><br/>0.3117 | 0.4109</br><br/>0.3117 |
| MIND_Large               | HR@5</br><br/>NDCG@5 | 0.3690</br><br/>0.2722 | 0.2382</br><br/>0.1915 | 0.2948</br><br/>0.1915 | 0.1804</br><br/>0.1207 | 0.2948</br><br/>0.1915 | 0.2948</br><br/>0.1915 |
| MovieLens-1M             | HR@5</br><br/>NDCG@5 | 0.5285</br><br/>0.3907 | 0.5285</br><br/>0.3907 | 0.5285</br><br/>0.3907 | 0.5285</br><br/>0.3907 | 0.5285</br><br/>0.3692 | 0.5109</br><br/>0.3692 |

