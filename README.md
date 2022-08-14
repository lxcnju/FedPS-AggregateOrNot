# FedPS-AggregateOrNot
code for paper: Aggregate or Not? Exploring Where to Privatize in DNN Based Federated Learning Under Different Non-IID Scenes

## Basic Introduction
For FL with a deep neural network (DNN), privatizing some layers is a simple yet effective solution for non-iid problems. However, which layers should we privatize to facilitate the learning process? Do different categories of non-iid scenes have preferred privatization ways? Can we automatically learn the most appropriate privatization way during FL? In this paper, we answer these questions via abundant experimental studies on several FL benchmarks.

## Environment Dependencies
The code files are written in Python, and the utilized deep learning tool is PyTorch.
  * `python`: 3.7.3
  * `numpy`: 1.21.5
  * `torch`: 1.9.0
  * `torchvision`: 0.10.0
  * `pillow`: 8.3.1

## Datasets
We provide several datasets including CIFAR-10, CIFAR-100, Shakespear, and FeMnist. The file names could be found in `paths.py`.
  * CIFAR-10: \[[cifar10-train-part1.pkl, cifar10-train-part2.pkl, cifar10-test.pkl](http://www.lamda.nju.edu.cn/lixc/data/CIFAR10.zip)\]
  * CIFAR-100: \[[cifar100-train-part1.pkl, cifar100-train-part2.pkl, cifar100-test.pkl](http://www.lamda.nju.edu.cn/lixc/data/CIFAR100.zip)\]
  * Shakespeare: \[[Shakespeare.json](http://www.lamda.nju.edu.cn/lixc/data/Shakespeare.zip)\]
  * FeMnist: \[[femnist_pkls/part1](http://www.lamda.nju.edu.cn/lixc/data/femnist_pkls/part1.zip)\] \[[femnist_pkls/part2](http://www.lamda.nju.edu.cn/lixc/data/femnist_pkls/part2.zip)\] \[[femnist_pkls/part3](http://www.lamda.nju.edu.cn/lixc/data/femnist_pkls/part3.zip)\] \[[femnist_pkls/part4](http://www.lamda.nju.edu.cn/lixc/data/femnist_pkls/part4.zip)\] \[[femnist_pkls/part5](http://www.lamda.nju.edu.cn/lixc/data/femnist_pkls/part5.zip)\] 

## Running Tips
  * `basic_nets.py`: several network architectures (Figure.4);
  * `ps_nets.py`: several private-shared ways (Figure.1);
  * `train_fedps.py`: fl performances with different private-shared models (Figure.5,6,7,8);
  * `train_fedps_inter.py`: interpolation performances between shared and private models (Figure.9);
  * `train_fedps_auto.py`: fl performances with automatical privatization ways (Figure.2, Figure.10).

## Citation
  * Xin-Chun Li, Le Gan, De-Chuan Zhan, Yunfeng Shao, Bingshuai Li, Shaoming Song. Aggregate or Not? Exploring Where to Privatize in DNN Based Federated Learning Under Different Non-IID Scenes. CoRR 2021.
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]
