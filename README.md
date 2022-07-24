# FedPS-AggregateOrNot
code for paper: Aggregate or Not? Exploring Where to Privatize in DNN Based Federated Learning Under Different Non-IID Scenes

## Basic Introduction
For FL with a deep neural network (DNN), privatizing some layers is a simple yet effective solution for non-iid problems. However, which layers should we privatize to facilitate the learning process? Do different categories of non-iid scenes have preferred privatization ways? Can we automatically learn the most appropriate privatization way during FL? In this paper, we answer these questions via abundant experimental studies on several FL benchmarks.

## Running Tips
  * `basic_nets.py`: several network architectures (Figure.4);
  * `ps_nets.py`: several private-shared ways (Figure.1);
  * `train_ps.py`: fl performances with different private-shared models (Figure.5,6,7,8);
  * `train_ps_inter.py`: interpolation performances between shared and private models (Figure.9);
  * `train_ps_auto.py`: fl performances with automatical privatization ways (Figure.2, Figure.10).

## Citation
  * Xin-Chun Li, Le Gan, De-Chuan Zhan, Yunfeng Shao, Bingshuai Li, Shaoming Song. Aggregate or Not? Exploring Where to Privatize in DNN Based Federated Learning Under Different Non-IID Scenes. CoRR 2021.
