# DGAT-LPS:  A new semi-supervised fault diagnosis method called dynamic graph attention network witg label propagation strategy
* Core codes for the paper ["Semi-supervised fault diagnosis of machinery using LPS-DGAT under speed  fluctuation and extremely low labeled rates"](https://www.sciencedirect.com/science/article/abs/pii/S1474034622001124)
* Created by Shen Yan, Haidong Shao, Yiming Xiao, Jian Zhou, Yuandong Xu and Jiafu Wan.
* Journal: Advanced Engineering Informatics

## Our operating environment
* Python 3.8
* torch-geometric 2.2.0
* pytorch  1.10.1
* pandas  1.5.3
* numpy  1.23.5
* and other necessary libs

## Guide 
* This repository provides a concise framework for semi-supervised fault diagnosis. It includes a demo dataset; the pre-processing and graph composition process for the data and the model proposed in the paper. 
* You just need to run `train_test_graph.py`. You can also adjust the structure and parameters of the model to suit your needs.

## Pakages
* `data` contians a demo dataset
* `datasets` contians the pre-processing and graph composition process for the data
* `models` contians the model proposed in the paper

## Acknowledgement
* The DGAT is derived from the paper: [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)
* Special thanks to Li et al. for the GNN base framework provided by [PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark)

## Citation
If you use our work as a comparison model, please cite:
```
@paper{DGAT-LPS,
  title = {Semi-supervised fault diagnosis of machinery using LPS-DGAT under speed  fluctuation and extremely low labeled rates},
  author = {Shen Yan, Haidong Shao, Yiming Xiao, Jian Zhou, Yuandong Xu and Jiafu Wan},
  journal = {Advanced Engineering Informatics},
  volume = {53},
  pages = {101648},
  year = {2022},
  doi = {https://doi.org/10.1016/j.aei.2022.101648},
  url = {https://www.sciencedirect.com/science/article/abs/pii/S1474034622001124},
}
```
If our work is useful to you, please star it, it is the greatest encouragement to our open source work, thank you very much!

## Contact
- yanshen0210@gmail.com
