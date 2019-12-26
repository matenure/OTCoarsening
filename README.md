# OTCoarsening
The codes are originally for our paper "[Unsupervised Learning of Graph Hierarchical Abstractions with Differentiable Coarsening and Optimal Transport](https://openreview.net/forum?id=Bkf4XgrKvS)" (ArXiv version: https://arxiv.org/abs/1912.11176).


The codes are built on [Pytorch Geometric library](https://github.com/rusty1s/pytorch_geometric). Please configure your environment as PyG requires. 

The main file is "main_opt2.py". The following is an example script for running:
```python -u ./src/main_opt2.py --eps=1.0 --lr=0.001 --opt_iters=10 --no_extra_mlp --train --epochs=100```