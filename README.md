# OTCoarsening
The codes are for our paper "[Unsupervised Learning of Graph Hierarchical Abstractions with Differentiable Coarsening and Optimal Transport](https://arxiv.org/abs/1912.11176)" in AAAI 2021.


The codes are built on [Pytorch Geometric library](https://github.com/rusty1s/pytorch_geometric). Please configure your environment as PyG requires (e.g. Pytorch version>=1.2.0). 

The main file is "main_opt2.py". The following is an example script for running:
```python -u ./src/main_opt2.py --eps=1.0 --lr=0.001 --opt_iters=10 --no_extra_mlp --train --epochs=100```

