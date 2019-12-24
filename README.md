# OTCoarsening
The codes are originally for our paper "[Unsupervised Learning of Graph Hierarchical Abstractions with Differentiable Coarsening and Optimal Transport](https://openreview.net/forum?id=Bkf4XgrKvS)" (we will put it on arxiv soon).


The codes are built on Pytorch Geometric library. Please configure your environment as PG requires. 

The main file is "main_opt2.py". The following is an example script for running:
```python -u ./src/main_opt2.py --eps=1.0 --lr=0.001 --opt_iters=10 --no_extra_mlp --train --epochs=100```