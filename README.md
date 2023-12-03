
# CUDA Implementation of Generalized Advantage Estimation (GAE)

## Introduction

Generalized Advantage Estimation (GAE) is widely used in RL, especially PPO.
The computation of GAE involves a for-loop to iterate over the entire trajectory, which is expensive in Python and may become the training bottleneck.

This repository provides a simple implementation of GAE in CUDA, which can achive at most **2000x higher throughput** than Python implementation.

## Usage

Installation requires a CUDA-enabled GPU with `nvcc` and `torch` installed.

```shell
git clone https://github.com/garrett4wade/cugae
cd cugae && pip3 install -e .
```

After installation, run
```pytest -q -s test_cugae.py```
to run tests and validate your installation.

See [`cugae.py`](https://github.com/garrett4wade/cugae/blob/main/cugae.py) for detailed documentation of each implemented function.

## Benchmark Results

This benchmark is performed using Python 3.10.12, CUDA 12.2 in WSL2 Unbuntu 22.04 on a laptop with Intel i7 i7-12700H CPU and Nvidia 3070 GPU.

![Benchmark Results](https://github.com/garrett4wade/cugae/blob/main/benchmark.png)
