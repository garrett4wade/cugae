
# CUDA Implementation of Generalized Advantage Estimation (GAE)

## Introduction

Generalized Advantage Estimation (GAE) is widely used in RL, especially PPO.
The computation of GAE involves a for-loop to iterate over the entire trajectory, which is expensive in Python and may become the bottleneck of PPO training.
This repository provides a simple implementation of GAE in CUDA, which can achive 1000x higher throughput than Python implementation.

## Usage

Installation requires a CUDA-enabled GPU and `torch` installed.

```shell
git clone https://github.com/garrett4wade/cugae
cd cugae && pip3 install -e .
```

After installation via any of the following approaches, run
```pytest -q -s test_cugae.py```
to run tests and validate your installation.
See `cugae.py` for detailed documentation of each implemented function.

## Benchmark Results

TODO
