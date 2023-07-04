<div align="center">

# DIFFUSION MODEL

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

My implementation of Diffusion Model.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/phamgialinhlx/hydra-diffusion
cd hydra-diffusion

# [OPTIONAL] create conda environment
conda create -n diffusion python=3.9
conda activate diffusion

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

[Inference](./notebooks/inference.ipynb)

## Results

Diffusion model on MNIST
<img src="./assets/mnist.gif" width="640" />

Classifier-Free Guidance Diffusion Model on MNIST
<img src="./assets/conditional_mnist.gif" width="640" />