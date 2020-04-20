# Generative Teaching Networks

Official PyTorch implementation of [Generative Teaching Networks](https://arxiv.org/abs/1912.07768). For research purpose only. Support and/or new releases may be limited.


### Setup
Clone the repo:
```
git clone https://github.com/uber-research/GTN.git && cd GTN
```

We use Python 3.6.2. Requirements can be installed by running:
```
pip install -r requirements.txt
```

There is a problem with torch>1.2 so we have a hard requirement on torch==1.2.

After installing the dependencies you can run `pip install -e .` to compile a custom torch kernel.

## Training on MNIST

To train on MNIST simply run ``python train_cgtn.py``. This command reads ``experiments/cgtn.json`` to get arguments.


## Architecture Search on CIFAR10

The architecture search experiments is a separate component based on [NAO](https://github.com/renqianluo/NAO_pytorch) which has a different LICENSE agreement. See ``architecture_search/`` for more details.