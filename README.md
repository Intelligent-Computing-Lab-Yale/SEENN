# SEENN
Pytorch implementation of [SEENN](https://arxiv.org/pdf/2304.01230.pdf) (Spiking Early Exit Neural Networks) (NeurIPS 2023), and [DT-SNN](https://arxiv.org/pdf/2305.17346.pdf) (Dynamic-Timestep Spiking Neural Network) (Design & Automation Conference 2023)

### Requirements

Generally the code require `Pytorch>1.8` and other common Python library like `PIL, Numpy`

The code adopts some implementation in [TET](https://github.com/brain-intelligence-lab/temporal_efficient_training).

### Training SNNs

We provide SNN training code in `train_snn.py`, for experiments reproduction, we have fixed some hyper-parameters in the file.
You can specify *dataset, model architecture*. Note that in-default we have turned on the TET loss function and AMP training by setting `store_false`. 
For example:

To train a ResNet-19 on CIFAR10 dataset, run

`python train_snn.py --dset c10 --model res20`

If you wish to disable TET loss function and AMP training, run

`python train_snn.py --dset c10 --model res20 --TET --amp`

### Evaluating SEENN-I

We have provided the evaluation code for SEENN-I, after training an SNN and saving a checkpoint,
you can run `test_snn.py` for SEENN-I performance evaluation. 

Here are the arguments that may be useful:

`-T` The maximum #timesteps, should be identical to the one used in training

`--t` The threshold of confidence/entropy 

`--test-all` If evaluate the accuracy from 1 to T

`--gpu-test` If test the throughput/latency 

`--aet-test` If calculate the AET values


### Evaluating SEENN-II

To be updated