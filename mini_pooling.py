from cgi import print_form
from data import MNIST, BernoulliSyntheticDataset, MNISTWavelet, Synthetic, BagOfWords
from data import BetaPriorSyntheticDataset
from recovery import NA_ALISTA, IHT, NNLAD
from baseline import run_experiment_baseline, NeighborGenerator
from sensing_matrices import Pooling, ConstructedPooling
from noise import GaussianNoise, StudentTNoise, Noiseless
import numpy as np
from conf import device
import matplotlib.pyplot as plt
from train import run_experiment
import pandas as pd
import torch.nn.functional as F
from conf import device
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn


def save_log(results, name):
  if len(results) == 2:
    train_logs, test_logs = results
  else:
    test_logs = results
    train_logs = False
  pd.DataFrame(test_logs).to_csv(name + "_test.csv", index=False)
  if train_logs:
    pd.DataFrame(train_logs).to_csv(name + "_train.csv", index=False)



"""
Petersen, Jung, Bah 
"Efficient Tuning-Free ℓ1-Regression of Nonnegative Compressible Signals"
https://arxiv.org/abs/2003.13092
"""

n = 961
s = 80
m = 248
nnlad_sigma= 0.1
nnlad_tau  = 0.6

train_len = 100 # number CS problems per training epoch
test_len = 100 # number CS problems per test epoch

model      = NNLAD(20,nnlad_sigma,nnlad_tau) # NNLAD(200,nnlad_sigma, nnlad_tau)
test_model = NNLAD(100,nnlad_sigma, nnlad_tau) # NNLAD(1000, nnlad_sigma, nnlad_tau)
epochs     = 10 #50
batch_size = 1 #512
seeds      = 1 #10
noise      = StudentTNoise(40)# GaussianNoise(40);

for data in [Synthetic(n, s, s, BetaPriorSyntheticDataset, batch_size=batch_size, train_len=train_len, test_len=test_len)]:

  for seed in range(seeds):
      losses = []
      scalars = []
      if False:
        print("Determining optimal scaling factor")
        for scalar in np.linspace(0.9, 1.05, 3):
          print(scalar)
          losses.append(run_experiment(
              n=n,
              sensing_matrix=Pooling(m, n, 31, scalar, seed), 
              model=model,
              data=data,
              use_mse=False,
              train_matrix=False,
              use_median=False,
              noise=noise,
              epochs=0,
              positive_threshold=0.01,
              lr=0.0002,
              test_model=test_model,
          ))
          scalars.append(scalar)

      min_scalar=1.0 #scalars[np.argmin(losses)]
      # print('minimal scalar = %f' % scalars[np.argmin(losses)])
      print("Training Pooling Matrix using GLODISMO")
      save_log(run_experiment(
          n=n,
          sensing_matrix=Pooling(m, n, 31, min_scalar, seed), 
          model=model,
          data=data,
          use_mse=False,
          train_matrix=True,
          use_median=False,
          noise=noise,
          epochs=epochs,
          positive_threshold=0.01,
          lr=0.00002,
          test_model=test_model,
          use_greedy_stabilization=False,
      ), "results/mini_pooling_learned_"+data.name+"_"+noise.name+"_seed_"+str(seed))
      
      