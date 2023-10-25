import os
import pickle

import numpy as np
import pystan
import torch
from torch import tensor
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (16, 9)

nsents = [int(x) for x in open("/tmp/hmc_nsents.txt").read().splitlines()][3:]
expmus = [eval(x).numpy() / 21 for x in open("/tmp/hmc_all_expmus.txt").read().splitlines()][3:]
stdevs = [eval(x).numpy() / 21 for x in open("/tmp/hmc_all_stdevs.txt").read().splitlines()][3:]

plt.clf()
print(np.array(nsents).shape, np.stack(expmus).shape, np.stack(stdevs).shape)

for j in range(21):
  plt.errorbar(nsents, np.stack(expmus)[:,j], yerr=np.stack(stdevs)[:,j])
plt.xlim(-120, 8120)
plt.ylim(0.045, 0.0513)
plt.savefig(f"stability_of_predictions_by_number_of_regress_sentences__HMC.png")

exit(0)
