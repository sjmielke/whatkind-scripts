import os
import pickle
import sys

import numpy as np
import pystan
import torch
from matplotlib import pyplot as plt
import datetime


# Switch backend for cluster
if os.path.exists("/home/smielke"):
  plt.switch_backend('agg')
else:
  assert os.path.exists("/home/sjm")

NA_VALUE = 101

class SuppressStdoutStderr(object):
  '''
  A context manager for doing a "deep suppression" of stdout and stderr in
  Python, i.e. will suppress all print, even if the print originates in a
  compiled C/Fortran sub-function.

  This will not suppress raised exceptions, since exceptions are printed
  to stderr just before a script exits, and after the context manager has
  exited (at least, I think that is why it lets exceptions through).

  https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
  '''

  def __init__(self):
    # Open a pair of null files
    self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    # Save the actual stdout (1) and stderr (2) file descriptors.
    self.save_fds = [os.dup(1), os.dup(2)]

  def __enter__(self):
    # Assign the null pointers to stdout and stderr.
    os.dup2(self.null_fds[0], 1)
    os.dup2(self.null_fds[1], 2)

  def __exit__(self, *_):
    # Re-assign the real stdout/stderr back to (1) and (2)
    os.dup2(self.save_fds[0], 1)
    os.dup2(self.save_fds[1], 2)
    # Close all file descriptors
    for fd in self.null_fds + self.save_fds:
      os.close(fd)


class MultiLMStanModel():
  MODEL_SHARED_DATA_PARAMS = """
  data {
    int<lower=0> NSENTS; // number of utterances
    int<lower=0> NLANGS; // number of languages
    matrix[NSENTS,NLANGS] log_y; // log (!) of the observed bits
  }
  parameters {
    simplex[NLANGS] expmus;
    real<lower=0> sigma;
    row_vector<lower=0>[NSENTS] n;
  }
  """

  MODEL1_MODEL = """
  model {
    {{PRIORS}}
    for (j in 1:NLANGS)
      log_y[:,j] ~ normal(n + log(NLANGS) + log(expmus[j]), sigma);
  }
  """
  MODEL2_MODEL = """
  model {
    row_vector[NSENTS] fw_sigma_squared;
    row_vector[NSENTS] fw_mu;
    {{PRIORS}}
    fw_sigma_squared = log1p(expm1(sigma^2) ./ n);
    fw_mu = log(n) + (sigma^2 - fw_sigma_squared) / 2;
    for (j in 1:NLANGS)
      log_y[:,j] ~ normal(fw_mu + log(NLANGS) + log(expmus[j]), sqrt(fw_sigma_squared));
  }
  """

  def __init__(self, model, *args, **kwargs):
    # Pick the right model code
    self.model_name = model
    # Basic model choice
    self.model_code = self.MODEL_SHARED_DATA_PARAMS
    if model.split("_")[0] == "MODEL1":
      self.model_code += self.MODEL1_MODEL
    elif model.split("_")[0] == "MODEL2":
      self.model_code += self.MODEL2_MODEL
    else:
      raise Exception("Wrong model " + model.split("_")[0])
    # Expmus from the simplex?
    if model.split("_")[1] == "SIMPLEX":
      self.pars = ['expmus', 'n', 'sigma']
    elif model.split("_")[1] == "SCALEDSIMPLEX":
      self.model_code = self.model_code.replace("bits", "bits\n    real<lower=0> dscale;")
      self.model_code = self.model_code.replace(" + log(NLANGS)", " + log(NLANGS) + log(dscale)")
      self.pars = ['expmus', 'n', 'sigma']
    elif model.split("_")[1] == "SCALEDSIMPLEXLEARNED":
      self.model_code = self.model_code.replace("simplex[NLANGS] expmus;", "real log_dscale;\n    simplex[NLANGS] expmus;")
      self.model_code = self.model_code.replace(" + log(NLANGS)", " + log(NLANGS) + log_dscale")
      self.pars = ['expmus', 'n', 'sigma', 'log_dscale']
    else:
      raise Exception("Wrong expmu constraint " + model.split("_")[1])
    # With prior?
    if "PRIOREXPMUS" in model:
      self.model_code = self.model_code.replace("data {", "data {\n    real<lower=0> alpha;")
      self.model_code = self.model_code.replace("{{PRIORS}}", "{{PRIORS}}\n    expmus * NLANGS ~ normal(1, alpha);")
    if "PRIORNS" in model:
      self.model_code = self.model_code.replace("data {", "data {\n    real<lower=0> a;\n    real<lower=0> b;")
      self.model_code = self.model_code.replace("{{PRIORS}}", "{{PRIORS}}\n    n ~ gamma(a, b);")
    self.model_code = self.model_code.replace("{{PRIORS}}\n    ", "")
    # Laplacian instead of Gaussian?
    if "LAPLACIAN" in model:
      self.model_code = self.model_code.replace("normal", "double_exponential")
      self.model_code = self.model_code.replace("sqrt(fw_sigma_squared)", "sqrt(fw_sigma_squared) / sqrt(2)")  # it only matters here
    if "CAUCHY" in model:
      self.model_code = self.model_code.replace("normal", "cauchy")
    # Fix language parameters and global parameters?
    if "FIXGLOBALS" in model:
      self.model_code = self.model_code.replace("\n  }\n  parameters {", "").replace("  real<lower=0> sigma;", "  real<lower=0> sigma;\n  }\n  parameters {\n")
      self.pars = [p for p in self.pars if p not in ['expmus', 'sigma', 'log_dscale']]
    elif "FIXEXPMUS" in model:
      self.model_code = self.model_code.replace("\n  }\n  parameters {", "").replace("  simplex[NLANGS] expmus;", "  simplex[NLANGS] expmus;\n  }\n  parameters {\n")
      self.pars = [p for p in self.pars if p not in ['expmus', 'log_dscale']]
    
    # Translationese special model
    if "TRANSLATIONESE" in model:
      codelines = self.model_code.split('\n')
      assert codelines[-3].startswith("      log_y[:,j] ~")
      self.model_code = '\n'.join(codelines[:-3]) + "\n      for (i in 1:NSENTS)\n        if (log_y[i,j] < 100)\n          log_y[i,j] ~ normal(fw_mu[i] + log(NLANGS) + log_dscale + log(expmus[j]), sqrt(fw_sigma_squared[i]));\n  }\n"

    # Try to open cached model, otherwise cache
    try:
      with open(f"{model}.stanmodel", 'rb') as file:
        (loaded_model_code, self.stanmodel) = pickle.load(file)
      assert loaded_model_code == self.model_code, "Cached code differs from current code! Failing now."
      # print(self.model_name)
      # print(self.model_code)
      # exit(0)
    except FileNotFoundError:
      print("Compiling", model)
      with SuppressStdoutStderr():
        self.stanmodel = pystan.StanModel(model_code = self.model_code, *args, **kwargs)
      with open(f"{model}.stanmodel", 'wb') as file:
        pickle.dump((self.model_code, self.stanmodel), file)

  def optimizing(self, *args, **kwargs):
    with SuppressStdoutStderr():
      return self.stanmodel.optimizing(*args, **kwargs)

  def sampling(self, *args, **kwargs):
    with SuppressStdoutStderr():
      return self.stanmodel.sampling(*args, **kwargs)

  def good_init(self, datadict):
    n = np.ndarray((datadict['NSENTS'],))
    for i in range(datadict['NSENTS']):
      lys = datadict['log_y'][i, :]
      n[i] = (lys[lys != NA_VALUE]).exp().mean()
    init = {
        'expmus': (torch.ones(datadict['NLANGS'], dtype=torch.double) / datadict['NLANGS']).numpy(),
        'n': n,
        'sigma': 1.0,
    }
    if self.model_name.split("_")[1] == "SCALEDSIMPLEXLEARNED":
      init['log_dscale'] = 1.0
    return init

  def map_run(self, data, iter, init="good_init", *args, **kwargs):
    if init == "good_init":
      init = self.good_init(data)
    elif init is None:
      init = "random"

    return self.optimizing(data=data, init=init, iter=iter, *args, **kwargs)

  def hmc_run(self, datadict, outputdir, nchains = 1, iters = 1000, warmup = 200, init="good_init"):
    # Sample, initalizing with MAP estimate
    map_estimate = self.optimizing(data=datadict, iter=10000, init=init)
    # print(datadict)
    # print(self.model_code)
    # exit(0)
    fit = self.sampling(data=datadict, pars=self.pars, iter=iters + warmup, warmup=warmup, chains=nchains, init=[map_estimate for i in range(nchains)], check_hmc_diagnostics=False)

    if outputdir is not None:
      # Output entire fit
      os.makedirs(outputdir, exist_ok=True)
      with open(f"{outputdir}/fit.txt", 'w') as outfile:
        print(fit, file=outfile)
      plt.rcParams["figure.figsize"] = (16, 9)
      fit.plot()
      plt.savefig(f"{outputdir}/fit.png")

      # Extract expmu data
      fit_extract = fit.extract()
      expmus = torch.tensor(fit_extract['expmus'])
      with open(f"{outputdir}/expmu_mean.txt", 'w') as outfile:
        print(' '.join([str(f.item()) for f in expmus.mean(dim=0)]), file=outfile)
      with open(f"{outputdir}/expmu_stdev.txt", 'w') as outfile:
        print(' '.join([str(f.item()) for f in expmus.std(dim=0)]), file=outfile)

      # Plot expmus
      (figsize_x, figsize_y) = plt.rcParams["figure.figsize"]
      plt.rcParams["figure.figsize"] = (160, 90)
      plt.close()
      (_fig, axes) = plt.subplots(datadict['NLANGS'], datadict['NLANGS'], sharex=True, sharey=True)
      for x in range(datadict['NLANGS']):
        for y in range(datadict['NLANGS']):
          axes[x][y].scatter(expmus[:, x].numpy(), expmus[:, y].numpy(), alpha = 25 / iters)
      plt.savefig(f"{outputdir}/expmus_correlation.png")

    return fit


class MultiLMDataset(object):
  def __init__(self, dataset, bpefactor = None, translationese = False, native_langs_file = None, split_languages = None):
    with open(dataset) as f:
      self.langs = next(f).split()
    self.log_y = torch.from_numpy(np.loadtxt(dataset, delimiter='\t', skiprows=1)).log()

    if translationese:
      with open(native_langs_file) as file:
        native_langs = file.read().splitlines()
      assert len(native_langs) == self.log_y.size(0)
      lang2idx = {l: i for i, l in enumerate(self.langs)}
      self.langs += [lang + "_native" for lang in split_languages]
      old_log_y = self.log_y
      self.log_y = torch.empty(self.log_y.size(0), len(self.langs))
      for i, native_lang in enumerate(native_langs):
        for j, sentence_lang in enumerate(self.langs):
          if sentence_lang.endswith("_native"):
            self.log_y[i, j] = old_log_y[i, lang2idx[sentence_lang[:2]]] if sentence_lang[:2] == native_lang else NA_VALUE
          else:
            self.log_y[i, j] = old_log_y[i, j] if sentence_lang[:2] != native_lang else NA_VALUE
    torch.manual_seed(0)
    # torch.manual_seed(int(sys.argv[1]) * 1000)
    self.log_y = self.log_y[torch.randperm(self.log_y.size(0)), :]

  def datadict(self, n_sentences, skip_sentences = 0, add_params = None):
    datadict = {
        'log_y': self.log_y[skip_sentences : skip_sentences + n_sentences, :],
        'NSENTS': min(n_sentences, self.log_y.size(0)),
        'NLANGS': self.log_y.size(1)
    }
    if add_params is not None:
      for k, v in add_params.items():
        datadict[k] = v
    return datadict


# # Compile all models
# models = [m + f for f in ["", "_FIXGLOBALS", "_FIXEXPMUS"] for m in ["MODEL1_SCALEDSIMPLEXLEARNED", "MODEL2_SCALEDSIMPLEXLEARNED", "MODEL2_SCALEDSIMPLEXLEARNED_LAPLACIAN"]]
# _, _, _ = MultiLMStanModel(models[0]), MultiLMStanModel(models[1]), MultiLMStanModel(models[2])
# _, _, _ = MultiLMStanModel(models[3]), MultiLMStanModel(models[4]), MultiLMStanModel(models[5])
# _, _, _ = MultiLMStanModel(models[6]), MultiLMStanModel(models[7]), MultiLMStanModel(models[8])
# exit(0)


# # Test HMC
# # for model in ["MODEL1_SIMPLEX", "MODEL1_VECTOR", "MODEL2_SIMPLEX", "MODEL2_VECTOR"]:
# for model in ["MODEL2_SCALEDSIMPLEXLEARNED"]:
#   MultiLMStanModel(model).hmc_run(
#       MultiLMDataset("europarl-bpe", bpefactor="04").datadict(n_sentences=250),
#       outputdir="hmc_results/ep-bpe-04-" + model + "-250sents-3000iters-200-warmup-2chains__", iters=5000, warmup=5000, nchains=1)
# exit(0)

# dataset = MultiLMDataset(f"ALL_13kEP_chars_adam_ptb.tsv")
# dataset = MultiLMDataset(f"ALL_13kEP_bpe_04_pelikan.tsv")
# dataset = MultiLMDataset(f"ALL_13kEP_bpe_best.tsv")
# dataset = MultiLMDataset(f"ALL_4336bibles_chars_adam_ptb.tsv")
# dataset = MultiLMDataset(f"ALL_4336bibles_bpe_04_pelikan.tsv")
# model = MultiLMStanModel("MODEL2_SCALEDSIMPLEXLEARNED")
# fit = model.map_run(data=dataset.datadict(n_sentences=999999), iter=1000000, init="good_init")
# results = list(zip(dataset.langs, len(dataset.langs) * fit['expmus']))
# print(results)
# exit(0)

# dataset = MultiLMDataset(f"ALL_13kEP_chars_adam_ptb.tsv", translationese=True, native_langs_file = "/home/sjm/projects/JHU/variance-and-variation/datasets/europarl_native_langs_in_test", split_languages = ["en", "fr", "de", "pt", "it", "es", "ro", "pl"])
# dataset = MultiLMDataset(f"ALL_13kEP_bpe_04_pelikan.tsv", translationese=True, native_langs_file = "/home/sjm/projects/JHU/variance-and-variation/datasets/europarl_native_langs_in_test", split_languages = ["en", "fr", "de", "pt", "it", "es", "ro", "pl"])
# dataset = MultiLMDataset(f"ALL_13kEP_bpe_best.tsv", translationese=True, native_langs_file = "/home/sjm/projects/JHU/variance-and-variation/datasets/europarl_native_langs_in_test", split_languages = ["en", "fr", "de", "pt", "it", "es", "ro", "pl"])
# dataset = MultiLMDataset(f"ALL_13kEP_chars_adam_ptb.tsv")

# for split_lang in ["da", "de", "en", "es", "fi", "fr", "it", "nl", "pt", "sv"]:
#   dataset = MultiLMDataset(f"translationese_balanced/ALL_europarl_balanced_{split_lang}.tsv", translationese=True, native_langs_file=f"translationese_balanced/test_native_lang_{split_lang}", split_languages = [split_lang])
#   model = MultiLMStanModel("MODEL2_SCALEDSIMPLEXLEARNED_TRANSLATIONESE")
#   fit = model.map_run(data=dataset.datadict(n_sentences=999999), iter=1000000, init="good_init")
#   results = list(zip(dataset.langs, len(dataset.langs) * fit['expmus']))
#   print(results)
#   sl_n = dict(results)[split_lang + "_native"]
#   sl_t = dict(results)[split_lang]
#   print(split_lang, ":", sl_n, "/", sl_t, "=", sl_n/sl_t)
# exit(0)

# # Get MAP estimates in line-follow plot
# # diffmat = np.ndarray((3, 21))
# # for rnn_i, rnn in enumerate(["chars", "BPE_04", "BPE_best"]):
# #   dataset = MultiLMDataset(f"ALL_13k_EP_{rnn}.tsv")
# # diffmat = np.ndarray((2, 106))
# # for rnn_i, rnn in enumerate(["chars_adam_ptb", "bpe_04_pelikan"]):
# #   dataset = MultiLMDataset(f"ALL_4336bibles_{rnn}.tsv")
diffmat = np.ndarray((2, 4))
for rnn_i, rnn in enumerate(["chars_adam_ptb", "bpe_04_pelikan"]):
  dataset = MultiLMDataset(f"ALL_nist_{rnn}.tsv")
  model = MultiLMStanModel("MODEL2_SCALEDSIMPLEXLEARNED")
  fit = model.map_run(data=dataset.datadict(n_sentences=999999), iter=100000, init="good_init")
  diffmat[rnn_i, :] = len(dataset.langs) * fit['expmus']  # np.exp(fit['log_dscale']) * fit['expmus']
  print("Setting", rnn, len(dataset.langs) * fit['expmus'])

# for rnn_i, rnn in enumerate(["chars", "BPE (0.4)"]):  #, "BPE (best)"]):
#   print(f"\\node at (-5, {-2.4 * rnn_i}) {{{rnn}}};")
for lang_id, lang in enumerate(dataset.langs):
  # intensity = int((diffmat[:, lang_id].std() * 3000).clip(max=80)) + 20
  # print(f"\\node[rotate=90] at ({(diffmat[0, lang_id] - 1.0) * 50}, .3) {{\\footnotesize\\color{{black!{intensity}}} {lang}}};")
  # print(f"\\node[rotate=90] at ({(diffmat[2, lang_id] - 1.0) * 50}, {-2.5*3-.3}) {{\\footnotesize\\color{{black!{intensity}}} {lang}}};")
  # print(f"\\draw[draw=black!{intensity},thick] plot [smooth,tension=0.3] coordinates {{", end='')
  # print(" ".join([f"({(d - 1.0) * 50}, {-2.5 * rnn_i})" for rnn_i, d in enumerate(diffmat[:, lang_id])]), end='};\n')
  intensity = int((diffmat[:, lang_id].std() * 1000).clip(max=80)) + 20
  print(f"\\draw[draw=black!{intensity},thick] plot coordinates {{", end='')
  print(" ".join([f"({(d - 1.0) * 50}, {-2.5 * rnn_i})" for rnn_i, d in enumerate(diffmat[:, lang_id])]), end='};\n')
exit(0)

# Get MAP estimates in scatter plot
import itertools
dss = itertools.repeat([])
corpus, langcolor = "4336bibles", {'deu': 'black!90', 'eng': 'brown!90', 'fra': 'olive!90'}
# corpus, langcolor = "13k_EP", {}
for rnn in ["bpe_04_pelikan", "chars_adam_ptb"]:
  dataset = MultiLMDataset(f"ALL_{corpus}_{rnn}.tsv")
  model = MultiLMStanModel("MODEL2_SCALEDSIMPLEXLEARNED")
  fit = model.map_run(data=dataset.datadict(n_sentences=999999), iter=100000, init="good_init")
  dss = [c + [x] for c, x in zip(dss, len(dataset.langs) * fit['expmus'])]  # np.exp(fit['log_dscale']) * fit['expmus'])]
langs = sorted(list(set([l[:3] for l in dataset.langs])))
for rnn_i in [0, 1]:
  print("RNN", rnn_i)
  avg = np.mean([ds[rnn_i] for l, ds in zip(dataset.langs, dss)])
  std = np.std([ds[rnn_i] for l, ds in zip(dataset.langs, dss)])
  print("Global avg", avg, "std", std)
  for lang in ["deu", "fra", "eng"]:
  # for lang in langs:
    avg = np.mean([ds[rnn_i] for l, ds in zip(dataset.langs, dss) if l[:3] == lang])
    std = np.std([ds[rnn_i] for l, ds in zip(dataset.langs, dss) if l[:3] == lang])
    print("Lang", lang, "avg", avg, "std", std, "from", [ds[rnn_i] for l, ds in zip(dataset.langs, dss) if l[:3] == lang])
for l, ds in zip(dataset.langs, dss):
  print(f"\\addplot[{langcolor.get(l[:3], 'red!20' if ds[0] > ds[1] else 'blue!20')}, mark=text, text mark={l[:3]}] coordinates {{({ds[0]}, {ds[1]})}};")
  if l[:3] == "deu":
    print("%", l, "bpe", ds[0], "char", ds[1])
exit(0)

# Goodness of fit by likelihood
# models = ["MODEL1_SIMPLEX", "MODEL2_SIMPLEX", "MODEL2_SIMPLEX_LAPLACIAN", "MODEL1_SCALEDSIMPLEXLEARNED", "MODEL2_SCALEDSIMPLEXLEARNED", "MODEL2_SCALEDSIMPLEXLEARNED_LAPLACIAN"]
models = ["MODEL2_SIMPLEX_LAPLACIAN", "MODEL2_SCALEDSIMPLEXLEARNED_LAPLACIAN"]
# models = ["MODEL2_SCALEDSIMPLEXLEARNED"]
initial_skip = int(sys.argv[1])
outputs = "Run started " + str(datetime.datetime.now()) + "\nmodel\teval_fit_proposed\teval_fit_random\teval_hmc_proposed.mean()\teval_hmc_random.mean()\teval_hmc_proposed.std()\teval_hmc_random.std()"
for model_id, modelname in enumerate(models):
  for fixation in ["_FIXGLOBALS", "_FIXEXPMUS"]:
    logliks = []
    # dataset = MultiLMDataset("ALL_13k_EP_BPE_04.tsv")
    dataset = MultiLMDataset("ALL_no_mri_4336_bibles_chars.tsv")
    model_fitter = MultiLMStanModel(modelname)
    model_evaler = MultiLMStanModel(modelname + fixation)
    data_for_fitting = dataset.datadict(n_sentences=3000, skip_sentences=initial_skip)
    data_for_eval = dataset.datadict(n_sentences=1000, skip_sentences=initial_skip+3000)
    global_fit = model_fitter.map_run(data=data_for_fitting, iter=100000, init="good_init")
    data_for_eval.update(global_fit)
    proposed_new_ns = ((global_fit['n'] / data_for_fitting['log_y'].exp().mean(dim=1)).mean() * data_for_eval['log_y'].exp().mean(dim=1)).numpy()
    eval_fit_proposed = model_evaler.map_run(data=data_for_eval, iter=100000, init = {'n': proposed_new_ns}, as_vector = False)['value']
    eval_fit_random = model_evaler.map_run(data=data_for_eval, iter=100000, init = "random", as_vector = False)['value']
    eval_hmc_proposed = model_evaler.hmc_run(datadict=data_for_eval, outputdir=None, iters=300, warmup=200, nchains=3, init = {'n': proposed_new_ns}).extract()['lp__']
    eval_hmc_random = model_evaler.hmc_run(datadict=data_for_eval, outputdir=None, iters=300, warmup=200, nchains=3, init = "random").extract()['lp__']
    outputs += f"\n{modelname + fixation}\t{eval_fit_proposed}\t{eval_fit_random}\t{eval_hmc_proposed.mean()}\t{eval_hmc_random.mean()}\t{eval_hmc_proposed.std()}\t{eval_hmc_random.std()}"
    print(outputs)
    with open(f"SUPERRUN_skip_{initial_skip}.log", 'w') as f:
      print(outputs, file=f)
print(outputs)
exit(0)


# How many sentences do we need for stable MAP estimates?
ns = list(range(1000, 8000, 250))
dataset = MultiLMDataset("europarl-bpe", bpefactor="04")
for a, b in [(2.50, 0.248)]:
  datas = [dataset.datadict(n_sentences=n, add_params={'a': a, 'b': b}) for n in ns]
  # for modelname in ["MODEL2_SIMPLEX_PRIORNS"]:
  for modelname in ["MODEL2_SIMPLEX", "MODEL2_SIMPLEX_LAPLACIAN"]:
    expmus = []
    model = MultiLMStanModel(modelname)
    for n, data in list(zip(ns, datas))[:15]:
      print(f"Regressing (MAP) on n = {n} sentences")
      fit = model.map_run(data=data, iter=100000)
      expmus.append(fit['expmus'])
      # print(fit['dscale'])
      plt.rcParams["figure.figsize"] = (16, 9)
      plt.close()
      plt.plot(ns, np.stack(expmus + [np.ones((21,)) / 21] * (len(ns) - len(expmus))))
      plt.savefig(f"stability_of_MAP_by_number_of_regress_sentences__{modelname}_shuffled0.png")
    # plt.rcParams["figure.figsize"] = (16, 9)
    # plt.close()
    # plt.plot(ns, np.stack(expmus))
    # plt.savefig(f"stability_of_MAP_by_number_of_regress_sentences__{modelname}_shuffled0.png")
exit(0)


# How many sentences do we need for stable posterior means?
ns = list(range(250, 8000, 250))
dataset = MultiLMDataset("europarl-bpe", bpefactor="04")
datas = [dataset.datadict(n_sentences=n) for n in ns]
for modelname in ["MODEL2_SIMPLEX"]:
  expmus = []
  expmu_stdevs = []
  model = MultiLMStanModel(modelname)
  for n, data in zip(ns, datas):
    print(f"Regressing (MAP+HMC) on n = {n} sentences")
    fit = model.hmc_run(data, outputdir=None, iters=250, warmup=200, nchains=2)
    expmus.append(fit.extract()['expmus'].mean(axis=0))
    expmu_stdevs.append(fit.extract()['expmus'].std(axis=0))
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.close()
    # plt.plot(ns, np.stack(expmus + [np.ones((21,)) / 21] * (len(ns) - len(expmus))), np.stack(expmu_stdevs))
    for i in range(21):
      plt.errorbar(ns,
        np.array(list(np.stack(expmus)[:, i]) + [1 / 21] * (len(ns) - len(expmus))),
        np.array(list(np.stack(expmu_stdevs)[:, i]) + list(np.zeros((len(ns) - len(expmu_stdevs), )))),
        capsize = 3)
    plt.savefig(f"stability_of_HMC_by_number_of_regress_sentences__{modelname}_shuffled.png")
exit(0)


# Do the mu estimates change for BPE changes?
MODEL = MultiLMStanModel("MODEL2")
print(torch.from_numpy(np.stack([MODEL.map_run(data=MultiLMDataset("europarl-bpe", bpefactor=bpefactor).datadict(n_sentences=2000), iter=100000)['expmus'] for bpefactor in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]])))


# for n in 1250 1750 2250 2750 3250 3500 3750 4000 4250 4500 4750 5000; do echo $'#!/usr/bin/env bash\ncd variance-and-variation\npython stanny.py' $n > run_$n.bash; qsub -l 'hostname=b*|c*' -l 'mem_free=20G,ram_free=20G' run_$n.bash; done
