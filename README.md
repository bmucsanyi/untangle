# Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks

## Introduction

This repository contains code for the NeurIPS Spotlight paper
["Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks"](https://arxiv.org/abs/2402.19460)
and also serves as a standalone benchmark suite for future methods.

The `untangle` repository is a comprehensive uncertainty quantification and uncertainty
disentanglement benchmark suite that comes with
- implementations of 19 uncertainty quantification methods as convenient wrapper classes ... (`untangle.wrappers`)
- ... and corresponding loss functions (`untangle.losses`)
- an efficient training script that supports these methods (`train.py`)
- an extensive evaluation suite for uncertainty quantification methods (`validate.py`)
- support for CIFAR-10 ResNet models, including pre-activation and Fixup variants of Wide ResNets (`untangle.models`)
- ImageNet-C and CIFAR-10C support (`untangle.transforms`)
- ImageNet-ReaL and CIFAR-10H support (`untangle.datasets`)
- out-of-the-box support for [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models/) (`timm`) models and configs
- plotting utilities to recreate the plots of the paper

If you found the paper or the code useful in your research, please cite our work as
```
@article{mucsanyi2024benchmarking,
  title={Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks},
  author={Mucs{\'a}nyi, B{\'a}lint and Kirchhof, Michael and Oh, Seong Joon},
  journal={arXiv preprint arXiv:2402.19460},
  year={2024}
}
```

If you use the benchmark, please also cite the datasets it uses.

## Installation

The package supports Python 3.11 and 3.12.
The package requirements are all listed in the `requirements.txt` file.

### Datasets

A local copy of the ImageNet-1k dataset is needed to run the ImageNet experiments.
CIFAR-10 is available in `torchvision.datasets` and is downloaded automatically.

The ImageNet-ReaL labels are available in [this GitHub repository](https://github.com/google-research/reassessed-imagenet). The needed files are `raters.npz` and `real.json`.

The CIFAR-10H test dataset can be downloaded from [this link](https://zenodo.org/records/8115942).

### Packages

The ImageNet-C and CIFAR-10C perturbations use
[Wand](https://docs.wand-py.org/en/latest/index.html), a Python binding of
[ImageMagick](https://imagemagick.org/index.php). Follow
[these instructions](https://docs.wand-py.org/en/latest/guide/install.html) to install
ImageMagick. Wand is installed below.

Create a virtual environment for `untangle` by running `python -m venv` (or `uv venv`)
in the root folder.
Activate the virtual environment with `source .venv/bin/activate` and run one of the
following commands based on your use case:
- Work with the existing code: `python -m pip install .` (or `uv pip install .`)
- Extend the code base: `python -m pip install -e '.[dev]'` (or `uv pip install -e '.[dev]'`)

## Reproducibility

The test metrics and hyperparameter sweeps used for all methods
on both ImageNet and CIFAR-10 (including the chosen hyperparameter ranges and logs) are
available on [Weights & Biases](https://wandb.ai/bmucsanyi/untangle/sweeps). Below, we provide direct links to per-method hyperparameter sweeps and final runs used in the paper for ImageNet, CIFAR-10, CIFAR-10 50%, and CIFAR-10 10%.

### ImageNet
- **CE Baseline:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/5rugaun5), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/znhyrrk6)
- **Correctness Prediction:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/v2yi16cs), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/11ueh7cq)
- **HetClassNN:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/qbzsnbl8), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/bryrtulr)
- **HET-XL:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/px1j9kqg), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/t1myokqo)
- **HET:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/jzl73q5k), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/s060twci)
- **Loss Prediction:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/xqseuogg), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/7flvihja)
- **Shallow Ensemble:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/adp0fyi8), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/pipwlaae)
- **DDU:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/ixrv6bih), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/k9myyurz)
- **MC Dropout:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/kvdgsjmc), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/1pqijue2)
- **GP:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/1yzdpvwt), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/4nr8lsd1)
- **SNGP:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/xibtpo9s), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/74rysdqf)
- **PostNet:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/yfsrcusv), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/zm0o0mo9)
- **EDL:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/08wija9l), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/gl6qgpv6)
- **Deep Ensemble:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/54kpysjy)
- **Laplace:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/42thx27s)
- **Mahalanobis:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/8a3palks/overview)
- **Temperature Scaling:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/jfnn98e3)
- **SWAG:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/o04c996o)
  - The repository contains an improved version of Automatic Mixed Precision inference, for which results are available [here](https://wandb.ai/bmucsanyi/untangle/sweeps/5yknlf4l).


### CIFAR-10
- **CE Baseline:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/cedfgnqz), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/uo3gu133)
- **Correctness Prediction:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/azcfycns), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/sgvtuzo5)
- **HetClassNN:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/pb3a6oru), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/wj1sesqf)
- **HET-XL:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/olapo0kg), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/xaz96x6d)
- **HET:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/e6rpfaue), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/w2gpx8od)
- **Loss Prediction:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/9fb4ansn), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/y5mljm78)
- **Shallow Ensemble:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/k6scvi1c), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/lcvixgvo)
- **DDU:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/5w1yjrmj), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/8apeaj9a)
- **MC Dropout:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/k4xc00mf), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/u1ozluxv)
- **GP:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/1692idyk), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/3h3gyzxj)
- **SNGP:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/jkzjb5vz), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/mhu72izt)
- **PostNet:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/8ugac5sn), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/8bqhu92u)
- **EDL:** [Hyperparameter Sweep](https://wandb.ai/bmucsanyi/untangle/sweeps/5lhhpt1y), [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/vuel80q8)
- **Deep Ensemble:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/bn1hsbqz)
- **Laplace:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/nnle8epz)
- **Mahalanobis:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/h0m0bybl)
- **Temperature Scaling:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/lh04ospw)
- **SWAG:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/zsiqsl6u)

### CIFAR-10 50%
- **CE Baseline:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/odx7l9d3)
- **Correctness Prediction:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/wtcur8hp)
- **HetClassNN:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/hk3iwjei)
- **HET-XL:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/tsrcmfce)
- **HET:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/s86d89ld)
- **Loss Prediction:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/djq57xel)
- **Shallow Ensemble:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/wxb20ctt)
- **DDU:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/6j4n8qpk)
- **MC Dropout:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/qght5cvn)
- **GP:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/2kr3j6ps)
- **SNGP:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/1gtdht1n)
- **PostNet:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/jiwagvmn)
- **EDL:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/lbo1l3tl)
- **Deep Ensemble:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/ssffhk0v)
- **Laplace:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/qwwaxtww)
- **Mahalanobis:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/pmbwo1g5)
- **Temperature Scaling:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/bqy7qulb)
- **SWAG:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/682ntyc8)

### CIFAR-10 10%
- **CE Baseline:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/qqvwtqbs)
- **Correctness Prediction:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/arat37in)
- **HetClassNN:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/9fh9jqa6)
- **HET-XL:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/jqspkx03)
- **HET:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/y66i10j7)
- **Loss Prediction:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/by5ugq6u)
- **Shallow Ensemble:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/ncwu3uqa)
- **DDU:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/ln2m618c)
- **MC Dropout:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/lelgi89v)
- **GP:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/akmjf76i)
- **SNGP:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/71vco142)
- **PostNet:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/s8837u8b)
- **EDL:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/3b6vc8jo)
- **Deep Ensemble:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/6dp9q2x8)
- **Laplace:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/ph5n7ds1)
- **Mahalanobis:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/g12uymou)
- **Temperature Scaling:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/4f8j92n5)
- **SWAG:** [Final Results](https://wandb.ai/bmucsanyi/untangle/sweeps/hctd423o)

We also provide access to the exact [Singularity container](https://drive.google.com/file/d/1ZhVY-3IAfU93F2B9wdYXxsGWsXj4-X8L/view?usp=sharing)
we used in our experiments. The `Singularity` file was used to create this container by
running `singularity build --fakeroot untangle.simg Singularity`.

To recreate the plots used in the paper, please refer to the [Plotting Utilities](#plotting-utilities) section below.

## Results

The results of our experiments are available in full on [Weights & Biases](https://wandb.ai/bmucsanyi/untangle/sweeps)
and in our [paper](https://arxiv.org/abs/2402.19460). The Weights & Biases results
can either be queried online or through our plotting utilities.

### Online Access
To access our results from the Weights & Biases console, click any of the "Test" links
above and type any metric from the [Metrics](#metrics) section.

### <a name="plotting_utilities"></a>Plotting Utilities
As our [paper](https://arxiv.org/abs/2402.19460) contains more than 50 plots, we provide
general plotting utilities that allow to visualize results on any metric as opposed to
providing scripts to reproduce a particular plot. These utilities are found in the
`plots` folder.
- `plot_ranking.py`: Our main plotting script that generates bar plots of method rankings on a particular metric. Needs a dataset (`imagenet` or `cifar10`), the label of the y axis, and the metric (see [Metrics](#metrics)). The script has several optional arguments to have fine-grained control over the plot, as documented in the source code.
- `plot_ranking_it.py`: Variant of `plot_ranking.py` which uses the information-theoretical decomposition's estimators for all methods. Can only be used with the `auroc_oodness` and `rank_correlation_bregman_au` metrics.
- `plot_ranking_shallow.py`: Variant of `plot_ranking.py` which uses the information-theoretical decomposition's estimators for the Shallow Ensemble method and the best-performing one for the others. The Shallow Ensemble gives the most decorrelated aleatoric and epistemic estimates withthe information-theoretical decomposition, and the plots checks how practically relevant the resulting estimates are. Can only be used with the `auroc_oodness` and `rank_correlation_bregman_au` metrics.
- `plot_full_correlation_matrix.py`: Calculates the correlation of method rankings across different metrics. Needs only a dataset (`imagenet` or `cifar10`).
- `plot_correlation_matrix.py`: Variant of `plot_full_correlation_matrix.py` that calculates the correlations between a smaller set of metrics.
- `plot_estimator_correlation_matrix.py`: Variant of `plot_full_correlation_matrix.py` that only calculates the correlations w.r.t. one estimator. This estimator can be either `one_minus_max_probs_of_dual_bma`, `one_minus_max_probs_of_bma`, or `one_minus_expected_max_probs`.
- `plot_correlation_datasets.py`: Prints correlation statistics of rankings on different metrics across different datasets (CIFAR-10 and ImageNet).
- `plot_correctness_robustness.py`: Plots the performance of estimators and methods on in-distribution and increasingly out-of-distribution datasets w.r.t. the accuracy, correctness AUROC, and AUAC metrics. Requires only a dataset (`imagenet` or `cifar10`).
- `plot_calibration_robustness.py`: Plots the robustness of the Laplace, Shallow Ensemble, EDL, and CE Baseline methods on the ECE metrics when going from in-distribution data to out-of-distribution data of severity level one. Requires only a dataset (`imagenet` or `cifar10`).

### <a name="metrics"></a>Metrics
In this section, we provide a (non-exhaustive) list of descriptions of several metrics we consider in our benchmarks. Their codes can be either used to search for results on the online [Weights & Biases](https://wandb.ai/bmucsanyi/untangle/sweeps) console or in our plotting utilities.
- `auroc_hard_bma_correctness_original`: AUROC of uncertainty estimators w.r.t. the Bayesian Model Average's correctness and the original hard (i.e., one-hot) labels.
- `auroc_soft_bma_correctness`: AUROC of uncertainty estimators w.r.t. the Bayesian Model Average's binary correctness indicators and the soft labels (either ImageNet-ReaL or CIFAR-10H).
- `auroc_oodness` AUROC of uncertainty estimators w.r.t. the binary OOD indicators on a balanced mixture of ID and OOD data.
- `hard_bma_accuracy_original`: Accuracy of the Bayesian Model Average w.r.t. the original hard (i.e., one-hot) labels.
- `cumulative_hard_bma_abstinence_auc`: AUAC value of uncertainty estimators w.r.t. the Bayesian Model Average.
- `log_prob_score_hard_bma_correctness_original`: The log probability proper scoring rule of uncertainty estimators w.r.t. the Bayesian Model Average's correctness on the original hard (i.e., one-hot) labels.
- `brier_score_hard_bma_correctness`: The Brier score of uncertainty estimators w.r.t. the Bayesian Model Average's correctness on the original hard (i.e., one-hot) labels.
- `log_prob_score_hard_bma_aleatoric_original` The log probability proper scoring rule of the Bayesian Model Average's predicted probability vector w.r.t. the ground-truth original hard (i.e., one-hot) labels. A.k.a. the log-likelihood of the labels under the model.
- `brier_score_hard_bma_aleatoric_original` The Brier score of the Bayesian Model Average's predicted probability vector w.r.t. the ground-truth original hard (i.e., one-hot) labels. A.k.a. the negative L2 loss of the model's predictions.
- `rank_correlation_bregman_au`: The rank correlation of uncertainty estimators with the groud-truth aleatoric uncertainty values from the Bregman decomposition.
- `rank_correlation_bregman_b_dual_bma`: The rank correlation of uncertainty estimators with the bias values from the Bregman decomposition (which uses the Dual Bayesian Model Average instead of the primal one).
- `rank_correlation_it_au_eu`: The rank correlation of the information-theoretical aleatoric and epistemic estimates.
- `rank_correlation_bregman_eu_au_hat`: The rank correlation of the Bregman decomposition's epistemic estimates with the aleatoric estimates predicted by the model.
- `rank_correlation_bregman_au_b_dual_bma`: The rank correlation of the Bregman decomposition's aleatoric and bias ground-truth values.
- `ece_hard_bma_correctness_original`: ECE of uncertainty estimators w.r.t. the Bayesian Model Average's correctness and the original hard (i.e., one-hot) labels.

For more metrics, please refer to `validate.py`.

## Contributing

Contributions are very welcome. Before contributing, please make sure to run
`pre-commit install`. Feel free to open a pull request with new methods or fixes.
