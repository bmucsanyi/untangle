# Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks

## Introduction

This repository contains code for the arXiv preprint ["Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks"](https://arxiv.org/abs/2402.19460) and also serves as a standalone benchmark suite for future methods.

The `untangle` repository is a comprehensive uncertainty quantification and uncertainty disentanglement benchmark suite that comes with
- implementations of various uncertainty quantification methods as convenient wrapper classes ... (`untangle.wrappers`)
- ... and corresponding loss functions (`untangle.losses`)
- a training script that supports these methods (`train.py`)
- an extensive evaluation suite for uncertainty quantification methods (`validate.py`)
- support for CIFAR-10 ResNet variants, including Wide ResNets (`untangle.models`)
- CIFAR-10C and ImageNet-C support (`untangle.transforms`)
- CIFAR-10H and ImageNet-ReaL support (`untangle.datasets`)
- out-of-the-box support for [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models/) (`timm`) models and configs
- TODO(bmucsanyi): plotting utilities to recreate the plots of the preprint
- TODO(bmucsanyi): scripts to reproduce the results of the preprint

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

### Datasets

CIFAR-10 is available in `torchvision.datasets` and is downloaded automatically. A local copy of the ImageNet-1k dataset is needed to run the ImageNet experiments.

The CIFAR-10H test dataset can be downloaded from [this link](https://zenodo.org/records/8115942).

The ImageNet-ReaL labels are available in [this GitHub repository](https://github.com/google-research/reassessed-imagenet). The needed files are `raters.npz` and `real.json`.

### Packages

The ImageNet-C and CIFAR-10C perturbations use [Wand](https://docs.wand-py.org/en/latest/index.html), a Python binding of [ImageMagick](https://imagemagick.org/index.php). Follow [these instructions](https://docs.wand-py.org/en/latest/guide/install.html) to install ImageMagick. Wand is installed below.

Create a virtual environment for `untangle` by running `python -m venv` (or `uv venv`) in the root folder.
Activate the virtual environment with `source .venv/bin/activate` and run one of the following commands based on your use case:
- Work with the existing code: `python -m pip install .` (or `uv pip install .`)
- Extend the code base: `python -m pip install -e '.[dev]'` (or `uv pip install -e '.[dev]'`)

## Reproducibility

We provide scripts that reproduce our results.
These are found in the `scripts` folder for both ImageNet and CIFAR-10 and are named after the respective method.

We also provide access to the exact TODO(bmucsanyi) [Singularity container](https://drive.google.com/file/d/1eYClorSZe3FMNFCZiXGLuSQJizMgPNfq/view?usp=sharing) we used in our experiments.
The `Singularity` file was used to create this container by running `singularity build --fakeroot untangle.simg Singularity`.

To recreate the plots used in the paper, use TODO(bmucsanyi) `plots/imagenet/create_main_plots.sh` for the main paper's plots and the individual scripts in the `plots` folder for all other results (incl. appendix figures). To use these utilities, you have to specify your `wandb` API key in the `WANDB_API_KEY` environment variable.

## Contributing

Contributions are very welcome. Before contributing, please make sure to run `pre-commit install`. Feel free to open a pull request with new methods or fixes.
