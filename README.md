# AllenNLP Multi-label Document Classification Plugin

A simple multi-label document classification plugin for [AllenNLP](https://allennlp.org/).

## Installation

This repository requires Python 3.7 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

First, clone the repository locally

```bash
git clone https://github.com/JohnGiorgi/allennlp-multi-label-document-classification.git
```

Then, install

```bash
cd allennlp-multi-label-document-classification
pip install --editable .
```

For the time being, please install [AllenNLP](https://github.com/allenai/allennlp) [from source](https://github.com/allenai/allennlp#installing-from-source). You should also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

#### Enabling mixed-precision training

If you want to train with [mixed-precision](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) (strongly recommended if your GPU supports it), you will need to [install Apex with CUDA and C++ extensions](https://github.com/NVIDIA/apex#quick-start). Once installed, you need only to set `"opt_level"` to `"O1"` in your training [config](configs), or, equivalently, pass the following flag to `allennlp train` (see [Training](#training))

```bash
--overrides "{'trainer.opt_level': 'O1'}"
```

## Usage

### Preparing a dataset

Datasets should be JSON lines files, where each line is valid JSON containing the fields `"text"` and `"labels"`. You can specify different partitions in the [configs](configs) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

### Training

To train the model, run the following command

```bash
allennlp train configs/multi_label_classifier.jsonnet \
    -s output \
    -o "{'train_data_path': 'path/to/input.txt'}" \
    --include-package src
```

During training, models, vocabulary, configuration and log files will be saved to `output`. This can be changed to any path you like.

#### Multi-GPU training

To train on more than one GPU, provide a list of CUDA devices in your training [config](configs) under `"distributed.cuda_devices"`, or, equivalently, pass the following flag to `allennlp train`

```bash
--overrides "{'distributed.cuda_devices': [0, 1, 2, 3]}"
```

This would train your model on four CUDA devices with IDs `0, 1, 2, 3`.

### Inference

Coming soon.