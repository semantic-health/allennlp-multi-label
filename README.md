![build](https://github.com/JohnGiorgi/allennlp-multi-label-classification/workflows/build/badge.svg?branch=master)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/allennlp-multi-label-classification?color=blue)

# AllenNLP Multi-label Classification Plugin

A multi-label classification plugin for [AllenNLP](https://allennlp.org/).

## Installation

This repository requires Python 3.6.1 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. See [here](https://github.com/allenai/allennlp#installing-via-pip) for detailed instructions.

### Installing the library and dependencies

First, clone the repository locally

```bash
git clone https://github.com/semantic-health/allennlp-multi-label.git
```

Then, install

```bash
cd allennlp-multi-label
pip install --editable .
```

You should also install [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-zone) support by following the instructions for your system [here](https://pytorch.org/get-started/locally/).

## Usage

### Preparing a dataset

Datasets should be [JSON Lines](http://jsonlines.org/) files, where each line is valid JSON containing the fields `"text"` and `"labels"`, e.g.

```json
{"text": "NO GRAIN SHIPMENTS TO THE USSR -- USDA There were no shipments of U.S. grain or soybeans to the Soviet Union in the week ended March 19, according to the U.S. Agriculture Department's latest Export Sales report. The USSR has purchased 2.40 mln tonnes of U.S. corn for delivery in the fourth year of the U.S.-USSR grain agreement. Total shipments in the third year of the U.S.-USSR grains agreement, which ended September 30, amounted to 152,600 tonnes of wheat, 6,808,100 tonnes of corn and 1,518,700 tonnes of soybeans.", "labels": ["soybean", "oilseed", "wheat", "corn", "grain"]}
```

You can specify the train set path in the [configs](training_config) under `"train_data_path"`, `"validation_data_path"` and `"test_data_path"`.

### Training

For convenience, we have provided an example config file that will fine-tune a pretrained transformer-based language model (like BERT) for multi-label document classification. To train the model, use the [`allennlp train`](https://docs.allennlp.org/master/api/commands/train/) command with our [`multi_label_classifier.jsonnet`](training_config/multi_label_classifier.jsonnet) config

```bash
# This can be (almost) any model from https://huggingface.co/
TRANSFORMER_MODEL="distilroberta-base"
# Should not be longer than the max length supported by TRANSFORMER_MODEL.
MAX_LENGTH=512

allennlp train "configs/multi_label_classifier.jsonnet" \
    --serialization-dir "output" \
    --overrides "{'train_data_path': 'path/to/your/dataset/train.txt'}" \
    --include-package "allennlp_multi_label"
```

The `--overrides` flag allows you to override any field in the config with a JSON-formatted string, but you can equivalently update the config itself if you prefer. During training, models, vocabulary, configuration, and log files will be saved to the directory provided by `--serialization-dir`. This can be changed to any directory you like. 

#### Multi-GPU training

To train on more than one GPU, provide a list of CUDA devices in your call to `allennlp train`. For example, to train with four CUDA devices with IDs `0, 1, 2, 3`

```bash
--overrides "{'distributed.cuda_devices': [0, 1, 2, 3]}"
```

#### Training with mixed-precision

If you want to train with [mixed-precision](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) (strongly recommended if your GPU supports it), you will need to [install Apex with CUDA and C++ extensions](https://github.com/NVIDIA/apex#quick-start). Once installed, you need only to set `"opt_level"` to `"O1"` in your training [config](configs), or, equivalently, pass the following flag to `allennlp train`

```bash
--overrides "{'trainer.opt_level': 'O1'}"
```

### Inference

Coming soon.