![build](https://github.com/JohnGiorgi/allennlp-multi-label-classification/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/semantic-health/allennlp-multi-label/branch/master/graph/badge.svg)](https://codecov.io/gh/semantic-health/allennlp-multi-label)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/allennlp-multi-label-classification?color=blue)

# AllenNLP Multi-label Classification Plugin

A multi-label classification plugin for [AllenNLP](https://allennlp.org/).

## Installation

This repository requires Python 3.6.1 or later. The preferred way to install is via pip:

```
pip install allennlp-multi-label
```

If you need pointers on setting up an appropriate Python environment, please see the [AllenNLP install instructions](https://github.com/allenai/allennlp#installing-via-pip).

### Installing from source

You can also install from source. This project is managed with [Poetry](https://python-poetry.org/), so that will need to be installed first.

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/semantic-health/allennlp-multi-label
cd allennlp-multi-label

# Install the package with poetry
poetry install
```

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

allennlp train "training_config/multi_label_classifier.jsonnet" \
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

### Inference

Coming soon.