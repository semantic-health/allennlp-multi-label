// This should be a registered name in the Transformers library (see https://huggingface.co/models) 
// OR a path on disk to a serialized transformer model.
local transformer_model = std.extVar("TRANSFORMER_MODEL");
// Inputs longer than this will be truncated.
local max_length = 512;

{
    "dataset_reader": {
        "type": "multi_label",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            // Account for special tokens (e.g. CLS and SEP), otherwise a cryptic error is thrown.
            "max_length": max_length - 2,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
        // If not null, a cache of already-processed data will be stored in this directory.
        // If a cache file exists at this directory, it will be loaded instead of re-processing the data.
        "cache_directory": null
    }, 
    "train_data_path": null,
    "validation_data_path": null,
    "model": {
        "type": "multi_label",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": transformer_model,
                },
            },
        },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["tokens"],
            "batch_size" : 16,
        },
        "num_workers": 1
    },
    "trainer": {
        // If Apex is installed, chose one of its opt_levels here to use mixed-precision training.
        "opt_level": null,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-5,
            "weight_decay": 0.0,
            "parameter_groups": [
                # Apply weight decay to pre-trained parameters, exlcuding LayerNorm parameters and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3/tests
                [["(?=.*transformer_model)(?=.*\\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": 0.1}],
            ],
        },
        "num_epochs": 1,
        "checkpointer": {
            "num_serialized_models_to_keep": 1,
        },
        "grad_norm": 1.0,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
    },
}