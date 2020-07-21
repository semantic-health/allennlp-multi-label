local COMMON = import 'common.jsonnet';
local transformer_model = "distilroberta-base";

{
    "dataset_reader": COMMON['dataset_reader'],
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": COMMON['train_data_path'],
    "validation_data_path": COMMON['validation_data_path'],
    "model": COMMON['model'],
    "data_loader": COMMON['data_loader'],
    "trainer": COMMON['trainer']
}