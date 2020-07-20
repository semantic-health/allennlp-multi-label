from pathlib import Path

import pytest

from allennlp.common.testing import ModelTestCase


class TestMultiLabelClassifier(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        # We need to override the path set by AllenNLP
        self.FIXTURES_ROOT = Path("tests/fixtures")
        self.set_up_model(
            self.FIXTURES_ROOT / "experiment.jsonnet",
            self.FIXTURES_ROOT / "data" / "reuters-21578" / "train.jsonl",
        )

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "logits" in output_dict.keys()
        assert "probs" in output_dict.keys()
        assert "labels" in output_dict.keys()

    @pytest.mark.skip(reason="takes far to long to run")
    def test_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
