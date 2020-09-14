from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register("multi_label")
class MultiLabelClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in some text and returns a multi-label class prediction
    for it. In particular, it can be used with the `MultiLabelClassifier(Model)` model.

    Registered as a `Predictor` with name "text_classifier".
    """

    def predict(self, text: str) -> JsonDict:
        return self.predict_json({"text": text})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"text": "..."}`. Runs the underlying model, and adds the
        `"labels"` to the output. Based on:
        https://github.com/allenai/allennlp/blob/master/allennlp/predictors/text_classifier.py
        """
        text = json_dict["text"]
        return self._dataset_reader.text_to_instance(text=text)
