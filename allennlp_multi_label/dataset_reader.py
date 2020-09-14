import json
import logging
from typing import Dict, Iterable, List, Optional, Union

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, MultiLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("multi_label")
class MultiLabelTextClassificationJsonReader(TextClassificationJsonReader):
    """
    Reads tokens and their labels from a multi-label text classification dataset.
    Expects a "text" field and a "labels" field in JSON format.

    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        labels : `MultiLabelField`

    Registered as a `DatasetReader` with name "mulit_label_text_classification_json".

    [0]: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

    # Parameters

    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    segment_sentences : `bool`, optional (default = `False`)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences, like [the Hierarchical
        Attention Network][0].
    max_sequence_length : `int`, optional (default = `None`)
        If specified, will truncate tokens to specified maximum length.
    skip_label_indexing : `bool`, optional (default = `False`)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    num_labels : `int`, optional (default=`None`)
        If `skip_indexing=True`, the total number of possible labels should be provided, which is
        required to decide the size of the output tensor. `num_labels` should equal largest label
        id + 1. If `skip_indexing=False`, `num_labels` is not required.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        num_labels: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            segment_sentences=segment_sentences,
            max_sequence_length=max_sequence_length,
            skip_label_indexing=skip_label_indexing,
            **kwargs,
        )

        self._num_labels = num_labels

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                text = items["text"]
                labels = items.get("labels")
                if labels is not None:
                    if self._skip_label_indexing:
                        try:
                            labels = [int(label) for label in labels]
                        except ValueError:
                            raise ValueError(
                                "Labels must be integers if skip_label_indexing is True."
                            )
                    else:
                        labels = [str(label) for label in labels]
                instance = self.text_to_instance(text=text, labels=labels)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(
        self, text: str, labels: List[Union[str, int]] = None
    ) -> Instance:  # type: ignore
        """
        # Parameters

        text : `str`, required.
            The text to classify
        labels : `List[Union[str, int]]`, optional, (default = `None`).
            The labels for this text.

        # Returns

        An `Instance` containing the following fields:
            - tokens (`TextField`) :
              The tokens in the sentence or phrase.
            - label (`MultiLabelField`) :
              The labels of the sentence or phrase.
        """

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens, self._token_indexers))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        if labels is not None:
            fields["labels"] = MultiLabelField(
                labels, skip_indexing=self._skip_label_indexing, num_labels=self._num_labels
            )
        return Instance(fields)
