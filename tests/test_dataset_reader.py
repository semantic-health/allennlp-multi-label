from pathlib import Path
from typing import List

import pytest

from allennlp.common.util import ensure_list, get_spacy_model
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp_multi_label.dataset_reader import MultiLabelTextClassificationJsonReader


class TestMultiLabelTextClassificationJsonReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_set_skip_indexing_true(self, lazy):
        reader = MultiLabelTextClassificationJsonReader(
            lazy=lazy, skip_label_indexing=True, num_labels=3
        )
        integer_label_path = Path("tests/fixtures") / "data" / "integer_labels.jsonl"
        instances = reader.read(integer_label_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["This", "text", "has", "labels", "0", "2"], "labels": [0, 2]}
        instance2 = {"tokens": ["This", "text", "has", "labels", "0", "1"], "labels": [0, 1]}

        assert len(instances) == 2
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["labels"].labels == instance1["labels"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["labels"].labels == instance2["labels"]

        with pytest.raises(ValueError) as exec_info:
            string_label_path = Path("tests/fixtures") / "data" / "reuters-21578" / "train.jsonl"
            ensure_list(reader.read(string_label_path))
        assert str(exec_info.value) == "Labels must be integers if skip_label_indexing is True."

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_reuters_corpus(self, lazy):
        reader = MultiLabelTextClassificationJsonReader(lazy=lazy)
        reuters_path = Path("tests/fixtures") / "data" / "reuters-21578" / "train.jsonl"
        instances = reader.read(reuters_path)
        instances = ensure_list(instances)

        instance1 = {
            "tokens": [
                "U.K.",
                "GROWING",
                "IMPATIENT",
                "WITH",
                "JAPAN",
                "-",
                "THATCHER",
                "Prime",
                "Minister",
                "Margaret",
                "Thatcher",
                "said",
                "the",
                "U.K.",
                "Was",
                "growing",
                "more",
                "impatient",
                "with",
                "Japanese",
                "trade",
                "barriers",
                "and",
                "warned",
                "that",
                "it",
                "would",
                "soon",
                "have",
                "new",
                "powers",
                "against",
                "countries",
                "not",
                "offering",
                "reciprocal",
                "access",
                "to",
                "their",
                "markets",
                ".",
            ],
            "labels": ["acq", "trade"],
        }
        instance2 = {
            "tokens": [
                "CANADA",
                "OIL",
                "EXPORTS",
                "RISE",
                "20",
                "PCT",
                "IN",
                "1986",
                "Canadian",
                "oil",
                "exports",
                "rose",
                "20",
                "pct",
                "in",
                "1986",
                "over",
                "the",
                "previous",
                "year",
                "to",
                "33.96",
                "mln",
                "cubic",
                "meters",
                ",",
                "while",
                "oil",
                "imports",
                "soared",
                "25.2",
                "pct",
                "to",
                "20.58",
                "mln",
                "cubic",
                "meters",
                ",",
                "Statistics",
                "Canada",
                "said",
                ".",
                "Production",
                ",",
                "meanwhile",
                ",",
                "was",
                "unchanged",
                "from",
                "the",
                "previous",
                "year",
                "at",
                "91.09",
                "mln",
                "cubic",
                "feet",
                ".",
            ],
            "labels": ["nat-gas", "crude"],
        }
        instance3 = {
            "tokens": [
                "COFFEE",
                ",",
                "SUGAR",
                "AND",
                "COCOA",
                "EXCHANGE",
                "NAMES",
                "CHAIRMAN",
                "The",
                "New",
                "York",
                "Coffee",
                ",",
                "Sugar",
                "and",
                "Cocoa",
                "Exchange",
                "(",
                "CSCE",
                ")",
                "elected",
                "former",
                "first",
                "vice",
                "chairman",
                "Gerald",
                "Clancy",
                "to",
                "a",
                "two",
                "-",
                "year",
                "term",
                "as",
                "chairman",
                "of",
                "the",
                "board",
                "of",
                "managers",
                ",",
                "replacing",
                "previous",
                "chairman",
                "Howard",
                "Katz",
                ".",
                "Katz",
                ",",
                "chairman",
                "since",
                "1985",
                ",",
                "will",
                "remain",
                "a",
                "board",
                "member",
                ".",
            ],
            "labels": ["sugar", "cocoa", "coffee"],
        }

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["labels"].labels == instance1["labels"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["labels"].labels == instance2["labels"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["labels"].labels == instance3["labels"]

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_reuters_corpus_and_truncates_properly(self, lazy):
        reader = MultiLabelTextClassificationJsonReader(lazy=lazy, max_sequence_length=5)
        reuters_path = Path("tests/fixtures") / "data" / "reuters-21578" / "train.jsonl"
        instances = reader.read(reuters_path)
        instances = ensure_list(instances)

        instance1 = {
            "tokens": ["U.K.", "GROWING", "IMPATIENT", "WITH", "JAPAN"],
            "labels": ["acq", "trade"],
        }
        instance2 = {
            "tokens": ["CANADA", "OIL", "EXPORTS", "RISE", "20"],
            "labels": ["nat-gas", "crude"],
        }
        instance3 = {
            "tokens": ["COFFEE", ",", "SUGAR", "AND", "COCOA"],
            "labels": ["sugar", "cocoa", "coffee"],
        }

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["labels"].labels == instance1["labels"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["labels"].labels == instance2["labels"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["labels"].labels == instance3["labels"]

    @pytest.mark.parametrize("max_sequence_length", (None, 5))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_reuters_corpus_and_segments_sentences_properly(
        self, lazy, max_sequence_length
    ):
        reader = MultiLabelTextClassificationJsonReader(
            lazy=lazy, segment_sentences=True, max_sequence_length=max_sequence_length
        )
        reuters_path = Path("tests/fixtures") / "data" / "reuters-21578" / "train.jsonl"
        instances = reader.read(reuters_path)
        instances = ensure_list(instances)

        splitter = SpacySentenceSplitter()
        spacy_tokenizer = get_spacy_model("en_core_web_sm", False, False, False)

        text1 = (
            "U.K. GROWING IMPATIENT WITH JAPAN - THATCHER Prime Minister Margaret Thatcher said the"
            " U.K. Was growing more impatient with Japanese trade barriers and warned that it would"
            " soon have new powers against countries not offering reciprocal access to their"
            " markets."
        )
        instance1 = {"text": text1, "labels": ["acq", "trade"]}
        text2 = (
            "CANADA OIL EXPORTS RISE 20 PCT IN 1986 Canadian oil exports rose 20 pct in 1986 over"
            " the previous year to 33.96 mln cubic meters, while oil imports soared 25.2 pct to"
            " 20.58 mln cubic meters, Statistics Canada said. Production, meanwhile, was unchanged"
            " from the previous year at 91.09 mln cubic feet."
        )
        instance2 = {"text": text2, "labels": ["nat-gas", "crude"]}
        text3 = (
            "COFFEE, SUGAR AND COCOA EXCHANGE NAMES CHAIRMAN The New York Coffee, Sugar and Cocoa"
            " Exchange (CSCE) elected former first vice chairman Gerald Clancy to a two-year term"
            " as chairman of the board of managers, replacing previous chairman Howard Katz. Katz,"
            " chairman since 1985, will remain a board member."
        )
        instance3 = {"text": text3, "labels": ["sugar", "cocoa", "coffee"]}

        for instance in [instance1, instance2, instance3]:
            sentences = splitter.split_sentences(instance["text"])
            tokenized_sentences: List[List[str]] = []
            for sentence in sentences:
                tokens = [token.text for token in spacy_tokenizer(sentence)]
                if max_sequence_length:
                    tokens = tokens[:max_sequence_length]
                tokenized_sentences.append(tokens)
            instance["tokens"] = tokenized_sentences

        assert len(instances) == 3
        fields = instances[0].fields
        text = [[token.text for token in sentence.tokens] for sentence in fields["tokens"]]
        assert text == instance1["tokens"]
        assert fields["labels"].labels == instance1["labels"]
        fields = instances[1].fields
        text = [[token.text for token in sentence.tokens] for sentence in fields["tokens"]]
        assert text == instance2["tokens"]
        assert fields["labels"].labels == instance2["labels"]
        fields = instances[2].fields
        text = [[token.text for token in sentence.tokens] for sentence in fields["tokens"]]
        assert text == instance3["tokens"]
        assert fields["labels"].labels == instance3["labels"]
