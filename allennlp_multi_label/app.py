from typing import List

import pandas as pd
import streamlit as st
import typer
import random

import seaborn as sns
from allennlp.common import util as common_util
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


@st.cache(allow_output_mutation=True)
def _load_predictor(
    archive_file: str,
    weights_file: str = None,
    cuda_device: int = -1,
    opt_level: str = "O0",
    include_package: List[str] = None,
) -> Predictor:
    # This allows us to import custom dataset readers and models that may exist in the AllenNLP
    # archive. See: https://tinyurl.com/whkmoqh
    include_package = include_package or []
    for package_name in include_package:
        common_util.import_module_and_submodules(package_name)

    # Load the pretrained model for multi-class classification.
    archive = load_archive(
        archive_file, cuda_device=cuda_device, opt_level=opt_level, weights_file=weights_file,
    )
    # Use st.cache so that it doesn't reload when you change the inputs.
    predictor = Predictor.from_archive(archive, "multi_label")
    return predictor


def main(
    archive_file: str,
    weights_file: str = None,
    cuda_device: int = -1,
    opt_level: str = "O0",
    include_package: List[str] = None,
) -> None:
    st.title("Patient Cohort Classification")
    (
        "This is a simple dashboard which allows you to make predictions with a"
        " __deep patient cohort classifier__. Try copy/pasting the text from an electronic health"
        " record (EHR) below (or running the example!)"
    )

    predictor = _load_predictor(archive_file, weights_file, cuda_device, opt_level, include_package)

    # Create a text area to input the text.
    text = st.text_area(
        "Copy/paste your EHR here", "The Matrix is a 1999 movie starring Keanu Reeves."
    )

    # Use the predictor to label the input.
    predictions = predictor.predict(text)

    # From the result, we want the predicted labels and their probabilities
    labels = predictions["labels"]
    # TODO (John): These are random for visualization purposes, but the model should
    # figure these out on its own.
    probabilities = [random.uniform(0.0, 1.0) * 100 for _ in labels]

    # Render the results
    "### Prediction:"
    if labels:
        df = pd.DataFrame({"Prediction": labels, "Probability (%)": probabilities})
        cm = sns.light_palette("green", as_cmap=True)
        st.dataframe(df.style.background_gradient(cmap=cm))
    else:
        "No labels predicted for the given input text."


if __name__ == "__main__":
    typer.run(main)
