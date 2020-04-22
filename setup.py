import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="allennlp-multi-label-document-classification",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=(
        "A multi-label document classification plugin for AllenNLP"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnGiorgi/allennlp-multi-label-document-classification",
    packages=setuptools.find_packages(),
    keywords=[
        "natural language processing",
        "pytorch",
        "allennlp",
        "transformers",
        "document classification",
        "multi-label",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.4.0",
    ],
    extras_require={"dev": ["black", "flake8", "pytest"]},
)
