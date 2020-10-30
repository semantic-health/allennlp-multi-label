# PyPI Release Process

1. Update the version number in `allennlp-multi-label/__init__.py` according to [SemVer](https://semver.org/). This should match the version number under `pyproject.toml` and `tests/test_allennlp_multi_label.py`. Note you can use the poetry command [version](https://python-poetry.org/docs/cli/#version) to update the `pyproject.toml` file. 
2. Generate a `setup.py` file by calling `dephell deps convert`.
3. Commit these changes to master.
4. Create a new release on GitHub. It should be named `v{VERSION}` where `VERSION` matches the version number from step 1.
5. Make sure your local copy of the repository is up-to-date: `git fetch ; git pull`.
6. Checkout the tag: `git checkout tags/v{VERSION}`.
7. Then run `poetry build ; poetry publish`. You will be prompted to sign into [PyPI](https://pypi.org/).
