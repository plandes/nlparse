# Zensols Natural Language Processing

[![Travis CI Build Status][travis-badge]][travis-link]
[![PyPI][pypi-badge]][pypi-link]
[![Python 3.7][python37-badge]][python37-link]

This framework wraps the [SpaCy] framework and creates features.  The
motivation is to generate features from the parsed text in an object oriented
fashion that is fast and easy to pickle.  Other features include:
* [Token normalization](doc/parse.md) as a stream of strings by lemmatization,
  stop word and/or punctuation filters, up/down casing, porter stemming and
  [others](doc/normalizers.md).
* [Detached features](doc/parse.md) that are safe and easy to pickle to disk.
* Configuration drive parsing and token normalization using [configuration
  factories].
* Pretty print functionality for easy natural language feature selection.


## Documentation

* [Framework documentation](https://plandes.github.io/nlparse/)
* [Natural Language Parsing](doc/parse.md)
* [List Token Normalizers and Mappers](doc/normalizers.md)


## Obtaining

The easist way to install the command line program is via the `pip` installer:
```bash
pip3 install zensols.nlp
```

Binaries are also available on [pypi].


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2020 Paul Landes


<!-- links -->
[travis-link]: https://travis-ci.org/plandes/nlparse
[travis-badge]: https://travis-ci.org/plandes/nlparse.svg?branch=master
[pypi]: https://pypi.org/project/zensols.nlp/
[pypi-link]: https://pypi.python.org/pypi/zensols.nlp
[pypi-badge]: https://img.shields.io/pypi/v/zensols.nlp.svg
[python37-badge]: https://img.shields.io/badge/python-3.7-blue.svg
[python37-link]: https://www.python.org/downloads/release/python-370

[SpaCy]: https://spacy.io

[configuration factories]: https://plandes.github.io/util/doc/config.html#configuration-factory
