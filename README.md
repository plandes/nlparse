# Zensols Natural Language Parsing

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.7][python37-badge]][python37-link]
[![Python 3.8][python38-badge]][python38-link]
[![Python 3.9][python39-badge]][python39-link]
[![Build Status][build-badge]][build-link]

This framework wraps the [spaCy] framework and creates features.  The
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


## Attribution

This project, or example code, uses:
* [spaCy] for natural language parsing
* [msgpack] and [smart-open] for Python disk serialization
* [nltk] for the [porter stemmer] functionality


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2020 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.nlp/
[pypi-link]: https://pypi.python.org/pypi/zensols.nlp
[pypi-badge]: https://img.shields.io/pypi/v/zensols.nlp.svg
[python37-badge]: https://img.shields.io/badge/python-3.7-blue.svg
[python37-link]: https://www.python.org/downloads/release/python-370
[python38-badge]: https://img.shields.io/badge/python-3.8-blue.svg
[python38-link]: https://www.python.org/downloads/release/python-380
[python39-badge]: https://img.shields.io/badge/python-3.9-blue.svg
[python39-link]: https://www.python.org/downloads/release/python-390
[build-badge]: https://github.com/plandes/nlparse/workflows/CI/badge.svg
[build-link]: https://github.com/plandes/nlparse/actions

[spaCy]: https://spacy.io
[nltk]: https://www.nltk.org
[smart-open]: https://pypi.org/project/smart-open/
[msgpack]: https://msgpack.org
[porter stemmer]: https://tartarus.org/martin/PorterStemmer/

[configuration factories]: https://plandes.github.io/util/doc/config.html#configuration-factory
