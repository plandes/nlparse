# Zensols Natural Language Parsing

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.7][python37-badge]][python37-link]
[![Python 3.8][python38-badge]][python38-link]
[![Python 3.9][python39-badge]][python39-link]
[![Build Status][build-badge]][build-link]

This framework wraps the [spaCy] framework and creates light weight features in
a class [hierarchy] that reflects the structure of natural language.  The
motivation is to generate features from the parsed text in an object oriented
fashion that is fast and easy to pickle.

* See the [full documentation].
* Paper on [arXiv](http://arxiv.org/abs/2109.03383).

Other features include:
* [Parse and normalize] a stream of tokens as stop words, punctuation
  filters, up/down casing, porter stemming and [others].
* [Detached features] that are safe and easy to pickle to disk.
* Configuration drive parsing and token normalization using [configuration
  factories].
* Pretty print functionality for easy natural language feature selection.


## Documentation

* [Framework documentation]
* [Natural Language Parsing]
* [List Token Normalizers and Mappers]


## Obtaining / Installing

1. The easist way to install the command line program is via the `pip`
   installer: `pip3 install zensols.nlp`
2. Install at least one spaCy model: `python -m spacy download en_core_web_sm`

Binaries are also available on [pypi].


## Attribution

This project, or example code, uses:
* [spaCy] for natural language parsing
* [msgpack] and [smart-open] for Python disk serialization
* [nltk] for the [porter stemmer] functionality


## Citation

If you use this project in your research please use the following BibTeX entry:
```
@article{Landes_DiEugenio_Caragea_2021,
  title={DeepZensols: Deep Natural Language Processing Framework},
  url={http://arxiv.org/abs/2109.03383},
  note={arXiv: 2109.03383},
  journal={arXiv:2109.03383 [cs]},
  author={Landes, Paul and Di Eugenio, Barbara and Caragea, Cornelia},
  year={2021},
  month={Sep}
}
```


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## License

[MIT License](LICENSE.md)

Copyright (c) 2020 - 2021 Paul Landes


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

[hierarchy]: https://plandes.github.io/nlparse/api/zensols.nlp.html#zensols.nlp.container.FeatureDocument
[Parse and normalize]: https://plandes.github.io/nlparse/doc/parse.html
[others]: https://plandes.github.io/nlparse/doc/normalizers.html
[Detached features]: https://plandes.github.io/nlparse/doc/parse.html#detached-features
[Framework documentation]: https://plandes.github.io/nlparse/
[Natural Language Parsing]: https://plandes.github.io/nlparse/doc/parse.html
[List Token Normalizers and Mappers]: https://plandes.github.io/nlparse/doc/normalizers.html


[spaCy]: https://spacy.io
[nltk]: https://www.nltk.org
[smart-open]: https://pypi.org/project/smart-open/
[msgpack]: https://msgpack.org
[porter stemmer]: https://tartarus.org/martin/PorterStemmer/

[configuration factories]: https://plandes.github.io/util/doc/config.html#configuration-factory
