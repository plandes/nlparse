# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]
### Added
- Sphinx documentation, which includes API docs.


## [0.0.9] - 2020-05-10
### Changed
- Home/master move lemmatizing out of default token normalizer.
- Update super method calls to modern (at least) Python 3.7.
- Fix annoying can't find smart_open.gcs bogus warning.
- Remove language resource factory.
- Upgrade to zensols.util 1.2.0 and get rid of custom factories.

### Added
- Feature to parse whole special tokens.

### Removed
- Moved word2vec embedding (`word2vec.py`) to [zensols.deepnlp] library.
- Moved feature normalization (`fnorm.py`) to [zensols.deepnlp] library.


## [0.0.8] - 2020-04-14
### Changed
- Upgrade to `spaCy` 2.2.4 and `textacy` 0.10.0


## [0.0.7] - 2020-01-24
### Added
- Added the Porter stemmer from the [NTLK].
### Changed
- Better class naming for token mapper.
- Features debugging bug fix.


## [0.0.6] - 2019-12-14
### Changed
- Fix Travis.


## [0.0.5] - 2019-12-14
Data classes are now used so Python 3.7 is now a requirement.

### Added
- Feature normalizers were added for neural networks.
- Implemented a better strategy for using language resources with token
  normalization.

## [0.0.4] - 2019-11-21
## Added
- Adding detachable and picklable token feature set.


## [0.0.3] - 2019-07-31
## Added
- ``DocStash`` that parses documents as a factory stash.


## [0.0.2] - 2019-07-25
### Added
- Feature to disable SpaCy pipeline components.
- Add configuration for removing punctuation and determiners.

### Changed
- Skip textacy for document creation since it wasn't used.  This is more
  efficient.


## [0.0.1] - 2019-07-06
### Added
- Initial version.


<!-- links -->
[Unreleased]: https://github.com/plandes/nlparse/compare/v0.0.9...HEAD
[0.0.9]: https://github.com/plandes/nlparse/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/plandes/nlparse/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/plandes/nlparse/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/plandes/nlparse/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/plandes/nlparse/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/plandes/nlparse/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/plandes/nlparse/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/plandes/nlparse/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/plandes/nlparse/compare/v0.0.0...v0.0.1

[NLTK]: https://www.nltk.org
[zensols.deepnlp]: https://github.com/plandes/deepnlp
