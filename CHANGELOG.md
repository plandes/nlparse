# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


## [1.3.0] - 2022-08-06
### Added
- Token indexing mappings accounting for (named entity) multi-word tokens.
- IOB (`iob_`, `iob`) features.
- Re-loadable components and component initializers.

### Changed
- Upgraded to spaCy 3.2
- Add spaCy tokens to spaCy feature tokens.
- Bug fixes in combining and overlapping sentences.
- Switched to shallow copy of document in overlapping sentence doc methods.


## [1.2.0] - 2022-06-16
### Removed
- Remove resource library `regular_expression_escape:dollar` configuration.
  Use [zensols.util] `conf_esc:dollar` as a replacement.


## [1.1.2] - 2022-06-14
### Changed
- Dependency bump.


## [1.1.1] - 2022-05-15
### Changed
- Dependency bump.


## [1.1.0] - 2022-05-04
### Changed
- Fix resource leaks and other bugs.
- Persist original text along with `FeatureDocument` rather than reconstruct it
  from sentence and/or token text.

### Added
- An lexical overlapping utility module (`overlap`).
- A token normalizer that merges tokens in to spans (`JoinTokenMapper`).
- Regular expression matching for entity and merge components (similar to
  `JoinTokenMapper`).
- Add back `TokenAnnotatedFeatureSentence` for down stream packages.
- Add token decorator to spacy parser to allow for add/modify
  features on creation separate from parser class hierarchy.


## [1.0.1] - 2022-01-25
### Added
- Sentences and tokens accessible by index.

### Changed
- More robust regular expression for token splitting.
- Mapping combiner is persistable with spaCy tokens and handles split named
  entities.


## [1.0.0] - 2021-10-22
First major development release.

### Added
- A `FeatureDocumentCombiner` that merges features from different document
  parsers.
- Top level library `NLPError`.
- A pipeline component and resource configuration library entry to remove
  sentence boundaries in a spaCy document.

### Changed
- Split out optional resource library content in to `mappers.conf`.
- The spaCy model has attribute `langres` set on `LanguageResource` to enable
  creation of factory instances from registered pipe components.
- Fix issue with component creation with no pipeline arguments.

### Removed
- The `DocStash` instance as it was too simple for any practical application.


## [0.1.3] - 2021-09-21
### Changed
- Dependency.

### Removed
- `zensols.nlp.lang.DocStash`


## [0.1.2] - 2021-09-21
### Changed
- Make `FeatureDocumentParser` callable.
- Fix memory leak in `LanguageResource`.

### Added
- Configuration Resource library.
- Configuration for keyword arguments to the `add_pipe_comp` and example.


## [0.1.1] - 2021-09-07
### Changed
- Fixed bug with creating a `dict` from a `FeatureToken`.
- Fixed/improved how `Feature{Token,Sentence,Document}` are `dict`ified with
  (`asdict`) and how they are written as text with `write`.

### Added
- Creates a Pandas dataframe from token feature attributes.
- Add back `FeatureToken` feature ID -> type for write dumping
- Add lexical location `SpacyTokenFeatures.loc` location in the document as an
  (starting, ending) range.


## [0.1.0] - 2021-08-16
This release simplifies the token attributes level classes in the `features`
module by:
  * Using feature IDs instead of trying to make sense of the class
    property/attribute member data.
  * Using the `FeatureDocumentParser` and `FeatureToken` to copy spaCy
    resources to simple picklable Python classes.

Not only does this greatly reduce complexity in class hierarchy and data
copy/move functionality, but speeds things up.

### Changes
- Attributes set on detached token features are no longer robust.  Before, if a
  token feature ID was specified, but didn't exist on the source token feature
  set, it would copy over a `None`.  This now raises an `AttributeError` instead.
- For `TokenAttributes`, creation of `dicts` (either by `asdict` or
  `get_features`) is now consistent with the set attributes and properties of
  the class.  Only those specified passed to methods, which default to
  `FIELD_IDS` of the class (which can be overridden at a class level).

### Removed
- The dictionary creation of attribute/property individual features methods
  `TokenAttributes.{string}features`.  These methods are obviated by the
  `get_features`, which returns all features in `FIELD_IDS`.
- `FeatureDocumentParser.additional_token_feature_ids` to simplify token
  feature IDs passed to feature tokens.
- The `TokenAttributes` class, as it was just a metadata member holder.

### Added
- A SpaCy implementation of the `TokenFeatures` class, that somewhat resembles
  the old `TokenFeatures` of the old class hierarchy.


## [0.0.15] - 2021-08-07
### Changes
- Upgrade from spaCy 2.x to 3.x.

### Added
- POS feature inclusion by default to support `is_pronoun`, which is needed
  after spaCy 3 changed how lemmatization works.
- Move feature containers and parser from `zensols.deepnlp`, including test
  cases.
- A sentence index feature (`i_sent`).
- An *index of sentence* feature (`sent_i`).
- Advanced spacy configuration by adding component classes.  This gives more
  control over configuring the spaCy pipeline.
- Add feature containers (`FeatureDocument`) and parser
  (`FeatureDocumentParser`), which were moved over from [zensols.deepnlp].


## [0.0.14] - 2021-04-29
## Changes
- Upgrade to [zensols.util]==1.4.1.
- Upgrade documentation API generation.
- Nail dependencies to spacy 2.3.5 until pip deps are fixed.
- Added sentence index features to reconstruct sentences from documents.


## [0.0.13] - 2021-01-14
### Changes
- Fix component adds for spacy > 2.0.
- Add langres model to API documentation.


## [0.0.12] - 2020-12-29
### Changed
- Upgraded [zenbuild].
- Switched from Travis to GitHub workflows.
- Tested with Python 3.9.1.


## [0.0.11] - 2020-12-09
### Changed
- Add basic token features for non-spacy parse use cases.
- Rename feature type to feature id.
- `TokeFeatures` is now a dictable with to_dict -> asdict.


## [0.0.10] - 2020-12-09
### Added
- Sphinx documentation, which includes API docs.

### Changed
- Settable detached `TokenAttributes` instances.
- Make `dataclasses`, and therefore, needs >= Python 3.7.


## [0.0.9] - 2020-05-10
### Changed
- Home/master move lemmatizing out of default token normalizer.
- Update super method calls to modern (at least) Python 3.7.
- Fix annoying can't find smart_open.gcs bogus warning.
- Remove language resource factory.
- Upgrade to zensols.util 1.2.0 and get rid of custom factories.

### Added
- Feature to parse whole special tokens.
- Added porter stemmer from [nltk].

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
[Unreleased]: https://github.com/plandes/nlparse/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/plandes/nlparse/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/plandes/nlparse/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/plandes/nlparse/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/plandes/nlparse/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/plandes/nlparse/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/plandes/nlparse/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/plandes/nlparse/compare/v0.1.3...v1.0.0
[0.1.3]: https://github.com/plandes/nlparse/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/plandes/nlparse/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/plandes/nlparse/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/plandes/nlparse/compare/v0.0.15...v0.1.0
[0.0.15]: https://github.com/plandes/nlparse/compare/v0.0.14...v0.0.15
[0.0.14]: https://github.com/plandes/nlparse/compare/v0.0.13...v0.0.14
[0.0.13]: https://github.com/plandes/nlparse/compare/v0.0.13...v0.0.13
[0.0.13]: https://github.com/plandes/nlparse/compare/v0.0.12...v0.0.13
[0.0.12]: https://github.com/plandes/nlparse/compare/v0.0.11...v0.0.12
[0.0.11]: https://github.com/plandes/nlparse/compare/v0.0.10...v0.0.11
[0.0.10]: https://github.com/plandes/nlparse/compare/v0.0.9...v0.0.10
[0.0.9]: https://github.com/plandes/nlparse/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/plandes/nlparse/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/plandes/nlparse/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/plandes/nlparse/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/plandes/nlparse/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/plandes/nlparse/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/plandes/nlparse/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/plandes/nlparse/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/plandes/nlparse/compare/v0.0.0...v0.0.1

[nltk]: https://www.nltk.org
[zensols.deepnlp]: https://github.com/plandes/deepnlp
[zenbuild]: https://github.com/plandes/zenbuild
[zensols.util]: https://github.com/plandes/util
