# Natural Language Parsing

Before reading this, please read about [feature documents].  If you want to
jump right in, its recommended to at least pursue the simple CLI example.

This framework wraps the [spaCy] framework and creates features.  The
motivation is to generate features from the parsed text in an object oriented
fashion that is fast and easy to pickle as many spaCy objects are C data
structures.

A secondary use of this package provides a simple, yet robust way to generate a
string stream of tokens using a [TokenNormalizer].  This allows for
configuration driven way of generating tokens used for downstream feature
vectorization such as word vectors, text classification, information
retrieval/search, latent semantic indexing or any task that uses a single
string token.

Token streams can be transformed using [TokenMapper] instances.  These take the
output of a tokenizer, and then modify them in various ways.  Finally, the
[MapTokenNormalizer] is a normalizer that uses a list of [TokenMapper]s to
first create the token stream and then transform them.  See the [norm package]
for token normalizers and mappers.


## Resource Library

The [NLP resource library] contains configuration for a language parser that
works for most use cases.  However, like any [resource library], importing and
overriding is straight forward.

A [TokenNormalizer] is defined in [obj.conf]:
```ini
[filter_token_mapper]
class_name = zensols.nlp.FilterTokenMapper
#remove_stop = True
#remove_punctuation = True
#remove_space = True

[map_filter_token_normalizer]
class_name = zensols.nlp.MapTokenNormalizer
mapper_class_list = list: filter_token_mapper
```

A [SpacyFeatureDocumentParser], which will be used to parse text in to
[spaCy] documents:
```ini
[doc_parser]
class_name = zensols.nlp.sparser.SpacyFeatureDocumentParser
lang = en
model_name = ${lang}_core_web_sm
token_normalizer = instance: map_filter_token_normalizer
```
which defines a language resource for English that uses our previously defined
token normalizer.  Note that the API provides for these two tasks (parsing and
token normalization) separately.


## Example

The [application example] consists of a full CLI application that configures
and uses a document parser.  In the example the `app.conf` imports the [NLP
resource libraries].  By default, the using the `parse` action shows all
features for all tokens.  However, when adding `--config terse.conf` stop
words, punctuation and white space tokens are removed.  Similarly, adding
`--config lemma.conf` configures the parser to use `lemma_token_mapper`, which
uses the lemmas as normalized tokens.

The `detailed` action/method in the `app.py` Python source code file
illustrates basic usage of the parser.  To get a [feature document], which has
all the configured parsed artifacts typically needed to use in machine learning
models, use the [FeatureDocumentParser], which is the base class of
[SpacyFeatureDocumentParser]:
```python
doc: FeatureDocument = self.doc_parser(sentence)
```

If you only want a [spaCy] `Doc` instance use the [FeatureDocumentParser]'s
`parse_spacy_doc` method:
```python
doc: Doc = self.doc_parser.parse_spacy_doc(sentence)
```

See the inline documentation/comments in the `app.py` Python file that explains
how to use the API and the `makefile` to run each example.


<!-- links -->
[spaCy]: https://spacy.io

[NLP resource library]: https://github.com/plandes/nlparse/tree/master/resources
[NLP resource libraries]: https://github.com/plandes/nlparse/tree/master/resources
[resource library]: https://plandes.github.io/util/doc/config.html#resource-libraries

[norm package]: ../api/zensols.nlp.html#module-zensols.nlp.norm
[FeatureDocumentParser]: ../api/zensols.nlp.html#zensols.nlp.parser.FeatureDocumentParser
[SpacyFeatureDocumentParser]: ../api/zensols.nlp.html#zensols.nlp.parser.SpacyFeatureDocumentParser
[TokenNormalizer]: ../api/zensols.nlp.html#zensols.nlp.norm.TokenNormalizer
[TokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.TokenMapper
[MapTokenNormalizer]: ../api/zensols.nlp.html#zensols.nlp.norm.MapTokenNormalizer
[feature documents]: feature-doc.md
[feature document]: feature-doc.md
[application example]: https://github.com/plandes/nlparse/tree/master/example/config
