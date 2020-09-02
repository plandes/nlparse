# Natural Language Parsing

This framework wraps the [SpaCy] framework and creates features.  The
motivation is to generate features from the parsed text in an object oriented
fashion that is fast and easy to pickle as many SpaCy objects are C data
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


## Configuring Token Normalizers and Mappers

First lets configure a [TokenNormalizer] that removes stop words and
punctuation in a configuration file `parser.conf`:
```ini
[filter_token_mapper]
class_name = zensols.nlp.FilterTokenMapper
remove_stop = True
remove_punctuation = True

[map_filter_token_normalizer]
class_name = zensols.nlp.MapTokenNormalizer
mapper_class_list = eval: 'filter_token_mapper'.split()
```

Now we create a [LanguageResource], which will be used to parse text in to
[SpaCy] documents:
```ini
[langres]
class_name = zensols.nlp.LanguageResource
lang = en
model_name = ${lang}_core_web_sm
token_normalizer = instance: map_filter_token_normalizer
```
which defines a language resource for English that uses our previously defined
token normalizer.  Note that the API provides for these two tasks (parsing and
token normalization) separately as we'll see later.

Next, let's parse some text with our configuration.  If you're new to the
configuration API, please read how [configuration factories] are used.  First
create the factory:
```python
>>> from zensols.config import ExtendedInterpolationEnvConfig, ImportConfigFactory
>>> conf = ExtendedInterpolationEnvConfig('parser.conf')
>>> fac = ImportConfigFactory(conf)
>>> lr = fac('langres')
>>> lr
model_name: en_core_web_sm, lang: en
```

Now we can use the [LanguageResource] to parse:
```python
>>> sent = 'California is part of the United States.'
>>> doc = lr.parse(sent)
>>> doc
California is part of the United States.
>>> type(doc)
<class 'spacy.tokens.doc.Doc'>
>>> for t in doc:
...   print(t, t.tag_, t.is_stop)
... 
California NNP False
is VBZ True
part NN True
of IN True
the DT True
United NNP False
States NNP False
. . False
```

So far, we've only parsed the text in to a [SpaCy] document.  Now let's use the
token normalizer to get the string token stream:
```python
>>> ', '.join(lr.normalized_tokens(doc))
'California, the United States'
```

Let's create a new language resource with a token mapper that down cases and adds
a little syntactic sugar:
```ini
[lc_lambda_token_mapper]
class_name = zensols.nlp.LambdaTokenMapper
map_lambda = lambda x: (x[0], f'<{x[1].lower()}>')

[lc_token_mapper]
class_name = zensols.nlp.MapTokenNormalizer
mapper_class_list = eval: 'filter_token_mapper lc_lambda_token_mapper'.split()

[lc_langres]
class_name = zensols.nlp.LanguageResource
lang = en
model_name = ${lang}_core_web_sm
token_normalizer = instance: lc_token_mapper
```
In this configuration, we create a tokenizer that first filters (as we did
previously), and then lowers the token and surrounds the string with `<>`s.
Note the `lambda` takes a tuple, which is the [SpaCy] token and the string
version of it, which in this case is the [SpaCy] `Token.orth_` attribute.

Next we'll create the normalized tokens using this new configuration:
```python
>>> lr = fac('lc_langres')
>>> ', '.join(lr.normalized_tokens(lr.parse(sent)))
'<california>, <the united states>'
```


## Detached Features

Many times you'll want to parse many features and save them off to disk for
feature selection later.  Creating a feature set that is separate in memory
space from [SpaCy] data structures is called *detaching*, which can be done as
follows:
```python
>>> feats = lr.features(doc)
>>> for feat in lr.features(doc):
...   print(f'{feat} {type(feat)}')
...   feat.detach().write(depth=1)
... 
California (<california>) <class 'zensols.nlp.feature.TokenFeatures'>
    text: California
    norm: <california>
    i: 0
    tag: NNP
    is_wh: False
    entity: GPE
    dep: nsubj
    children: 0
    numerics: {'tag': 15794550382381185553, 'is_wh': False, 'is_stop': False, 'is_pronoun': False, 'index': 0, 'i': 0, 'is_space': False, 'is_punctuation': False, 'is_contraction': False, 'entity': 384, 'is_entity': True, 'shape': 16072095006890171862, 'is_superlative': False, 'children': 0, 'dep': 429}
the United States (<the united states>) <class 'zensols.nlp.feature.TokenFeatures'>
    text: the United States
    norm: <the united states>
    i: 4
    tag: DT
    is_wh: False
    entity: GPE
    dep: det
    children: 0
    numerics: {'tag': 15267657372422890137, 'is_wh': False, 'is_stop': False, 'is_pronoun': False, 'index': 22, 'i': 4, 'is_space': False, 'is_punctuation': False, 'is_contraction': False, 'entity': 384, 'is_entity': True, 'shape': 4088098365541558500, 'is_superlative': False, 'children': 0, 'dep': 415}
```
The `write` method prints each of the features to standard output.  The [detach
method] parsed from the API provides an instance of [TokenFeatures] that is
safe to pickle to the file system.  If you only want to keep specific features
(to increase speed and reduce feature footprint) This method accepts a `set` of
those features you want to keep.

The [TokenFeatures] instance has properties that provide easy ways of
vectorizing from either strings or numeric data:
```python
>>> next(feats).string_features
{'text': 'California', 'norm': '<california>', 'lemma': 'California', 'is_wh': False, 'is_stop': False, 'is_space': False, 'is_punctuation': False, 'is_contraction': False, 'i': 0, 'index': 0, 'tag': 'NNP', 'entity': 'GPE', 'is_entity': True, 'shape': 'Xxxxx', 'children': 0, 'superlative': False, 'dep': 'nsubj'}
>>> next(feats).features
{'tag': 15267657372422890137, 'is_wh': False, 'is_stop': False, 'is_pronoun': False, 'index': 22, 'i': 4, 'is_space': False, 'is_punctuation': False, 'is_contraction': False, 'entity': 384, 'is_entity': True, 'shape': 4088098365541558500, 'is_superlative': False, 'children': 0, 'dep': 415}
```


<!-- links -->
[SpaCy]: https://spacy.io
[configuration factories]: https://plandes.github.io/util/doc/config.html#configuration-factory

[norm package]: ../api/zensols.nlp.html#module-zensols.nlp.norm
[LanguageResource]: ../api/zensols.nlp.html#zensols.nlp.lang.LanguageResource
[TokenNormalizer]: ../api/zensols.nlp.html#zensols.nlp.norm.TokenNormalizer
[TokenMapper]: ../api/zensols.nlp.html#zensols.nlp.norm.TokenMapper
[MapTokenNormalizer]: ../api/zensols.nlp.html#zensols.nlp.norm.MapTokenNormalizer
[TokenFeatures]: ../api/zensols.nlp.html#zensols.nlp.feature.TokenFeatures
[detach method]: ../api/zensols.nlp.html#zensols.nlp.feature.DetatchableTokenFeatures.detach
