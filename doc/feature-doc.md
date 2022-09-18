# Feature Documents

The primary advantage of this API is for an object oriented structured
representation of natural language using the following classes:

* [FeatureDocument]: Represents a complete document containing at minimum one
  sentence up to a full document.  The constituent language components can
  include paragraphs with white space or one long set of sentences, which
  depends on the use case.
* [FeatureSentence]: Represents a single natural language sentence.
* [FeatureToken]: Represents a single token, which is a word, punctuation,
  named entity or a multi-word expression.

This hierarchy is composed by using [spaCy] to tokenize and chunk sentences by
assigning sentence boundaries.  [FeatureToken] instances are created with
linguistic features such as part of speech tags and named entities.  Then
[FeatureSentence] instances are created with the respective parsed tokens.
Finally a single [FeatureDocument] is created with the sentences of the text
given to the [FeatureDocumentParser].

Additional processing includes optionally performing part of speech tagging,
named entity recognition, and tree parsing also provided by [spaCy], which is
specified in the configuration.  Most of this configuration is provided in the
packages [resource library], so you do not need to know the details for a
default configuration that handles most parsing use cases.  However, the
configuration is easily overridable as given shown in the [natural language
parsing] documentation.


## Example

The following [simple.py] is given in the [examples] directory in the repo,
which starts with an inline [configuration].  First we start by telling the
[configuration API] to load this API package's [resource library]:
```ini
[import]
sections = list: imp_conf

[imp_conf]
type = importini
config_files = list: resource(zensols.nlp): resources/obj.conf
```
In the [simple.py], this is defined as a string in the variable ``CONFIG``.

After importing the package's [resource library], the `[doc_parser]` provides
an entry for the [FeatureDocumentParser].  Next we override its configuration
to only keep only the norm, ent during parsing:
```ini
[doc_parser]
token_feature_ids = set: ent_, tag_
```

With this configuration, creating a parser is straight forward using an
application context and the [configuration API]:
```python
from io import StringIO
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocument, FeatureDocumentParser

fac = ImportConfigFactory(ImportIniConfig(StringIO(CONFIG)))
```

Now we use the factory to get the application context's provided `doc_parser`
entry:
```python
doc_parser: FeatureDocumentParser = fac('doc_parser')
```

To parse natural language text in the to a [FeatureDocument] with the hierarchy
detailed in the [feature documents](feature-documents) section, we call the
parser instance:
```python
sent = 'He was George Washington and first president of the United States.'
doc: FeatureDocument = doc_parser(sent)
for tok in doc.tokens:
    tok.write()
```
This code snippet iterates through all the tokens of the document producing:
```
FeatureToken: org=<He>, norm=<He>
    attributes:
        ent_=-<N>- (str)
        i=0 (int)
        i_sent=0 (int)
        idx=0 (int)
        norm=He (str)
        tag_=PRP (str)
FeatureToken: org=<was>, norm=<was>
    attributes:
        ent_=-<N>- (str)
        i=1 (int)
        i_sent=1 (int)
        idx=3 (int)
        norm=was (str)
        tag_=VBD (str)
...
```

<!-- links -->
[FeatureDocument]: ../api/zensols.nlp.html#zensols.nlp.container.FeatureDocument
[FeatureSentence]: ../api/zensols.nlp.html#zensols.nlp.container.FeatureSentence
[FeatureToken]: ../api/zensols.nlp.html#zensols.nlp.container.FeatureToken
[FeatureDocumentParser]: ../api/zensols.nlp.html#zensols.nlp.container.FeatureDocumentParser

[natural language parsing]: parse.html
[spaCy]: https://spacy.io
[resource library]: https://plandes.github.io/util/doc/config.html#resource-libraries
[configuration]: https://plandes.github.io/util/doc/config.html
[configuration API]: https://plandes.github.io/util/doc/config.html#import-ini-configuration
[examples]: https://github.com/plandes/nlparse/tree/master/example
[simple.py]: https://github.com/plandes/nlparse/blob/master/example/simple.py
