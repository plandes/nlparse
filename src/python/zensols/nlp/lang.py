"""Spacy and textacy convenience classes.

"""
__author__ = 'Paul Landes'

import logging
import sys
import textacy
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from zensols.actioncli import (
    SingleClassConfigManager,
    Config,
    DelegateStash,
    StashFactory,
)
from zensols.nlp import (
    TokenFeatures,
    TokenNormalizer,
)

logger = logging.getLogger(__name__)


class DocUtil(object):
    """Utility class to pretty print and serialize Spacy documents.

    """
    @staticmethod
    def write(doc, writer=sys.stdout):
        """Pretty print ``doc`` using ``writer``, which defauls to standard out.

        """
        writer.write(f'text: {doc.text}\n')
        writer.write('tokens:\n')
        for t in doc:
            writer.write(f'  {t}: tag={t.tag_}, pos={t.pos_}, ' +
                         f'stop={t.is_stop}, lemma={t.lemma_}, dep={t.dep_}\n')
        writer.write('entities:\n')
        for ent in doc.ents:
            writer.write(f'  {ent}: {ent.label_}\n')

    @staticmethod
    def to_json(doc):
        """Convert ``doc`` to a JSON Python object.

        """
        sents = tuple(doc.sents)
        em = {}
        for e in doc.ents:
            for tok in doc[e.start:e.end]:
                em[tok.i] = e.label_

        def tok_json(t):
            return {'tag': t.tag_, 'pos': t.pos_,
                    'is_stop': t.is_stop, 'lemma': t.lemma_, 'dep': t.dep_,
                    'text': t.text, 'idx': t.idx,
                    'ent': None if t.i not in em else em[t.i],
                    'childs': tuple(map(lambda c: c.i, t.children))}

        def sent_json(idx):
            s = sents[idx]
            return {t.i: tok_json(t) for t in doc[s.start:s.end]}

        return {'text': doc.text,
                'sents': {i: sent_json(i) for i in range(len(sents))},
                'ents': [(str(e), e.label_,) for e in doc.ents]}


class LanguageResource(object):
    """This langauge resource parses text in to Spacy documents.  It also uses the
    textacy library to normalize text white sapce to generate better Spacy
    document parses.

    Don't create instances of this directly.  Instead use
    ``LanguageResourceFactory`` created with application contexts with entries
    like this:

    [default_langres]
    lang = en
    model_name = ${lang}_core_web_sm

    """
    def __init__(self, config: Config, model_name: str, lang: str = 'en',
                 components: list = None, disable_components: list = None,
                 token_normalizer: TokenNormalizer = None):
        """Initialize the language resource.

        :param config: the application configuration used to create the Spacy
                       model
        :param model_name: the Spacy model name (i.e. ``en_core_web_sm``)
        :param lang: the natural language the identify the model
        :param components: additional Spacy components to add to the pipeline
        :param tn: the token normalizer for methods that use it, i.e. ``features``

        """
        self.model_name = model_name
        self.lang = lang
        nlp = textacy.cache.load_spacy(model_name)
        if components is not None:
            for comp in components:
                comp.add_to_pipeline(nlp)
        self.disable_components = disable_components
        self.model = nlp
        self.token_normalizer = token_normalizer

    def parse(self, text: str) -> Doc:
        """Parse ``text`` in to a Spacy document.

        """
        logger.debug(f'creating document with model: {self.model_name}, ' +
                     f'disable components: {self.disable_components}')
        text = self.normalize(text)
        if self.disable_components is None:
            doc = self.model(text)
        else:
            doc = self.model(text, disable=self.disable_components)
        return doc

    def features(self, doc: Doc,
                 token_normalizer: TokenNormalizer = None) -> iter:
        """Generate an iterator of ``TokenFeatures`` instances with features on a per
        token level.

        """
        tn = self.token_normalizer if token_normalizer is None else token_normalizer
        return map(lambda t: TokenFeatures(doc, *t), tn.normalize(doc))

    def normalized_tokens(self, doc: Doc, tn: TokenNormalizer) -> iter:
        """Return an iterator of the normalized text of each token.

        """
        return map(lambda t: t[1], tn.normalize(doc))

    def tokenizer(self, text: str):
        """Create a simple Spacy tokenizer.  Currently only English is supported.

        """
        if self.lang == 'en':
            tokenizer = English().Defaults.create_tokenizer(self.model)
        else:
            raise ValueError(f'no such language: {self.lang}')
        return tokenizer(text)

    @staticmethod
    def normalize(text):
        return textacy.preprocess.normalize_whitespace(text)

    def __str__(self):
        return f'model_name: {self.model_name}, lang: {self.lang}'

    def __repr__(self):
        return self.__str__()


class DocStash(DelegateStash):
    """A stash that transforms loaded items in to a SpaCy document.

    All items returned from the delegate must have a ``text`` attribute or
    override ``item_to_text``.

    """
    def __init__(self, delegate, lang_res: LanguageResource):
        """Initialize.

        :param delegate: the delegate to use objects that have the ``text``
                         attribute
        :param lang_res: used to parse and create the SpaCy documents.

        """
        super(DocStash, self).__init__(delegate)
        self.lang_res = lang_res

    def item_to_text(self, item: object) -> str:
        """Return the text of the item that is loaded with ``load``.  This default
        method uses the ``text`` attribute from ``item``.

        """
        return item.text

    def load(self, name: str):
        item = super(DocStash, self).load(name)
        text = self.item_to_text(item)
        return self.lang_res.parse(text)


StashFactory.register(DocStash)


class LanguageResourceFactory(SingleClassConfigManager):
    """Creates instances of ``LanguageResource`` from application context
    configuration.  See that class for details.

    """
    INSTANCE_CLASSES = {}

    def __init__(self, config, *args, **kwargs):
        super(LanguageResourceFactory, self).__init__(
            config,
            cls=LanguageResource,
            pattern='{name}_langres',
            stash=None,
            *args, **kwargs)
