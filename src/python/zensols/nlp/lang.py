"""Spacy and textacy convenience classes.

"""
__author__ = 'Paul Landes'

import logging
import sys
from dataclasses import dataclass, field
from typing import List
import textacy
from spacy.symbols import ORTH, LEMMA, POS, DEP
#from spacy.symbols import ORTH
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from zensols.config import Config
from zensols.persist import DelegateStash
from zensols.nlp import (
    TokenFeatures,
    TokenNormalizer,
    SpacyFeatureNormalizer,
    NamedEntityRecognitionFeatureNormalizer,
    DependencyFeatureNormalizer,
    PartOfSpeechFeatureNormalizer,
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


@dataclass
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

    :param config: the application configuration used to create the Spacy
                   model
    :param model_name: the Spacy model name (defualts to
                       ``en_core_web_sm``)
    :param lang: the natural language the identify the model
    :param components: additional Spacy components to add to the pipeline
    :param token_normalizer: the token normalizer for methods that use it,
                             i.e. ``features``
    :param special_case_tokens: tokens that will be parsed as one token,
                                i.e. ``</s>``

    """
    config: Config
    lang: str = field(default='en')
    model_name: str = field(default=None)
    components: List = field(default=None)
    disable_components: List = field(default=None)
    token_normalizer: TokenNormalizer = field(default=None)
    special_case_tokens: List = field(default_factory=list)

    def __post_init__(self):
        if self.model_name is None:
            self.model_name = f'{self.lang}_core_web_sm'
        nlp = textacy.load_spacy_lang(self.model_name)
        if self.components is not None:
            for comp in self.components:
                logger.debug(f'adding {comp} to the pipeline')
                comp.add_to_pipeline(nlp)
        self.disable_components = self.disable_components
        self.model = nlp
        if self.token_normalizer is None:
            logger.debug('adding default tokenizer')
            self.token_normalizer = TokenNormalizer()
        for stok in self.special_case_tokens:
            rule = [{ORTH: stok}]
            logger.debug(f'adding special token: {stok} with rule: {rule}')
            self.model.tokenizer.add_special_case(stok, rule)

    def parse(self, text: str, normalize=False) -> Doc:
        """Parse ``text`` in to a Spacy document.

        :param normalize: if ``True`` use the text normalization method
                          ``normalize`` before parsing

        """
        logger.debug(f'creating document with model: {self.model_name}, ' +
                     f'disable components: {self.disable_components}')
        if normalize:
            text = self.normalize(text)
        if self.disable_components is None:
            doc = self.model(text)
        else:
            doc = self.model(text, disable=self.disable_components)
        return doc

    def parse_tokens(self, tokens: List[str]) -> Doc:
        """Just like ``parse`` but process a stream of (already) tokenized
        words/tokens.

        """
        doc = Doc(self.model.vocab, words=tokens)
        for pipe in map(lambda n: self.model.get_pipe(n),
                        self.model.pipe_names):
            pipe(doc)
        return doc

    def features(self, doc: Doc):
        """Generate an iterator of ``TokenFeatures`` instances with features on a per
        token level.

        :return: an iterable of ``TokenFeatures`` objects

        """
        return map(lambda t: TokenFeatures(doc, *t),
                   self.token_normalizer.normalize(doc))

    def normalized_tokens(self, doc: Doc, tn: TokenNormalizer = None) -> iter:
        """Return an iterator of the normalized text of each token.

        """
        tn = self.token_normalizer if tn is None else tn
        return map(lambda t: t[1], tn.normalize(doc))

    def feature_normalizer(self, feature_type: str) -> SpacyFeatureNormalizer:
        """Return a feature normalizer.

        :param feature_type: a string identifying the type of feature that will
                             be normalized, which is one of:
                             ``ent``: named entity
                             ``pos``: part of speech tag
                             ``dep``: dependency
        :return: the feature normalizer or ``None`` if the key did not match.

        """
        cls = {'ent': NamedEntityRecognitionFeatureNormalizer,
               'dep': DependencyFeatureNormalizer,
               'tag': PartOfSpeechFeatureNormalizer}.get(feature_type)
        if cls is not None:
            return cls(self.model.vocab)

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
        text = text.replace('\n', ' ')
        return textacy.preprocess.normalize_whitespace(text)

    def __str__(self):
        return f'model_name: {self.model_name}, lang: {self.lang}'

    def __repr__(self):
        return self.__str__()


@dataclass
class DocStash(DelegateStash):
    """A stash that transforms loaded items in to a SpaCy document.

    All items returned from the delegate must have a ``text`` attribute or
    override ``item_to_text``.

    :param lang_res: used to parse and create the SpaCy documents.

    """
    langres: LanguageResource

    def item_to_text(self, item: object) -> str:
        """Return the text of the item that is loaded with ``load``.  This default
        method uses the ``text`` attribute from ``item``.

        """
        return item.text

    def load(self, name: str):
        item = super().load(name)
        text = self.item_to_text(item)
        return self.langres.parse(text)
