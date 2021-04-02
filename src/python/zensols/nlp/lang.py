"""Spacy and textacy convenience classes.

"""
__author__ = 'Paul Landes'

from typing import List, Iterable, Dict, Any
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from io import TextIOBase
import spacy
from spacy.symbols import ORTH
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy.language import Language
from spacy.lang.en import English
from zensols.config import Configurable, Dictable
from zensols.persist import DelegateStash
from zensols.nlp import TokenFeatures, TokenNormalizer

logger = logging.getLogger(__name__)


@dataclass
class DictableDoc(Dictable):
    """Utility class to pretty print and serialize Spacy documents.

    """
    doc: Doc = field(repr=False)

    def _write_token(self, tok: Token, depth: int, writer: TextIOBase):
        s = (f'{tok}: tag={tok.tag_}, pos={tok.pos_}, stop={tok.is_stop}, ' +
             f'lemma={tok.lemma_}, dep={tok.dep_}')
        self._write_line(s, depth, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              token_limit: int = sys.maxsize):
        """Pretty print the document.

        :param token_limit: the max number of tokens to write, which defaults
                            to all of them

        """
        text = self._trunc(str(self.doc.text))
        self._write_line(f'text: {text}', depth, writer)
        self._write_line('tokens:', depth, writer)
        for sent in self.doc.sents:
            self._write_line(self._trunc(str(sent)), depth + 1, writer)
            for t in it.islice(sent, token_limit):
                self._write_token(t, depth + 2, writer)
        self._write_line('entities:', depth, writer)
        for ent in self.doc.ents:
            self._write_line(f'{ent}: {ent.label_}', depth + 1, writer)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        sents = tuple(self.doc.sents)
        em = {}
        for e in self.doc.ents:
            for tok in self.doc[e.start:e.end]:
                em[tok.i] = e.label_

        def tok_json(t):
            return {'tag': t.tag_, 'pos': t.pos_,
                    'is_stop': t.is_stop, 'lemma': t.lemma_, 'dep': t.dep_,
                    'text': t.text, 'idx': t.idx,
                    'ent': None if t.i not in em else em[t.i],
                    'childs': tuple(map(lambda c: c.i, t.children))}

        def sent_json(idx):
            s = sents[idx]
            return {t.i: tok_json(t) for t in self.doc[s.start:s.end]}

        return {'text': self.doc.text,
                'sents': {i: sent_json(i) for i in range(len(sents))},
                'ents': [(str(e), e.label_,) for e in self.doc.ents]}


@dataclass
class LanguageResource(object):
    """This langauge resource parses text in to Spacy documents.

    Configuration example::

        [default_langres]
        class_name = zensols.nlp.LanguageResource
        lang = en
        model_name = ${lang}_core_web_sm

    """
    MODELS = {}

    config: Configurable = field()
    """The application configuration used to create the Spacy model."""

    lang: str = field(default='en')
    """The natural language the identify the model."""

    model: Language = field(default=None)
    """The spaCy model, or ``None`` (the default) to create a new one using
    ``model_name``.

    """

    model_name: str = field(default=None)
    """The Spacy model name (defualts to ``en_core_web_sm``); this is ignored
    if ``model`` is not ``None``.

    """

    components: List = field(default=None)
    """Additional Spacy components to add to the pipeline."""

    disable_components: List = field(default=None)
    """Components to disable in the spaCy model when creating documents in
    :meth:`parse`.

    """

    token_normalizer: TokenNormalizer = field(default=None)
    """The token normalizer for methods that use it, i.e. ``features``."""

    special_case_tokens: List = field(default_factory=list)
    """Tokens that will be parsed as one token, i.e. ``</s>``."""

    def __post_init__(self):
        if self.model_name is None:
            self.model_name = f'{self.lang}_core_web_sm'
        nlp = self.model
        if nlp is None:
            # cache model in class space
            nlp = self.MODELS.get(self.model_name)
            if nlp is None:
                nlp = spacy.load("en_core_web_sm")
                self.MODELS[self.model_name] = nlp
            self.model = nlp
        if self.components is not None:
            for comp in self.components:
                logger.debug(f'adding {comp} to the pipeline')
                nlp.add_pipe(comp)
        self.disable_components = self.disable_components
        if self.token_normalizer is None:
            logger.debug('adding default tokenizer')
            self.token_normalizer = TokenNormalizer()
        for stok in self.special_case_tokens:
            rule = [{ORTH: stok}]
            logger.debug(f'adding special token: {stok} with rule: {rule}')
            self.model.tokenizer.add_special_case(stok, rule)

    def parse(self, text: str) -> Doc:
        """Parse ``text`` in to a Spacy document.

        """
        logger.debug(f'creating document with model: {self.model_name}, ' +
                     f'disable components: {self.disable_components}')
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

    def features(self, doc: Doc) -> Iterable[TokenFeatures]:
        """Generate an iterator of :class:`.TokenFeatures` instances with features on a
        per token level.

        """
        return map(lambda t: TokenFeatures(doc, *t),
                   self.token_normalizer.normalize(doc))

    def normalized_tokens(self, doc: Doc, tn: TokenNormalizer = None) -> \
            Iterable[str]:
        """Return an iterator of the normalized text of each token.

        """
        tn = self.token_normalizer if tn is None else tn
        return map(lambda t: t[1], tn.normalize(doc))

    def tokenizer(self, text: str):
        """Create a simple Spacy tokenizer.  Currently only English is supported.

        """
        if self.lang == 'en':
            tokenizer = English().Defaults.create_tokenizer(self.model)
        else:
            raise ValueError(f'no such language: {self.lang}')
        return tokenizer(text)

    def __str__(self):
        return f'model_name: {self.model_name}, lang: {self.lang}'

    def __repr__(self):
        return self.__str__()


@dataclass
class DocStash(DelegateStash):
    """A stash that transforms loaded items in to a SpaCy document.

    All items returned from the delegate must have a ``text`` attribute or
    override ``item_to_text``.

    """
    langres: LanguageResource = field()
    """Used to parse and create the SpaCy documents."""

    def item_to_text(self, item: object) -> str:
        """Return the text of the item that is loaded with ``load``.  This default
        method uses the ``text`` attribute from ``item``.

        """
        return item.text

    def load(self, name: str):
        item = super().load(name)
        text = self.item_to_text(item)
        return self.langres.parse(text)
