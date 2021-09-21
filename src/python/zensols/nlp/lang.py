"""Spacy and textacy convenience classes.

"""
__author__ = 'Paul Landes'

from typing import List, Sequence, Iterable, Dict, Any, Type
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
from zensols.persist import (
    DelegateStash, persisted, PersistedWork, PersistableContainer
)
from . import ParseError, TokenFeatures, SpacyTokenFeatures, TokenNormalizer

logger = logging.getLogger(__name__)


@dataclass
class DictableDoc(Dictable):
    """Utility class to pretty print and serialize Spacy documents.

    """
    doc: Doc = field(repr=False)
    """The document from which to create a :class:`.dict`."""

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
class Component(object):
    """A pipeline component to be added to the spaCy model.  There are a list of
    these set in the :class:`.LanguageResource`.

    """
    name: str = field()
    """The section name."""

    pipe_name: str = field(default=None)
    """The pipeline component name to add to the pipeline.  If ``None``, use
    :obj:`name`.

    """

    pipe_config: Dict[str, str] = field(default=None)
    """The configuration to add with the ``config`` kwarg in the
    :meth:`.Language.add_pipe` call to the spaCy model.

    """

    pipe_add_kwargs: Dict[str, Any] = field(default=dict)
    """Arguments to add along with the call to
    :meth:`~spacy.language.Language.add_pipe`.

    """

    modules: Sequence[str] = field(default=())
    """The module to import before adding component pipelines.  This will register
    components mentioned in :obj:`components` when the resepctive module is
    loaded.

    """
    def __post_init__(self):
        if self.pipe_name is None:
            self.pipe_name = self.name

    def __hash__(self) -> int:
        x = hash(self.name)
        x += 13 * hash(self.pipe_name)
        if self.pipe_config:
            x += 13 * hash(self.pipe_config.values())
        return x

    def init(self, model: Language):
        """Initialize the component and add it to the NLP pipe line.  This base class
        implementation loads the :obj:`module`, then calls
        :meth:`.Language.add_pipe`.

        :param model: the model to add the spaCy model (``nlp`` in their
                      parlance)

        """
        for mod in self.modules:
            __import__(mod)
        if self.pipe_config is None:
            model.add_pipe(self.pipe_name, **self.pipe_add_kwargs)
        else:
            model.add_pipe(self.pipe_name, config=self.pipe_config,
                           **self.pipe_add_kwargs)


@dataclass
class LanguageResource(PersistableContainer):
    """This langauge resource parses text in to Spacy documents.

    Configuration example::

        [default_langres]
        class_name = zensols.nlp.LanguageResource
        lang = en
        model_name = ${lang}_core_web_sm

    """
    _MODELS = {}
    """Contains cached models, such as ``en_core_web_sm``."""

    name: str = field()
    """The name of the language resource, which is taken from the section name when
    created with a :class:`~zensols.config.ConfigFactory`.

    """

    config: Configurable = field()
    """The application configuration used to create the Spacy model."""

    lang: str = field(default='en')
    """The natural language the identify the model."""

    model_name: str = field(default=None)
    """The Spacy model name (defualts to ``en_core_web_sm``); this is ignored
    if ``model`` is not ``None``.

    """

    components: Sequence[Component] = field(default=())
    """Additional Spacy components to add to the pipeline."""

    disable_component_names: Sequence[str] = field(default=None)
    """Components to disable in the spaCy model when creating documents in
    :meth:`parse`.

    """

    token_normalizer: TokenNormalizer = field(default=None)
    """The token normalizer for methods that use it, i.e. ``features``."""

    special_case_tokens: List = field(default_factory=list)
    """Tokens that will be parsed as one token, i.e. ``</s>``."""

    feature_type: Type[TokenFeatures] = field(default=SpacyTokenFeatures)
    """The class to use for instances created by :meth:`features`."""

    def __post_init__(self):
        super().__init__()
        self._model = PersistedWork('_model', self)

    def _create_model_key(self) -> str:
        comps = sorted(map(lambda c: f'{c.pipe_name}:{hash(c)}',
                           self.components))
        comp_str = '-' + '|'.join(comps)
        return f'{self.model_name}{comp_str}'

    def _create_model(self) -> Language:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading model: {self.model_name}')
        nlp = spacy.load(self.model_name)
        if self.components is not None:
            comp: Component
            for comp in self.components:
                if comp.pipe_name in nlp.pipe_names:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'{comp} already registered--skipping')
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'adding {comp} to the pipeline')
                    comp.init(nlp)
        return nlp

    @property
    @persisted('_model')
    def model(self) -> Language:
        """The spaCy model.  On first access, this creates a new instance using
        ``model_name``.

        """
        mkey: str = self._create_model_key()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'model key: {mkey}')
        if self.model_name is None:
            self.model_name = f'{self.lang}_core_web_sm'
        # cache model in class space
        nlp: Language = self._MODELS.get(mkey)
        if nlp is None:
            nlp: Language = self._create_model()
            self._MODELS[mkey] = nlp
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'cached model: {mkey} ({self.model_name})')
        if self.token_normalizer is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('adding default tokenizer')
            self.token_normalizer = TokenNormalizer()
        for stok in self.special_case_tokens:
            rule = [{ORTH: stok}]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'adding special token: {stok} with rule: {rule}')
            nlp.tokenizer.add_special_case(stok, rule)
        return nlp

    def parse(self, text: str) -> Doc:
        """Parse ``text`` in to a Spacy document.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating document with model: {self.model_name}, ' +
                         f'disable components: {self.disable_component_names}')
        if self.disable_component_names is None:
            doc = self.model(text)
        else:
            doc = self.model(text, disable=self.disable_component_names)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'parsed document: <{text}> -> {doc}')
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
        tp: Type[TokenFeatures] = self.feature_type
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'parsing features in {doc}')
        norm_doc = tuple(
            map(lambda t: tp(doc, *t), self.token_normalizer.normalize(doc)))
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'normalized document: <{doc}> -> {norm_doc} ({tp})')
        return norm_doc

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
            raise ParseError(f'no such language: {self.lang}')
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
