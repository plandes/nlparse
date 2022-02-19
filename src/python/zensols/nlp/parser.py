"""Parse documents and generate features in an organized taxonomy.

"""
__author__ = 'Paul Landes'

from typing import (
    Tuple, Dict, Any, Sequence, Set, List, Iterable, Type, ClassVar
)
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import logging
import itertools as it
import sys
from io import TextIOBase
import spacy
from spacy.symbols import ORTH
from spacy.tokens import Doc, Span, Token
from spacy.language import Language
from zensols.persist import persisted, PersistedWork
from zensols.config import Dictable, Configurable, ConfigFactory
from . import (
    ParseError, TokenNormalizer, FeatureToken, SpacyFeatureToken,
    FeatureSentence, FeatureDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class _DictableDoc(Dictable):
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

    pipe_add_kwargs: Dict[str, Any] = field(default_factory=dict)
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating pipe {self.pipe_name} with args: ' +
                         f'{self.pipe_add_kwargs}')
        if self.pipe_config is None:
            model.add_pipe(self.pipe_name, **self.pipe_add_kwargs)
        else:
            model.add_pipe(self.pipe_name, config=self.pipe_config,
                           **self.pipe_add_kwargs)


@dataclass
class FeatureDocumentParser(Dictable, metaclass=ABCMeta):
    TOKEN_FEATURE_IDS: ClassVar[Set[str]] = FeatureToken.FEATURE_IDS
    """The default value for :obj:`token_feature_ids`."""

    name: str = field()
    """The name of the parser, which is taken from the section name when created
    with a :class:`~zensols.config.ConfigFactory`.

    """
    def __post_init__(self):
        super().__init__()

    @abstractmethod
    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        """Parse text or a text as a list of sentences.

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        :param args: the arguments used to create the FeatureDocument instance

        :param kwargs: the key word arguments used to create the
                       FeatureDocument instance

        """

    def __call__(self, text: str, *args, **kwargs) -> FeatureDocument:
        """Invoke :meth:`parse` with the context arguments.

        :see: :meth:`parse`

        """
        return self.parse(text, *args, **kwargs)


@dataclass
class SpacyFeatureDocumentParser(FeatureDocumentParser):
    """This langauge resource parses text in to Spacy documents.  Loaded spaCy
  models have attribute ``langres`` set enable creation of factory instances
  from registered pipe components (i.e. specified by :class:`.Component`).

    Configuration example::

        [doc_parser]
        class_name = zensols.nlp.SpacyFeatureDocumentParser
        lang = en
        model_name = ${lang}_core_web_sm

    """
    _MODELS = {}
    """Contains cached models, such as ``en_core_web_sm``."""

    lang: str = field(default='en')
    """The natural language the identify the model."""

    """The Spacy model name (defualts to ``en_core_web_sm``); this is ignored
    model_name: str = field(default=None)
    if ``model`` is not ``None``.

    """
    model_name: str = field(default=None)
    """The Spacy model name (defualts to ``en_core_web_sm``); this is ignored
    if ``model`` is not ``None``.

    """
    token_feature_ids: Set[str] = field(
        default_factory=lambda: FeatureDocumentParser.TOKEN_FEATURE_IDS)
    """The features to keep from spaCy tokens.

    :see: :obj:`TOKEN_FEATURE_IDS`

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

    feature_type: Type[FeatureToken] = field(default=SpacyFeatureToken)
    """The class to use for instances created by :meth:`features`."""

    doc_class: Type[FeatureDocument] = field(default=FeatureDocument)
    """The type of document instances to create."""

    sent_class: Type[FeatureSentence] = field(default=FeatureSentence)
    """The type of sentence instances to create."""

    token_class: Type[FeatureToken] = field(default=SpacyFeatureToken)
    """The type of document instances to create."""

    remove_empty_sentences: bool = field(default=False)
    """If ``True``, remove sentences that only have space tokens."""

    def __post_init__(self):
        super().__post_init__()
        self._model = PersistedWork('_model', self)

    def _create_model_key(self) -> str:
        """Create a unique key used for storing expensive-to-create spaCy language
        models in :obj:`_MODELS`.

        """
        comps = sorted(map(lambda c: f'{c.pipe_name}:{hash(c)}',
                           self.components))
        comp_str = '-' + '|'.join(comps)
        return f'{self.model_name}{comp_str}'

    def _create_model(self) -> Language:
        """Load, configure and return a new spaCy model instance."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading model: {self.model_name}')
        nlp = spacy.load(self.model_name)
        # pipe components can create other application context instance via the
        # :obj:`config_factory` with access to this instance
        nlp.langres = self
        return nlp

    def _add_components(self, nlp: Language):
        """Add components to the pipeline that was just created."""
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
            self._add_components(nlp)
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

    def parse_spacy_doc(self, text: str) -> Doc:
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

    def get_dictable(self, doc: Doc) -> Dictable:
        """Return a dictionary object graph and pretty prints spaCy docs.

        """
        return _DictableDoc(doc)

    def _normalize_tokens(self, doc: Doc) -> Iterable[FeatureToken]:
        """Generate an iterator of :class:`.TokenFeatures` instances with features on a
        per token level.

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'parsing features in {doc}')
        tokens: Tuple[FeatureToken] = \
            map(lambda tup: self._create_token(*tup),
                self.token_normalizer.normalize(doc))
        return tokens

    def _create_token(self, tok: Token, norm: Tuple[Token, str]) -> FeatureToken:
        tp: Type[FeatureToken] = self.token_class
        return tp(tok, norm)

    def _create_sent(self, spacy_sent: Span, stoks: Iterable[FeatureToken],
                     text: str) -> FeatureSentence:
        return self.sent_class(tuple(stoks), text, spacy_sent)

    def _from_string(self, text: str) -> Tuple[Doc, List[FeatureSentence]]:
        """Parse a document from a string.

        """
        doc: Doc = self.parse_spacy_doc(text)
        toks: Tuple[FeatureToken] = tuple(self._normalize_tokens(doc))
        ntoks = len(toks)
        tix = 0
        sents = []
        sent: Span
        for sent in doc.sents:
            if self.remove_empty_sentences and \
               (all(map(lambda t: t.is_space, sent)) or len(sent) == 0):
                continue
            e = sent[-1].i
            stoks = []
            while tix < ntoks:
                tok = toks[tix]
                if tok.i <= e:
                    stoks.append(tok)
                else:
                    break
                tix += 1
            sents.append(self._create_sent(sent, stoks, sent.text))
        return doc, sents

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        if not isinstance(text, str):
            raise ParseError(f'Expecting string text but got: {text}')
        spacy_doc, sents = self._from_string(text)
        try:
            return self.doc_class(sents, spacy_doc, *args, **kwargs)
        except Exception as e:
            raise ParseError(
                f'Could not parse <{text}> for {self.doc_class} ' +
                f"with args {args} for parser '{self.name}'") from e

    def __str__(self):
        return f'model_name: {self.model_name}, lang: {self.lang}'

    def __repr__(self):
        return self.__str__()
