from  __future__ import annotations
"""Parse documents and generate features in an organized taxonomy.

"""
__author__ = 'Paul Landes'

from typing import (
    Tuple, Dict, Any, Sequence, Set, List, Iterable, Type, ClassVar
)
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta, ABC
import logging
import itertools as it
import sys
import re
from io import TextIOBase
import spacy
from spacy.symbols import ORTH
from spacy.tokens import Doc, Span, Token
from spacy.language import Language
from zensols.util import Hasher
from zensols.persist import (
    persisted, PersistedWork, PersistableContainer, Stash
)
from zensols.config import Dictable, ConfigFactory
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


class ComponentInitializer(ABC):
    """Called by :class:`.Component` to do post spaCy initialization.

    """
    @abstractmethod
    def init_nlp_model(self, model: Language, component: Component):
        """Do any post spaCy initialization on the the referred framework."""
        pass


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
    initializers: Tuple[ComponentInitializer, ...] = field(default=())
    """Instances to initialize upon this object's initialization."""

    def __post_init__(self):
        if self.pipe_name is None:
            self.pipe_name = self.name

    def __hash__(self) -> int:
        x = hash(self.name)
        x += 13 * hash(self.pipe_name)
        if self.pipe_config:
            x += 13 * hash(str(self.pipe_config.values()))
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
        for to_init in self.initializers:
            to_init.init_nlp_model(model, self)


@dataclass
class FeatureDocumentParser(PersistableContainer, Dictable, metaclass=ABCMeta):
    """This class parses text in to instances of :class:`.FeatureDocument`
    instances using :meth:`parse`.

    """
    TOKEN_FEATURE_IDS: ClassVar[Set[str]] = FeatureToken.FEATURE_IDS
    """The default value for :obj:`token_feature_ids`."""

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


class SpacyFeatureTokenDecorator(ABC):
    """Implementations can add, remove or modify features on a token.

    """
    @abstractmethod
    def decorate(self, spacy_tok: Token, feature_token: FeatureToken):
        pass


class SpacyFeatureSentenceDecorator(ABC):
    """Implementations can add, remove or modify features on a sentence.

    """
    @abstractmethod
    def decorate(self, spacy_sent: Span, feature_sent: FeatureSentence):
        pass


@dataclass
class SpacyFeatureDocumentParser(FeatureDocumentParser):
    """This langauge resource parses text in to Spacy documents.  Loaded spaCy
    models have attribute ``doc_parser`` set enable creation of factory
    instances from registered pipe components (i.e. specified by
    :class:`.Component`).

    Configuration example::

        [doc_parser]
        class_name = zensols.nlp.SpacyFeatureDocumentParser
        lang = en
        model_name = ${lang}_core_web_sm

    """
    _MODELS = {}
    """Contains cached models, such as ``en_core_web_sm``."""

    config_factory: ConfigFactory = field()
    """A configuration parser optionally used by pipeline :class:`.Component`
    instances.

    """
    name: str = field()
    """The name of the parser, which is taken from the section name when created
    with a :class:`~zensols.config.ConfigFactory`.

    """
    lang: str = field(default='en')
    """The natural language the identify the model."""

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

    token_decorators: Sequence[SpacyFeatureTokenDecorator] = field(default=())
    """A list of decorators that can add, remove or modify features on a token.

    """
    sentence_decorators: Sequence[SpacyFeatureSentenceDecorator] = field(
        default=())
    """A list of decorators that can add, remove or modify features on a
    sentence.

    """
    disable_component_names: Sequence[str] = field(default=None)
    """Components to disable in the spaCy model when creating documents in
    :meth:`parse`.

    """
    token_normalizer: TokenNormalizer = field(default=None)
    """The token normalizer for methods that use it, i.e. ``features``."""

    special_case_tokens: List = field(default_factory=list)
    """Tokens that will be parsed as one token, i.e. ``</s>``."""

    doc_class: Type[FeatureDocument] = field(default=FeatureDocument)
    """The type of document instances to create."""

    sent_class: Type[FeatureSentence] = field(default=FeatureSentence)
    """The type of sentence instances to create."""

    token_class: Type[FeatureToken] = field(default=SpacyFeatureToken)
    """The type of document instances to create."""

    sentence_filters: List[str] = field(default_factory=list)
    """A list of functions that return a boolean used to filter sentences.

    """
    remove_empty_sentences: bool = field(default=False)
    """If ``True``, remove sentences that only have space tokens."""

    reload_components: bool = field(default=False)
    """Removes, then re-adds components for cached models.  This is helpful for
    when there are component configurations that change on reruns with a
    difference application context but in the same Python interpreter session.

    A spaCy component can get other instances via :obj:`config_factory`, but if
    this is ``False`` it will be paired with the first instance of this class
    and not the new ones created with a new configuration factory.

    """
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
                        logger.debug(f'adding {comp} ({id(comp)}) to pipeline')
                    comp.init(nlp)

    def _remove_components(self, nlp: Language):
        for comp in self.components:
            name, comp = nlp.remove_pipe(comp.pipe_name)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'removed {name} ({id(comp)})')

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
            # pipe components can create other application context instance via
            # the :obj:`config_factory` with access to this instance
            nlp.doc_parser = self
            self._add_components(nlp)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'adding {mkey} to cached models ({len(self._MODELS)})')
            self._MODELS[mkey] = nlp
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'cached models: {len(self._MODELS)}')
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'cached model: {mkey} ({self.model_name})')
            if self.reload_components:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f're-adding components to {id(self)}')
                nlp.doc_parser = self
                self._remove_components(nlp)
                self._add_components(nlp)
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

    @classmethod
    def clear_models(self):
        """Clears all cached models."""
        self._MODELS.clear()

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
            logger.info(f'parsed text: <{self._trunc(text)}>')
        if logger.isEnabledFor(logging.DEBUG):
            doc_text = self._trunc(str(doc))
            logger.debug(f'parsed document: <{doc_text}>')
        return doc

    def get_dictable(self, doc: Doc) -> Dictable:
        """Return a dictionary object graph and pretty prints spaCy docs.

        """
        return _DictableDoc(doc)

    def _normalize_tokens(self, doc: Doc, *args, **kwargs) -> \
            Iterable[FeatureToken]:
        """Generate an iterator of :class:`.FeatureToken` instances with features on a
        per token level.

        """
        if logger.isEnabledFor(logging.DEBUG):
            doc_text = self._trunc(str(doc))
            logger.debug(f'normalizing features in {doc_text}')
            logger.debug(f'args: <{args}>')
            logger.debug(f'kwargs: <{kwargs}>')
        tokens: Tuple[FeatureToken, ...] = \
            map(lambda tup: self._create_token(*tup, *args, **kwargs),
                self.token_normalizer.normalize(doc))
        return tokens

    def _decorate_token(self, spacy_tok: Token, feature_token: FeatureToken):
        decorator: SpacyFeatureTokenDecorator
        for decorator in self.token_decorators:
            decorator.decorate(spacy_tok, feature_token)

    def _create_token(self, tok: Token, norm: Tuple[Token, str],
                      *args, **kwargs) -> FeatureToken:
        tp: Type[FeatureToken] = self.token_class
        ft: FeatureToken = tp(tok, norm, *args, **kwargs)
        self._decorate_token(tok, ft)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'detaching using features: {self.token_feature_ids}')
        return ft.detach(self.token_feature_ids)

    def _decorate_sent(self, spacy_sent: Span, feature_sent: FeatureSentence):
        decorator: SpacyFeatureSentenceDecorator
        for decorator in self.sentence_decorators:
            decorator.decorate(spacy_sent, feature_sent)

    def _create_sent(self, spacy_sent: Span, stoks: Iterable[FeatureToken],
                     text: str) -> FeatureSentence:
        sent: FeatureSentence = self.sent_class(tuple(stoks), text, spacy_sent)
        self._decorate_sent(spacy_sent, sent)
        return sent

    def _filter_sent(self, sent: Span, fsent: FeatureSentence) -> \
            List[FeatureSentence]:
        def remove_empty_sentences(sent, fsent) -> bool:
            return len(sent) > 0 and not all(map(lambda t: t.is_space, sent))

        filters = list(map(eval, self.sentence_filters))
        if self.remove_empty_sentences:
            filters.append(remove_empty_sentences)
        if len(filters) == 0:
            return True
        else:
            return all(map(lambda f: f(sent, fsent), filters))

    def _create_sents(self, doc: Doc) -> List[FeatureSentence]:
        """Create sentences from a spaCy doc."""
        toks: Tuple[FeatureToken, ...] = tuple(self._normalize_tokens(doc))
        sents: List[FeatureSentence] = []
        ntoks = len(toks)
        tix = 0
        sent: Span
        for sent in doc.sents:
            e = sent[-1].i
            stoks = []
            while tix < ntoks:
                tok = toks[tix]
                if tok.i <= e:
                    stoks.append(tok)
                else:
                    break
                tix += 1
            fsent: FeatureSentence = self._create_sent(sent, stoks, sent.text)
            if not self._filter_sent(sent, fsent):
                continue
            sents.append(fsent)
        return sents

    def from_spacy_doc(self, doc: Doc, *args, text: str = None,
                       **kwargs) -> FeatureDocument:
        """Create s :class:`.FeatureDocument` from a spaCy doc.

        :param doc: the spaCy generated document to transform in to a feature
                    document

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        :param args: the arguments used to create the FeatureDocument instance

        :param kwargs: the key word arguments used to create the
                       FeatureDocument instance

        """
        text = doc.text if text is None else text
        sents: List[FeatureSentence] = self._create_sents(doc)
        try:
            return self.doc_class(tuple(sents), text, doc, *args, **kwargs)
        except Exception as e:
            raise ParseError(
                f'Could not parse <{text}> for {self.doc_class} ' +
                f"with args {args} for parser '{self.name}'") from e

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        if not isinstance(text, str):
            raise ParseError(
                f'Expecting string text but got: {text} ({type(str)})')
        doc: Doc = self.parse_spacy_doc(text)
        return self.from_spacy_doc(doc, *args, text=text, **kwargs)

    def to_spacy_doc(self, doc: FeatureDocument, norm: bool = True,
                     add_features: Set[str] = None) -> Doc:
        """Convert a feature document back in to a spaCy document.

        **Note**: not all data is copied--only text, ``pos_``, ``tag_``,
        ``lemma_`` and ``dep_``.

        :param doc: the spaCy doc to convert

        :param norm: whether to use the normalized text as the ``orth_`` spaCy
                     token attribute or ``text``

        :pram add_features: whether to add POS, NER tags, lemmas, heads and
                            dependnencies

        :return: the feature document with copied data from ``doc``

        """
        def conv_iob(t: FeatureToken) -> str:
            if t.ent_iob_ == 'O':
                return 'O'
            return f'{t.ent_iob_}-{t.ent_}'

        if norm:
            words = list(doc.norm_token_iter())
        else:
            words = [t.text for t in doc.token_iter()]
        if add_features is None:
            add_features = set('pos tag lemma head dep ent'.split())
        sent_starts = [False] * len(words)
        sidx = 0
        for sent in doc:
            sent_starts[sidx] = True
            sidx += len(sent)
        params = dict(vocab=self.model.vocab,
                      words=words,
                      spaces=[True] * len(words),
                      sent_starts=sent_starts)
        if add_features and doc.token_len > 0:
            assert len(words) == doc.token_len
            tok = next(iter(doc.token_iter()))
            if hasattr(tok, 'pos_') and 'pos' in add_features:
                params['pos'] = [t.pos_ for t in doc.token_iter()]
            if hasattr(tok, 'tag_') and 'tag' in add_features:
                params['tags'] = [t.tag_ for t in doc.token_iter()]
            if hasattr(tok, 'lemma_') and 'lemma' in add_features:
                params['lemmas'] = [t.lemma_ for t in doc.token_iter()]
            if hasattr(tok, 'head_') and 'head' in add_features:
                params['heads'] = [t.head_ for t in doc.token_iter()]
            if hasattr(tok, 'dep_') and 'dep' in add_features:
                params['deps'] = [t.dep_ for t in doc.token_iter()]
            if hasattr(tok, 'ent_') and 'ent' in add_features:
                params['ents'] = [conv_iob(t) for t in doc.token_iter()]
        return Doc(**params)

    def __str__(self):
        return f'model_name: {self.model_name}, lang: {self.lang}'

    def __repr__(self):
        return self.__str__()


@dataclass
class CachingFeatureDocumentParser(FeatureDocumentParser):
    """A document parser that persists previous parses using the hash of the
    text as a key.  Caching is optional given the value of :obj:`stash`, which
    is useful in cases this class is extended using other use cases other than
    just caching.

    """
    delegate: FeatureDocumentParser = field()
    """Used to parse in to documents on cache misses."""

    stash: Stash = field(default=None)
    """The stash that persists the feature document instances.  If this is not
    provided, no caching will happen.

    """
    hasher: Hasher = field(default_factory=Hasher)
    """Used to hash the natural langauge text in to string keys."""

    @property
    def token_feature_ids(self) -> Set[str]:
        return self.delegate.token_feature_ids

    def _hash_text(self, text: str) -> str:
        self.hasher.reset()
        self.hasher.update(text)
        return self.hasher()

    def _load_or_parse(self, text: str, dump: bool, *args, **kwargs) -> \
            Tuple[FeatureDocument, str, bool]:
        key: str = self._hash_text(text)
        doc: FeatureDocument = None
        loaded: bool = False
        if self.stash is not None:
            doc = self.stash.load(key)
        if doc is None:
            doc = self.delegate.parse(text, *args, **kwargs)
            if dump and self.stash is not None:
                self.stash.dump(key, doc)
        else:
            loaded = True
        return doc, key, loaded

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        return self._load_or_parse(text, True, *args, **kwargs)[0]

    def clear(self):
        """Clear the caching stash."""
        if self.stash is not None:
            self.stash.clear()


@dataclass
class WhiteSpaceTokenizerFeatureDocumentParser(SpacyFeatureDocumentParser):
    """This class parses text in to instances of :class:`.FeatureDocument`
    instances using :meth:`parse`.  This parser does no sentence chunking so
    documents have one and only one sentence for each parse.

    """
    _TOK_REGEX: ClassVar[re.Pattern] = re.compile(r'\S+')
    """The whitespace regular expression for splitting tokens."""

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        toks: List[FeatureToken] = []
        m: re.Match
        for i, m in zip(it.count(), re.finditer(self._TOK_REGEX, text)):
            tok = FeatureToken(i, m.start(), 0, m.group(0))
            tok.default_detached_feature_ids = \
                FeatureToken.REQUIRED_FEATURE_IDS
            toks.append(tok)
        sent = self.sent_class(tokens=tuple(toks), text=text)
        return self.doc_class(sents=(sent,), text=text, *args, **kwargs)
