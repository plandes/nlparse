"""Parse documents and generate features in an organized taxonomy.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Tuple, Dict, List, Iterable, Any, Sequence, Set, ClassVar, Union
)
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta, ABC
import logging
from io import StringIO
from spacy.language import Language
from zensols.util import Hasher
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.persist import PersistableContainer, Stash
from zensols.config import Dictable
from . import NLPError, FeatureToken, FeatureSentence, FeatureDocument

logger = logging.getLogger(__name__)


class ComponentInitializer(ABC):
    """Called by :class:`.Component` to do post spaCy initialization.

    """
    @abstractmethod
    def init_nlp_model(self, model: Language, component: Component):
        """Do any post spaCy initialization on the the referred framework."""
        pass


@dataclass
class Component(object):
    """A pipeline component to be added to the spaCy model.

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
    """The module to import before adding component pipelines.  This will
    register components mentioned in :obj:`components` when the resepctive
    module is loaded.

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
        """Initialize the component and add it to the NLP pipe line.  This base
        class implementation loads the :obj:`module`, then calls
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

    @staticmethod
    def default_instance() -> FeatureDocumentParser:
        """Create the parser as configured in the resource library of the
        package.

        """
        config: str = (
            '[import]\n' +
            'config_file = resource(zensols.nlp): resources/obj.conf')
        factory = ImportConfigFactory(ImportIniConfig(StringIO(config)))
        return factory('doc_parser')

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

        def __str__(self):
            return f'model_name: {self.model_name}, lang: {self.lang}'

        def __repr__(self):
            return self.__str__()


class FeatureTokenDecorator(ABC):
    """Implementations can add, remove or modify features on a token.

    """
    @abstractmethod
    def decorate(self, token: FeatureToken):
        pass


class FeatureSentenceDecorator(ABC):
    """Implementations can add, remove or modify features on a sentence.

    """
    @abstractmethod
    def decorate(self, sent: FeatureSentence):
        pass


class FeatureDocumentDecorator(ABC):
    """Implementations can add, remove or modify features on a document.

    """
    @abstractmethod
    def decorate(self, doc: FeatureDocument):
        pass


@dataclass
class DecoratedFeatureDocumentParser(FeatureDocumentParser):
    """This class adapts the :class:`.FeatureDocumentParser` adaptors to the
    general case using a GoF decorator pattern.  This is useful for any post
    processing needed on existing configured document parsers.

    """
    delegate: FeatureDocumentParser = field()
    """Used to create the feature documents."""

    token_decorators: Sequence[FeatureTokenDecorator] = field(default=())
    """A list of decorators that can add, remove or modify features on a token.

    """
    sentence_decorators: Sequence[FeatureSentenceDecorator] = field(
        default=())
    """A list of decorators that can add, remove or modify features on a
    sentence.

    """
    document_decorators: Sequence[FeatureDocumentDecorator] = field(
        default=())
    """A list of decorators that can add, remove or modify features on a
    document.

    """
    def decorate(self, doc: FeatureDocument):
        td: FeatureTokenDecorator
        for td in self.token_decorators:
            tok: FeatureToken
            for tok in doc.token_iter():
                td.decorate(tok)
        sd: FeatureSentenceDecorator
        for sd in self.sentence_decorators:
            sent: FeatureSentence
            for sent in doc.sents:
                sd.decorate(sent)
        dd: FeatureDocumentDecorator
        for dd in self.document_decorators:
            dd.decorate(doc)

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        doc: FeatureDocument = self.delegate.parse(text, *args, **kwargs)
        self.decorate(doc)
        return doc


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
            if doc.text != text:
                raise NLPError(
                    f'Document text does not match: <{text}> != >{doc.text}>')
            loaded = True
        return doc, key, loaded

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        return self._load_or_parse(text, True, *args, **kwargs)[0]

    def clear(self):
        """Clear the caching stash."""
        if self.stash is not None:
            self.stash.clear()


@dataclass
class FeatureSentenceFactory(object):
    """Create a :class:`.FeatureSentence` out of single tokens or split on
    whitespace.  This is a utility class to create data structures when only
    single tokens are the source data.

    For example, if you only have tokens that need to be scored with Unigram
    Rouge-1, use this class to create sentences, which is a subclass of
    :class:`.TokenContainer`.

    """
    token_decorators: Sequence[FeatureTokenDecorator] = field(default=())
    """A list of decorators that can add, remove or modify features on a token.

    """
    def _decorate_token(self, feature_token: FeatureToken):
        decorator: FeatureTokenDecorator
        for decorator in self.token_decorators:
            decorator.decorate(feature_token)

    def create(self, tokens: Union[str, Iterable[str]]) -> FeatureSentence:
        """Create a sentence from tokens.

        :param tokens: if a string, then split on white space

        """
        toks: List[FeatureToken] = []
        slen: int = 0
        has_tok_decorators: bool = len(self.token_decorators) > 0
        tokens = tokens.split() if isinstance(tokens, str) else tokens
        tok: str
        for i, tok in enumerate(tokens):
            ftok = FeatureToken(i=i, idx=slen, i_sent=0, norm=tok)
            if has_tok_decorators:
                self._decorate_token(ftok)
            toks.append(ftok)
            slen += len(tok) + 1
        return FeatureSentence(tokens=tuple(toks))

    def __call__(self, tokens: Union[str, Iterable[str]]) -> FeatureSentence:
        """See :meth:`.create`."""
        return self.create(tokens)
