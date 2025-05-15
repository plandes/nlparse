"""Serializes :class:`.FeatureToken` and :class:`.TokenContainer` instances
using the :class:`~zensols.config.dictable.Dictable` interface.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Dict, List, Set, Union
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from collections import OrderedDict
from zensols.config import Dictable
from . import (
    NLPError, FeatureToken, TokenContainer,
    FeatureSpan, FeatureSentence, FeatureDocument
)


class Include(Enum):
    """Indicates what to include at each level.

    """
    original = auto()
    """The original text."""

    normal = auto()
    """The normalized form of the text."""

    tokens = auto()
    """The tokens of the :class:`.TokenContainer`."""

    sentences = auto()
    """The sentences of the :class:`.FeatureDocument`."""


@dataclass
class Serialized(Dictable, metaclass=ABCMeta):
    """A base strategy class that can serialize :class:`.TokenContainer`
    instances.

    """
    container: TokenContainer = field()
    """The container to be serialized."""

    includes: Set[Include] = field()
    """The things to be included at the level of the subclass serializer."""

    feature_ids: Tuple[str, ...] = field()
    """The feature IDs used when serializing tokens."""

    @abstractmethod
    def _serialize(self) -> Dict[str, Any]:
        """Implemented to serialize :obj:`container` in to a dictionary."""
        pass

    def _from_dictable(self, recurse: bool, readable: bool,
                       class_name_param: str = None) -> Dict[str, Any]:
        return self._serialize()


@dataclass
class SerializedTokenContainer(Serialized):
    """Serializes instance of :class:`.TokenContainer`.  This is used to
    serialize spans and sentences.

    """
    def _feature_tokens(self, container: TokenContainer) -> \
            List[Dict[str, Any]]:
        """Serialize tokens of ``container`` in to a list of dictionary
        features.

        """
        tok_feats: List[Dict[str, Any]] = []
        tok: FeatureToken
        for tok in container.token_iter():
            tfs: Dict[str, Any] = tok.get_features(self.feature_ids)
            if len(tfs) > 0:
                tok_feats.append(tfs)
        return tok_feats

    def _serialize(self) -> Dict[str, Any]:
        dct = OrderedDict()
        if Include.original in self.includes:
            dct[Include.original.name] = self.container.text
        if Include.normal in self.includes:
            dct[Include.normal.name] = self.container.norm
        if Include.tokens in self.includes:
            dct[Include.tokens.name] = self._feature_tokens(self.container)
        return dct


@dataclass
class SerializedFeatureDocument(Serialized):
    """A serializer for feature documents.  The :obj:`container` has to be an
    instance of a :class:`.FeatureDocument`.

    """
    sentence_includes: Set[Include] = field()
    """The list of things to include in the sentences of the document."""

    def _serialize(self) -> Dict[str, Any]:
        doc = SerializedTokenContainer(
            container=self.container,
            includes=self.includes,
            feature_ids=self.feature_ids)
        dct = OrderedDict(doc=doc.asdict())
        if Include.sentences in self.includes:
            sents: List[Dict[str, Any]] = []
            dct[Include.sentences.name] = sents
            sent: FeatureSentence
            for sent in self.container.sents:
                ser = SerializedTokenContainer(
                    container=sent,
                    includes=self.sentence_includes,
                    feature_ids=self.feature_ids)
                sents.append(ser.asdict())
        return dct


@dataclass
class SerializedTokenContainerFactory(Dictable):
    """Creates instances of :class:`.Serialized` from instances of
    :class:`.TokenContainer`.  These can then be used as
    :class:`~zensols.config.dictable.Dictable` instances, specifically with the
    ``asdict`` and ``asjson`` methods.

    """
    sentence_includes: Set[Union[Include, str]] = field()
    """The things to be included in sentences."""

    document_includes: Set[Union[Include, str]] = field()
    """The things to be included in documents."""

    feature_ids: Tuple[str, ...] = field(default=None)
    """The feature IDs used when serializing tokens."""

    def __post_init__(self):
        def map_thing(x):
            if isinstance(x, str):
                x = Include.__members__[x]
            return x

        # convert strings to enums for easy app configuration
        for ai in 'sentence document'.split():
            attr = f'{ai}_includes'
            val = set(map(map_thing, getattr(self, attr)))
            setattr(self, attr, val)

    def create(self, container: TokenContainer) -> Serialized:
        """Create a serializer from ``container`` (see class docs).

        :param container: he container to be serialized

        :return: an object that can be serialized using ``asdict`` and
                 ``asjson`` method.

        """
        serialized: Serialized
        if isinstance(container, FeatureDocument):
            serialized = SerializedFeatureDocument(
                container=container,
                includes=self.document_includes,
                sentence_includes=self.sentence_includes,
                feature_ids=self.feature_ids)
        elif isinstance(container, FeatureSpan):
            serialized = SerializedFeatureDocument(
                conatiner=container,
                includes=self.sentence_includes)
        else:
            raise NLPError(f'No serialization method for {type(container)}')
        return serialized

    def __call__(self, container: TokenContainer) -> Serialized:
        """See :meth:`create`."""
        return self.create(container)
