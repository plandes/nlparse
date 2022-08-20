"""Feature document persistence.

"""
__author__ = 'Paul Landes'

from typing import Iterable
from dataclasses import dataclass, field
from zensols.util import Hasher
from zensols.persist import ReadOnlyStash, Stash
from . import FeatureDocument, FeatureDocumentParser


@dataclass
class FeatureDocumentStash(ReadOnlyStash):
    """Persists documents using a delegate stash with keys as the natural
    language text and :class:`.FeatureDocument` instances parsed by
    :obj:`.doc_parser` as the values.  The :obj:`delegate` uses the hashes of
    the text as keys.

    """
    delegate: Stash = field()
    """The backing stash that persists the feature document instances."""

    doc_parser: FeatureDocumentParser = field()
    """The parser used for cache misses."""

    hasher: Hasher = field(default_factory=Hasher)
    """Used to hash the natural langauge text in to string keys."""

    def _hash_text(self, text: str) -> str:
        self.hasher.reset()
        self.hasher.update(text)
        return self.hasher()

    def load(self, text: str) -> FeatureDocument:
        key: str = self._hash_text(text)
        item: FeatureDocument = self.delegate.load(key)
        if item is None:
            item: FeatureDocument = self.doc_parser(text)
            self.delegate.dump(key, item)
        return item

    def keys(self) -> Iterable[str]:
        return self.delegate.keys()

    def exists(self, text: str) -> bool:
        key: str = self._hash_text(text)
        return self.delegate.exists(key)
