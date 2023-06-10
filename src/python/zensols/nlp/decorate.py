"""Contains useful classes for decorating feature sentences.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple
from dataclasses import dataclass, field
import re
from . import (
    LexicalSpan, FeatureToken, FeatureSentence, FeatureDocument,
    FeatureSentenceDecorator, FeatureDocumentDecorator
)


@dataclass
class SplitTokenSentenceDecorator(FeatureSentenceDecorator):
    """A decorator that splits feature tokens by white space.

    """
    def _split_tok(self, ftok: FeatureToken, matches: Tuple[re.Match]):
        toks: List[FeatureToken] = []
        norm: str
        for match in matches:
            ctok: FeatureToken = ftok.clone()
            ctok.norm = match.group(0)
            ctok.lexspan = LexicalSpan(ftok.lexspan.begin + match.start(0),
                                       ftok.lexspan.begin + match.end(0))
            ctok.idx = ctok.lexspan.begin
            toks.append(ctok)
        return toks

    def decorate(self, sent: FeatureSentence):
        split_toks: List[FeatureToken] = []
        tok: FeatureToken
        for ftok in sent.token_iter():
            tnorms: Tuple[str, ...] = tuple(re.finditer(r'\S+', ftok.norm))
            if len(tnorms) == 1:
                split_toks.append(ftok)
            else:
                split_toks.extend(self._split_tok(ftok, tnorms))
        if sent.token_len != len(split_toks):
            sent.tokens = tuple(split_toks)


@dataclass
class StripSentenceDecorator(FeatureSentenceDecorator):
    """A decorator that strips whitespace from sentences.

    :see: :meth:`.TokenContainer.strip`

    """
    def decorate(self, sent: FeatureSentence):
        sent.strip()


@dataclass
class FilterTokenSentenceDecorator(FeatureSentenceDecorator):
    """A decorator that strips whitespace from sentences.

    :see: :meth:`.TokenContainer.strip`

    """
    remove_stop: bool = field(default=False)
    remove_space: bool = field(default=False)
    remove_pronouns: bool = field(default=False)
    remove_punctuation: bool = field(default=False)
    remove_determiners: bool = field(default=False)
    remove_empty: bool = field(default=False)

    def decorate(self, sent: FeatureSentence):
        def filter_tok(t: FeatureToken) -> bool:
            return \
                (not self.remove_stop or not t.is_stop) and \
                (not self.remove_space or not t.is_space) and \
                (not self.remove_pronouns or not t.pos_ == 'PRON') and \
                (not self.remove_punctuation or not t.is_punctuation) and \
                (not self.remove_determiners or not t.tag_ == 'DT') and \
                (not self.remove_empty or len(t.norm) > 0)
        toks: Tuple[FeatureToken] = tuple(filter(filter_tok, sent))
        if sent.token_len != len(toks):
            sent.tokens = toks


@dataclass
class FilterEmptySentenceDocumentDecorator(FeatureDocumentDecorator):
    """Filter zero length sentences.

    """
    filter_space: bool = field(default=True)
    """Whether to filter space tokens when comparing zero length sentences."""

    def _filter_empty_sentences(self, fsent: FeatureSentence) -> bool:
        toks: Tuple[FeatureToken] = fsent.tokens
        if self.filter_space:
            toks = tuple(filter(lambda t: not t.is_space, fsent.token_iter()))
        return len(toks) > 0

    def decorate(self, doc: FeatureDocument):
        olen: int = len(doc)
        fsents: Tuple[FeatureSentence] = tuple(filter(
            self._filter_empty_sentences, doc.sents))
        nlen: int = len(fsents)
        if olen != nlen:
            doc.sents = fsents


@dataclass
class UpdateDocumentDecorator(FeatureDocumentDecorator):
    """Updates document indexes and spans (see fields).

    """
    update_indexes: bool = field(default=True)
    """Whether to update the document indexes with
    :meth:`.FeatureDocument.update_indexes`.

    """
    update_entity_spans: bool = field(default=True)
    """Whether to update the document indexes with
    :meth:`.FeatureDocument.update_entity_spans`.

    """
    def decorate(self, doc: FeatureDocument):
        if self.update_indexes:
            doc.update_indexes()
        if self.update_entity_spans:
            doc.update_entity_spans()
