"""Contains useful classes for decorating feature sentences.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple
from dataclasses import dataclass, field
import re
from spacy.tokens import Span, Doc
from . import (
    LexicalSpan, FeatureToken, FeatureSentence, FeatureDocument,
    SpacyFeatureSentenceDecorator, SpacyFeatureDocumentDecorator
)


@dataclass
class SplitTokenSentenceDecorator(SpacyFeatureSentenceDecorator):
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

    def decorate(self, spacy_sent: Span, feature_sent: FeatureSentence):
        split_toks: List[FeatureToken] = []
        tok: FeatureToken
        for ftok in feature_sent.token_iter():
            tnorms: Tuple[str, ...] = tuple(re.finditer(r'\S+', ftok.norm))
            if len(tnorms) == 1:
                split_toks.append(ftok)
            else:
                split_toks.extend(self._split_tok(ftok, tnorms))
        feature_sent.tokens = tuple(split_toks)


@dataclass
class StripSpacyFeatureSentenceDecorator(SpacyFeatureSentenceDecorator):
    """A decorator that strips whitespace from sentences.

    :see: :meth:`.TokenContainer.strip`

    """
    def decorate(self, spacy_sent: Span, feature_sent: FeatureSentence):
        feature_sent.strip()


@dataclass
class FilterSentenceFeatureDocumentDecorator(SpacyFeatureDocumentDecorator):
    """Filter zero length sentences.

    """
    filter_space: bool = field(default=True)
    """Whether to filter space tokens when comparing zero length sentences."""

    def _filter_empty_sentences(self, fsent: FeatureSentence) -> bool:
        toks: Tuple[FeatureToken] = fsent.tokens
        if self.filter_space:
            toks = tuple(filter(lambda t: not t.is_space, fsent.token_iter()))
        return len(toks) > 0

    def decorate(self, spacy_doc: Doc, feature_doc: FeatureDocument):
        olen: int = len(feature_doc)
        fsents: Tuple[FeatureSentence] = tuple(filter(
            self._filter_empty_sentences, feature_doc.sents))
        nlen: int = len(fsents)
        if olen != nlen:
            feature_doc.sents = fsents


@dataclass
class UpdateFeatureDocumentDecorator(SpacyFeatureDocumentDecorator):
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
    def decorate(self, spacy_doc: Doc, feature_doc: FeatureDocument):
        if self.update_indexes:
            feature_doc.update_indexes()
        if self.update_entity_spans:
            feature_doc.update_entity_spans()
