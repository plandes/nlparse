"""Contains useful classes for decorating feature sentences.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple
from dataclasses import dataclass
import re
from spacy.tokens import Span
from . import (
    LexicalSpan, FeatureToken, FeatureSentence, SpacyFeatureSentenceDecorator
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
