"""Stem text using the Porter stemmer.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from nltk.stem import PorterStemmer
from zensols.nlp import TokenMapper

logger = logging.getLogger(__name__)


@dataclass
class PorterStemmerTokenMapper(TokenMapper):
    """Use the Porter Stemmer from the NTLK to stem as normalized tokens.

    """
    stemmer: PorterStemmer = field(default_factory=PorterStemmer)

    def map_tokens(self, token_tups):
        return (map(lambda t: (t[0], self.stemmer.stem(t[1])),
                    token_tups),)
