"""Stem text using the Porter stemmer.

"""
__author__ = 'Paul Landes'

import logging
from nltk.stem import PorterStemmer
from zensols.nlp import TokenMapper, TokenMapperFactory

logger = logging.getLogger(__name__)


class PorterStemmerTokenMapper(TokenMapper):
    """Use the Porter Stemmer from the NTLK to stem as normalized tokens.

    """
    def __init__(self, *args, **kwargs):
        super(PorterStemmerTokenMapper, self).__init__(*args, **kwargs)
        self.stemmer = PorterStemmer()

    def map_tokens(self, token_tups):
        return (map(lambda t: (t[0], self.stemmer.stem(t[1])),
                    token_tups),)


TokenMapperFactory.register(PorterStemmerTokenMapper)
