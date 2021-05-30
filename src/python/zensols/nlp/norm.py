"""Normalize text and map Spacy documents.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from typing import List, Iterable, Tuple
from abc import abstractmethod, ABC
import logging
import re
import itertools as it
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from zensols.util import APIError
from zensols.config import Configurable, ImportConfigFactory

logger = logging.getLogger(__name__)


class ParseError(APIError):
    """Raised for any parsing errors for this API."""
    pass


@dataclass
class TokenNormalizer(object):
    """Base token extractor returns tuples of tokens and their normalized version.

    Configuration example::

        [default_token_normalizer]
        class_name = zensols.nlp.TokenNormalizer
        embed_entities = False

    """

    embed_entities: bool = field(default=True)
    """Whether or not to replace tokens with their respective named entity
    version.

    """

    def __embed_entities(self, doc: Doc):
        """For each token, return the named entity form if it exists.

        :param doc: the spacy document to iterate over

        """
        tlen = len(doc)
        ents = {}
        for ent in doc.ents:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'adding entity start: {ent.start} -> {ent}')
            ents[ent.start] = ent
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'entities: {ents}')
        i = 0
        while i < tlen:
            if i in ents:
                ent = ents[i]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'adding entity: {ent}')
                yield ent
                i = ent.end
            else:
                tok = doc[i]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'adding token: {tok}')
                yield tok
                i += 1

    def _to_token_tuple(self, doc: Doc) -> Iterable[Tuple[Token, str]]:
        "Normalize the document in to (token, normal text) tuples."
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding entities: {self.embed_entities}')
        if self.embed_entities:
            toks = self.__embed_entities(doc)
        else:
            toks = doc
        toks = map(lambda t: (t, t.orth_,), toks)
        return toks

    def _map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        """Map token tuples in sub classes.

        :param token_tups: tuples generated from ``_to_token_tuple``
        """
        return None

    def normalize(self, doc) -> Iterable[Tuple[Token, str]]:
        """Normalize Spacey document ``doc`` in to (token, normal text) tuples.
        """
        tlist = self._to_token_tuple(doc)
        maps = self._map_tokens(tlist)
        if maps is not None:
            tlist = tuple(maps)
        return iter(tlist)

    def __str__(self):
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = type(self).__name__
        return f'{name}: embed={self.embed_entities}'

    def __repr__(self):
        return self.__str__()


@dataclass
class TokenMapper(ABC):
    """Abstract class used to transform token tuples generated from
    :meth:`.TokenNormalizer.normalize`.

    """
    @abstractmethod
    def map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        """Transform token tuples.

        """
        pass


@dataclass
class SplitTokenMapper(TokenMapper):
    """Splits the normalized text on a per token basis with a regular expression.

    Configuration example::

        [lemma_token_mapper]
        class_name = zensols.nlp.SplitTokenMapper
        regex = r'[ \\t]'

    """
    regex: str = field(default='')

    def __post_init__(self):
        self.regex = re.compile(eval(self.regex))

    def map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        rg = self.regex
        return map(lambda t: map(lambda s: (t[0], s), re.split(rg, t[1])),
                   token_tups)


@dataclass
class LemmatizeTokenMapper(TokenMapper):
    """Lemmatize tokens and optional remove entity stop words.

    **Important:** This completely ignores the normalized input token string
    and essentially just replaces it with the lemma found in the token
    instance.

    Configuration example::

        [lemma_token_mapper]
        class_name = zensols.nlp.LemmatizeTokenMapper

    :param lemmatize: lemmatize if ``True``; this is an option to allow (only)
                      the removal of the first top word in named entities

    :param remove_first_stop: whether to remove the first top word in named
                              entities when ``embed_entities`` is ``True``

    """
    lemmatize: bool = field(default=True)
    remove_first_stop: bool = field(default=False)

    def _lemmatize(self, tok_or_ent):
        if isinstance(tok_or_ent, Token):
            stok = tok_or_ent.lemma_
        else:
            if self.remove_first_stop and tok_or_ent[0].is_stop:
                tok_or_ent = tok_or_ent[1:]
            stok = tok_or_ent.text.lower()
        return stok

    def map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        return (map(lambda x: (x[0], self._lemmatize(x[0])), token_tups),)


@dataclass
class FilterTokenMapper(TokenMapper):
    """Filter tokens based on token (Spacy) attributes.

    Configuration example::

        [filter_token_mapper]
        class_name = zensols.nlp.FilterTokenMapper
        remove_stop = True
        remove_punctuation = True

    """
    remove_stop: bool = field(default=False)
    remove_space: bool = field(default=False)
    remove_pronouns: bool = field(default=False)
    remove_punctuation: bool = field(default=False)
    remove_determiners: bool = field(default=False)

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created {self.__class__}: ' +
                         f'remove_stop: {self.remove_stop}, ' +
                         f'remove_space: {self.remove_space}, ' +
                         f'remove_pronouns: {self.remove_pronouns}, ' +
                         f'remove_punctuation: {self.remove_punctuation}, ' +
                         f'remove_determiners: {self.remove_determiners}')

    def _filter(self, tok_or_ent_tup):
        tok_or_ent = tok_or_ent_tup[0]
        keep = False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'filter: {tok_or_ent} ({type(tok_or_ent)})')
        if isinstance(tok_or_ent, Token):
            t = tok_or_ent
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'token {t}: l={len(t)}, ' +
                             f's={t.is_stop}, p={t.is_punct}')
            if (not self.remove_stop or not t.is_stop) and \
               (not self.remove_space or not t.is_space) and \
               (not self.remove_pronouns or not t.pos_ == 'PRON') and \
               (not self.remove_punctuation or not t.is_punct) and \
               (not self.remove_determiners or not t.tag_ == 'DT') and \
               len(t) > 0:
                keep = True
        else:
            keep = True
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'filter: keeping={keep}')
        return keep

    def map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('filter mapper: map_tokens')
        return (filter(self._filter, token_tups),)


@dataclass
class SubstituteTokenMapper(TokenMapper):
    """Replace a regular expression in normalized token text.

    Configuration example::

        [subs_token_mapper]
        class_name = zensols.nlp.SubstituteTokenMapper
        regex = r'[ \\t]'
        replace_char = _

    """
    regex: str = field(default='')
    replace_char: str = field(default='')

    def __post_init__(self):
        self.regex = re.compile(eval(self.regex))

    def map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        return (map(lambda x: (x[0], re.sub(
            self.regex, self.replace_char, x[1])),
                    token_tups),)


@dataclass
class LambdaTokenMapper(TokenMapper):
    """Use a lambda expression to map a token tuple.

    This is handy for specialized behavior that can be added directly to a
    configuration file.

    Configuration example::

        [lc_lambda_token_mapper]
        class_name = zensols.nlp.LambdaTokenMapper
        map_lambda = lambda x: (x[0], f'<{x[1].lower()}>')

    """
    add_lambda: str = field(default=None)
    map_lambda: str = field(default=None)

    def __post_init__(self):
        if self.add_lambda is None:
            self.add_lambda = lambda x: ()
        else:
            self.add_lambda = eval(self.add_lambda)
        if self.map_lambda is None:
            self.map_lambda = lambda x: x
        else:
            self.map_lambda = eval(self.map_lambda)

    def map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        return (map(self.map_lambda, token_tups),)


@dataclass
class MapTokenNormalizer(TokenNormalizer):
    """A normalizer that applies a sequence of :class:`.TokenMapper` instances to
    transform the normalized token text.

    Configuration example::

        [map_filter_token_normalizer]
        class_name = zensols.nlp.MapTokenNormalizer
        mapper_class_list = eval: 'filter_token_mapper'.split()

    """
    config: Configurable = field(default=None)
    """The application context."""

    mapper_class_list: List[str] = field(default_factory=list)
    """The configuration names to create with ``ImportConfigFactory``."""

    reload: bool = field(default=False)
    """Whether or not to reload the module when creating the instance, which is
    useful while prototyping.

    """

    def __post_init__(self):
        ta = ImportConfigFactory(self.config, reload=self.reload)
        self.mappers = tuple(map(ta.instance, self.mapper_class_list))

    def _map_tokens(self, token_tups: Iterable[Tuple[Token, str]]) -> \
            Iterable[Tuple[Token, str]]:
        for mapper in self.mappers:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'mapping token_tups with {mapper}')
            token_tups = it.chain(*mapper.map_tokens(token_tups))
        return token_tups
