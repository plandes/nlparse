"""Normalize text and munge Spacy documents.

"""
__author__ = 'Paul Landes'

import logging
import re
import itertools as it
from abc import abstractmethod
from spacy.tokens.token import Token
from zensols.actioncli import ConfigFactory

logger = logging.getLogger(__name__)


class TokenNormalizer(object):
    """Base token extractor returns tuples of tokens and their normalized version.

    """
    def __init__(self, normalize=True, embed_entities=True,
                 remove_first_stop=False, limit=None):
        """Initialize the normalizer.

        :param normalize: whether or not to normalize the text (useful since
                          this class has other functionality.
        :param embed_entities: whether or not to replace tokens with their
                          respective named entity version
        :param remove_first_stop: whether to remove the first top word in named
                                  entities when ``embed_entities`` is ``True``

        """
        logger.debug(f'init embedding entities: {embed_entities}')
        self._normalize = normalize
        self.embed_entities = embed_entities
        self.remove_first_stop = remove_first_stop
        self.limit = limit

    def __embed_entities(self, doc):
        """For each token, return the named entity form if it exists.

        :param doc: the spacy document to iterate over

        """
        tlen = len(doc)
        ents = {}
        for ent in doc.ents:
            logger.debug(f'adding entity start: {ent.start} -> {ent}')
            ents[ent.start] = ent
        logger.debug(f'entities: {ents}')
        i = 0
        while i < tlen:
            if i in ents:
                ent = ents[i]
                logger.debug(f'adding entity: {ent}')
                yield ent
                i = ent.end
            else:
                tok = doc[i]
                logger.debug(f'adding token: {tok}')
                yield tok
                i += 1

    def __norm_to_tok_tups(self, doc):
        "Normalize the document in to (token, normal text) tuples."
        def norm(tok_or_ent):
            if isinstance(tok_or_ent, Token):
                stok = tok_or_ent.lemma_
            else:
                if self.remove_first_stop and tok_or_ent[0].is_stop:
                    tok_or_ent = tok_or_ent[1:]
                stok = tok_or_ent.text.lower()
            return (tok_or_ent, stok)

        logger.debug(f'embedding entities: {self.embed_entities}')
        if self.embed_entities:
            toks = self.__embed_entities(doc)
        else:
            toks = doc

        if self._normalize:
            toks = map(norm, toks)
        else:
            toks = map(lambda t: (t, t.orth_,), toks)
        return toks

    def _map_tokens(self, token_tups):
        """Map token tuples in sub classes.

        :param token_tups: tuples generated from ``__norm_to_tok_tups``
        """
        return None

    def normalize(self, doc):
        """Normalize Spacey document ``doc`` in to (token, normal text) tuples.
        """
        tlist = self.__norm_to_tok_tups(doc)
        maps = self._map_tokens(tlist)
        if maps is not None:
            tlist = tuple(maps)
        return iter(tlist)

    def __str__(self):
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = type(self).__name__
        return (f'{name}: embed={self.embed_entities}, ' +
                f'normalize: {self._normalize} ' +
                f'remove first stop: {self.remove_first_stop}')

    def __repr__(self):
        return self.__str__()


class TokenNormalizerFactory(ConfigFactory):
    INSTANCE_CLASSES = {}

    def __init__(self, config):
        super(TokenNormalizerFactory, self).__init__(
            config, '{name}_token_normalizer')


TokenNormalizerFactory.register(TokenNormalizer)


class TokenMapper(object):
    """Abstract class used to transform token tuples generated from
    ``TokenNormalizer.normalize``.

    """
    @abstractmethod
    def map_tokens(self, token_tups):
        """Transform token tuples.

        """
        pass


class TokenMapperFactory(ConfigFactory):
    INSTANCE_CLASSES = {}

    def __init__(self, config):
        super(TokenMapperFactory, self).__init__(
            config, '{name}_token_munger')


class SplitTokenMapper(TokenMapper):
    """Splits the normalized text on a per token basis with a regular expression.

    """
    def __init__(self, regex, *args, **kwargs):
        super(SplitTokenMapper, self).__init__(*args, **kwargs)
        self.regex = re.compile(eval(regex))

    def map_tokens(self, token_tups):
        rg = self.regex
        return map(lambda t: map(lambda s: (t[0], s), re.split(rg, t[1])),
                   token_tups)


TokenMapperFactory.register(SplitTokenMapper)


class FilterTokenMapper(TokenMapper):
    """Filter tokens based on token (Spacy) attributes.

    """
    def __init__(self, *args, remove_stop=False, remove_pronouns=False,
                 remove_punctuation=True, remove_determiners=True, **kwargs):
        super(FilterTokenMapper, self).__init__(*args, **kwargs)
        self.remove_stop = remove_stop
        self.remove_pronouns = remove_pronouns
        self.remove_punctuation = remove_punctuation
        self.remove_determiners = remove_determiners

    def _filter(self, tok_or_ent_tup):
        tok_or_ent = tok_or_ent_tup[0]
        logger.debug(f'{tok_or_ent} ({type(tok_or_ent)})')
        keep = False
        if isinstance(tok_or_ent, Token):
            t = tok_or_ent
            logger.debug(f'token {t}: l={len(t)}, s={t.is_stop}, p={t.is_punct}')
            if (not self.remove_stop or not t.is_stop) and \
               (not self.remove_pronouns or not t.lemma_ == '-PRON-') and \
               (not self.remove_punctuation or not t.is_punct) and \
               (not self.remove_determiners or not t.tag_ == 'DT') and \
               len(t) > 0:
                keep = True
        else:
            keep = True
        return keep

    def map_tokens(self, token_tups):
        return (filter(self._filter, token_tups),)


TokenMapperFactory.register(FilterTokenMapper)


class SubstituteTokenMapper(TokenMapper):
    """Replace a string in normalized token text.

    """
    def __init__(self, regex, replace_char, *args, **kwargs):
        super(SubstituteTokenMapper, self).__init__(*args, **kwargs)
        self.regex = re.compile(eval(regex))
        self.replace_char = replace_char

    def map_tokens(self, token_tups):
        return (map(lambda x: (x[0], re.sub(self.regex, self.replace_char, x[1])),
                    token_tups),)


TokenMapperFactory.register(SubstituteTokenMapper)


class LambdaTokenMapper(TokenMapper):
    """Use a lambda expression to map a token tuple.

    This is handy for specialized behavior that can be added directly to a
    configuration file.

    """
    def __init__(self, add_lambda=None, map_lambda=None,
                 *args, **kwargs):
        super(LambdaTokenMapper, self).__init__(*args, **kwargs)
        if add_lambda is None:
            self.add_lambda = lambda x: ()
        else:
            self.add_lambda = eval(add_lambda)
        if map_lambda is None:
            self.map_lambda = lambda x: x
        else:
            self.map_lambda = eval(map_lambda)

    def map_tokens(self, terms):
        return (map(self.map_lambda, terms),)


TokenMapperFactory.register(LambdaTokenMapper)


class MapTokenNormalizer(TokenNormalizer):
    """A normalizer that applies a sequence of ``TokenMappers`` to transform
    the normalized token text.

    """

    def __init__(self, config, munger_class_list, *args, **kwargs):
        super(MapTokenNormalizer, self).__init__(*args, **kwargs)
        te = TokenNormalizerFactory(config)
        ta = TokenMapperFactory(config)
        self.mungers = tuple(map(ta.instance, munger_class_list))

    def _map_tokens(self, token_tups):
        for munger in self.mungers:
            logger.debug(f'munging token_tups with {munger}')
            token_tups = it.chain(*munger.map_tokens(token_tups))
        return token_tups


TokenNormalizerFactory.register(MapTokenNormalizer)
