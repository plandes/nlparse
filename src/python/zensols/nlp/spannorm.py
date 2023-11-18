"""Normalize spans (of tokens) into strings by reconstructing based on language
rules from the normalized form of the tokens.  This is needed after any token
manipulation from :class:`.TokenNormalizer` or other changes to
:obj:`.FeatureToken.norm`.

For now, only English is supported, but the module is provided for other
languages and future enhancements of normalization configuration.

"""
__author__ = 'Paul Landes'

from typing import Set, Iterable, Tuple
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from io import StringIO
from . import ParseError, FeatureToken


class SpanNormalizer(metaclass=ABCMeta):
    """Subclasses normalize feature tokens on a per :class:`spacy.Language`.
    All subclasses must be re-entrant.

    """
    @abstractmethod
    def get_norm(self) -> str:
        """Create a string that follows the langauge spacing rules."""
        pass

    @abstractmethod
    def get_canonical(self, tokens: Iterable[FeatureToken]) -> str:
        """A canonical representation of the container, which are non-space
        tokens separated by :obj:`CANONICAL_DELIMITER`.

        """


@dataclass(frozen=True)
class EnglishSpanNormalizer(SpanNormalizer):
    """An implementation of a span normalizer for the Enlish language.

    """
    post_space_skip: Set[str] = field(default=frozenset("""`‘“[({<-"""))
    """Characters after which no space is added for span normalization."""

    pre_space_skip: Set[str] = field(default=frozenset(
        "'s n't 'll 'm 've 'd 're -".split()))
    """Characters before whcih no space is added for span normalization."""

    keep_space_skip: Set[str] = field(default=frozenset("""_"""))
    """Characters that retain space on both sides."""

    canonical_delimiter: str = field(default='|')
    """The token delimiter used in :obj:`canonical`."""

    def __post_init__(self):
        # bypass frozen setattr guards
        self.__dict__['_longest_pre_space_skip'] = \
            max(map(len, self.pre_space_skip))

    def get_norm(self, tokens: Iterable[FeatureToken]) -> str:
        nsent: str
        toks: Tuple[FeatureToken] = tuple(
            filter(lambda t: t.text != '\n', tokens))
        tlen: int = len(toks)
        has_punc = tlen > 0 and hasattr(toks[0], 'is_punctuation')
        if has_punc:
            post_space_skip: Set[str] = self.post_space_skip
            pre_space_skip: Set[str] = self.pre_space_skip
            keep_space_skip: Set[str] = self.keep_space_skip
            n_pre_space_skip: int = self._longest_pre_space_skip
            sio = StringIO()
            last_avoid = False
            tix: int
            tok: FeatureToken
            for tix, tok in enumerate(toks):
                norm: str = tok.norm
                if norm is None:
                    raise ParseError(f'Token {tok.text} has no norm')
                if tix > 0 and tix < tlen:
                    nlen: int = len(norm)
                    if nlen == 1 and norm in keep_space_skip:
                        sio.write(' ')
                    else:
                        do_post_space_skip: bool = False
                        if nlen == 1:
                            do_post_space_skip = norm in post_space_skip
                        if (not tok.is_punctuation or do_post_space_skip) and \
                           not last_avoid and \
                           not (nlen <= n_pre_space_skip and
                                norm in pre_space_skip):
                            sio.write(' ')
                        last_avoid = do_post_space_skip or tok.norm == '--'
                sio.write(norm)
            nsent = sio.getvalue()
        else:
            nsent = ' '.join(map(lambda t: t.norm, toks))
        return nsent.strip()

    def get_canonical(self, tokens: Iterable[FeatureToken]) -> str:
        return self.canonical_delimiter.join(
            map(lambda t: t.text,
                filter(lambda t: not t.is_space, tokens)))

    def __getstate__(self):
        raise RuntimeError(f'Instances of {type(self)} are not picklable')


DEFAULT_FEATURE_TOKEN_NORMALIZER = EnglishSpanNormalizer()
