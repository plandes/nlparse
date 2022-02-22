from __future__ import annotations
"""Interfaces, contracts and errors.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Union
from abc import ABCMeta
import sys
from io import TextIOBase
from spacy.tokens import Token
from spacy.tokens import Span
from spacy.tokens import Doc
from zensols.util import APIError
from zensols.config import Dictable


class NLPError(APIError):
    """Raised for any errors for this library."""
    pass


class ParseError(APIError):
    """Raised for any parsing errors."""
    pass


class LexicalSpan(Dictable):
    """A lexical character span of text in a document.

    """
    _DICTABLE_ATTRIBUTES = {'begin', 'end'}

    def __init__(self, begin: int, end: int):
        """Initialize the interval.

        :param begin: the begin of the span

        :param end: the end of the span

        """
        self.begin = begin
        self.end = end

    @property
    def astuple(self) -> Tuple[int, int]:
        """The span as a ``(begin, end)`` tuple."""
        return (self.begin, self.end)

    @classmethod
    def from_token(cls, tok: Union[Token, Span]) -> Tuple[int, int]:
        """Create a span from a spaCy :class:`~spacy.tokens.Token` or
        :class:`~spacy.tokens.Span`.

        """
        if isinstance(tok, Span):
            doc: Doc = tok.doc
            etok = doc[tok.end - 1]
            start = doc[tok.start].idx
            end = etok.idx + len(etok.orth_)
        else:
            start = tok.idx
            end = tok.idx + len(tok.orth_)
        return cls(start, end)

    @staticmethod
    def overlaps(a0: int, a1: int, b0: int, b1: int, inclusive: bool = True):
        """Return whether or not one text span overlaps with another.

        :param inclusive: whether to check include +1 on the end component

        :return: any overlap detected returns ``True``

        """
        if inclusive:
            m = (a0 <= b0 and a1 >= b0) or (b0 <= a0 and b1 >= a0)
        else:
            m = (a0 <= b0 and a1 > b0) or (b0 <= a0 and b1 > a0)
        return m

    def overlaps_with(self, other: LexicalSpan,
                      inclusive: bool = True) -> bool:
        """Return whether or not one text span overlaps non-inclusively with another.

        :param other: the other location

        :param inclusive: whether to check include +1 on the end component

        :return: any overlap detected returns ``True``

        """
        return self.overlaps(
            self.begin, self.end, other.begin, other.end, inclusive)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(str(self), depth, writer)

    def _from_dictable(self, *args, **kwargs):
        # prettier printing
        return dict(super()._from_dictable(*args, **kwargs))

    def __eq__(self, other):
        return isinstance(other, LexicalSpan) and \
            self.begin == other.begin and self.end == other.end

    def __lt__(self, other):
        if self.begin == other.begin:
            return self.end < other.end
        else:
            return self.begin < other.begin

    def __hash__(self) -> int:
        return hash(self.begin) + (13 * hash(self.end))

    def __setattr__(self, name, value):
        if hasattr(self, 'end'):
            raise AttributeError(f'{self.__class__.__name__} is immutable')
        super().__setattr__(name, value)

    def __getitem__(self, ix: int) -> int:
        if ix == 0:
            return self.begin
        elif ix == 1:
            return self.end
        raise KeyError(f'LexicalSpan index: {ix}')

    def __len__(self) -> int:
        return self.end - self.begin

    def __str__(self) -> str:
        return f'({self.begin}, {self.end})'

    def __repr__(self):
        return self.__str__()


class TextContainer(Dictable, metaclass=ABCMeta):
    """A *writable* class that has a ``text`` property or attribute.  All
    subclasses need a ``norm`` attribute or property.

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.__class__.__name__}:', depth, writer)
        self._write_line(f'original: {self.text}', depth + 1, writer)
        self._write_line(f'normalized: {self.norm}', depth + 1, writer)

    def __str__(self):
        return f'<{self.norm[:79]}>'

    def __repr__(self):
        return self.__str__()
