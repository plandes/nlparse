"""Lexical overlap detection.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any
from dataclasses import dataclass, field
import sys
from io import TextIOBase
from zensols.config import Dictable


def overlaps(a: Tuple[int, int], b: Tuple[int, int], inclusive: bool = True):
    """Return whether or not one text span overlaps with another.

    :return: any overlap detected returns ``True``

    """
    if inclusive:
        m = (a[0] <= b[0] and a[1] >= b[0]) or (b[0] <= a[0] and b[1] >= a[0])
    else:
        m = (a[0] <= b[0] and a[1] > b[0]) or (b[0] <= a[0] and b[1] > a[0])
    return m


@dataclass
class Location(Dictable):
    """A text location which is an interval of lexical text in a text document.

    """
    start: int = field()
    """The start of the span."""

    end: int = field()
    """The end of the span."""

    def __post_init__(self):
        self.interval = (self.start, self.end)

    def overlaps(self, other: Any) -> bool:
        """Return whether or not one text span overlaps non-inclusively with another.

        :param other: the other location
        :type other: Location

        :return: any overlap detected returns ``True``

        """
        a = self.interval
        b = other.interval
        return overlaps(a, b)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(str(self), depth, writer)

    def __getitem__(self, ix: int) -> int:
        return self.interval[ix]

    def __eq__(self, other):
        return self.interval == other.interval

    def __lt__(self, other):
        if self.interval[0] == other.interval[0]:
            return self.interval[1] < other.interval[1]
        else:
            return self.interval[0] < other.interval[0]

    def __hash__(self) -> int:
        return hash(self.interval)

    def __len__(self) -> int:
        return self.end - self.start

    def __str__(self) -> str:
        return f'[{self.start}, {self.end}]'
