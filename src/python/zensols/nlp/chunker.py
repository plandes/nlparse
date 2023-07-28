"""Clasess that segment text from :class:`.FeatureDocument` instances, but
retain the original structure by preserving sentence and token indicies.

"""
__author__ = 'Paul Landes'

from typing import ClassVar, Tuple, List, Iterable, Optional
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import textwrap as tw
import re
import logging
from . import LexicalSpan, TokenContainer, FeatureSentence, FeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunker(object, metaclass=ABCMeta):
    """Splits :class:`~zensols.nlp.container.TokenContainer` instances using
    regular expression :obj:`pattern`.  Matched container (implementation of the
    container is based on the subclass) are given if used as an iterable.  The
    document of all parsed containers is given if used as a callable.

    """
    doc: FeatureDocument = field()
    """The document that contains the entire text (i.e. :class:`.Note`)."""

    pattern: re.Pattern = field()
    """The chunk regular expression.  There should be a default for each
    subclass.

    """
    sub_doc: FeatureDocument = field(default=None)
    """A lexical span created document of :obj:`doc`, which defaults to the
    global document.  Providing this and :obj:`char_offset` allows use of a
    document without having to use :meth:`.TokenContainer.reindex`.

    """
    char_offset: int = field(default=0)
    """The 0-index absolute character offset where :obj:`sub_doc` starts.
    However, if the value is -1, then the offset is used as the begging
    character offset of the first token in the :obj:`sub_doc`.

    """
    def __post_init__(self):
        if self.sub_doc is None:
            self.sub_doc = self.doc

    def _get_coff(self) -> int:
        coff: int = self.char_offset
        if coff == -1:
            coff = next(self.sub_doc.token_iter()).lexspan.begin
        return coff

    def __iter__(self) -> Iterable[TokenContainer]:
        def match_to_span(m: re.Match) -> LexicalSpan:
            s: Tuple[int, int] = m.span(1)
            return LexicalSpan(s[0] + coff, s[1] + coff)

        def trunc(s: str) -> str:
            sh: str = tw.shorten(s, 50).replace('\n', '\\n')
            sh = f'<<{s}>>'
            return sh

        conts = []
        if self.sub_doc.token_len > 0:
            coff: int = self._get_coff()
            text: str = self.sub_doc.text
            gtext: str = self.doc.text
            matches: List[LexicalSpan] = \
                list(map(match_to_span, self.pattern.finditer(text)))
            if len(matches) > 0:
                tl: int = len(text) + coff
                start: int = matches[0].begin
                end: int = matches[-1].end
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'coff: {coff}, start={start}, end={end}')
                if start > coff:
                    fms = LexicalSpan(coff, start - 1)
                    matches.insert(0, fms)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'adding offset match: {start}, {coff}: ' +
                                     f'{gtext[fms[0]:fms[1]]}')
                if tl > end:
                    matches.append(LexicalSpan(end, tl))
                while len(matches) > 0:
                    span: LexicalSpan = matches.pop(0)
                    cont: TokenContainer = None
                    empty: bool = False
                    if logger.isEnabledFor(logging.DEBUG):
                        st: str = trunc(gtext[span[0]:span[1]])
                        logger.debug(
                            f'span begin: {span.begin}, start: {start}, ' +
                            f'match {span}: {st}')
                    if span.begin > start:
                        cont = self._create_container(
                            LexicalSpan(start, span.begin - 1))
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'create (trailing): {cont}')
                        empty = cont is None
                        if not empty:
                            if len(conts) > 0:
                                conts[-1] = self._merge_containers(
                                    conts[-1], cont)
                            else:
                                conts.append(cont)
                            cont = None
                            empty = True
                        matches.insert(0, span)
                    if not empty and cont is None:
                        cont = self._create_container(span)
                        if logger.isEnabledFor(logging.DEBUG):
                            st: str = trunc(gtext[span[0]:span[1]])
                            logger.debug(f'create (not empty) {st} -> {cont}')
                    if cont is not None:
                        conts.append(cont)
                    start = span.end + 1
        return iter(conts)

    @abstractmethod
    def _create_container(self, span: LexicalSpan) -> Optional[TokenContainer]:
        pass

    @abstractmethod
    def _merge_containers(self, a: TokenContainer, b: TokenContainer) -> \
            TokenContainer:
        pass

    @abstractmethod
    def to_document(self, conts: Iterable[TokenContainer]) -> FeatureDocument:
        pass

    def __call__(self) -> FeatureDocument:
        return self.to_document(self)


@dataclass
class ParagraphChunker(Chunker):
    """A :class:`.Chunker` that splits list item and enumerated lists into
    separate sentences.  Matched sentences are given if used as an iterable.

    """
    DEFAULT_SPAN_PATTERN: ClassVar[re.Pattern] = re.compile(
        r'(.+?)(?:(?=\n{2})|\Z)', re.MULTILINE | re.DOTALL)
    """The default paragraph regular expression, which uses two newline positive
    lookaheads to avoid matching on paragraph spacing.

    """
    pattern: re.Pattern = field(default=DEFAULT_SPAN_PATTERN)
    """The list regular expression, which defaults to
    :obj:`DEFAULT_SPAN_PATTERN`.

    """
    def _create_container(self, span: LexicalSpan) -> Optional[TokenContainer]:
        overlap_doc: FeatureDocument = self.doc.get_overlapping_document(span)
        sents: Iterable[FeatureSentence] = \
            filter(lambda s: len(s) > 0, map(lambda x: x.strip(), overlap_doc))
        merged_doc = FeatureDocument(
            sents=tuple(sents),
            text=overlap_doc.text.strip())
        if len(merged_doc) > 0:
            return merged_doc.strip()

    def _merge_containers(self, a: TokenContainer, b: TokenContainer) -> \
            TokenContainer:
        return FeatureDocument.combine_documents((a, b))

    def to_document(self, conts: Iterable[TokenContainer]) -> FeatureDocument:
        return FeatureDocument.combine_documents(conts)


@dataclass
class ListItemChunker(Chunker):
    """A :class:`.Chunker` that splits list item and enumerated lists into
    separate sentences.  Matched sentences are given if used as an iterable.
    This is useful when spaCy sentence chunks lists incorrectly and finds lists
    using a regular expression to find lines that star with a decimal, or list
    characters such as ``-`` and ``+``.

    """
    DEFAULT_SPAN_PATTERN: ClassVar[re.Pattern] = re.compile(
        r'^((?:[0-9-+]+|[a-zA-Z]+:)[^\n]+)$', re.MULTILINE)
    """The default list item regular expression, which uses an initial character
    item notation or an initial enumeration digit.

    """
    pattern: re.Pattern = field(default=DEFAULT_SPAN_PATTERN)
    """The list regular expression, which defaults to
    :obj:`DEFAULT_SPAN_PATTERN`.

    """
    def _create_container(self, span: LexicalSpan) -> Optional[TokenContainer]:
        doc: FeatureDocument = self.doc.get_overlapping_document(span)
        sent: FeatureSentence = doc.to_sentence()
        sent.strip()
        if sent.token_len > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'narrowed sent: <{sent.text}>')
            return sent

    def _merge_containers(self, a: TokenContainer, b: TokenContainer) -> \
            TokenContainer:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging: {a}||{b}')
        return FeatureDocument((a, b)).to_sentence(delim='\n')

    def to_document(self, conts: Iterable[TokenContainer]) -> FeatureDocument:
        sents: Tuple[FeatureSentence] = tuple(conts)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('creating doc from:')
            for s in sents:
                logger.debug(f'  {s}')
        return FeatureDocument(
            sents=sents,
            text='\n'.join(map(lambda s: s.text, sents)))
