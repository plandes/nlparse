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
    char_offset: int = field(default=None)
    """The 0-index absolute character offset where :obj:`sub_doc` starts.
    However, if the value is -1, then the offset is used as the begging
    character offset of the first token in the :obj:`sub_doc`.

    """
    def __post_init__(self):
        if self.sub_doc is None:
            self.sub_doc = self.doc

    def _get_coff(self) -> int:
        coff: int = self.char_offset
        if coff is None:
            coff = self.doc.lexspan.begin
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

        conts: List[TokenContainer] = []
        if self.sub_doc.token_len > 0:
            # offset from the global document (if a subdoc from get_overlap...)
            coff: int = self._get_coff()
            # the text to match on, or ``gtext`` if there is no subdoc
            subdoc_text: str = self.sub_doc.text
            # the global document
            gtext: str = self.doc.text
            # all regular expression matches found in ``subdoc_text``
            matches: List[LexicalSpan] = \
                list(map(match_to_span, self.pattern.finditer(subdoc_text)))
            # guard on no-matches-found edge case
            if len(matches) > 0:
                subdoc_len: int = len(subdoc_text) + coff
                start: int = matches[0].begin
                end: int = matches[-1].end
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'coff: {coff}, start={start}, end={end}')
                # add a start front content match when not match on first char
                if start > coff:
                    fms = LexicalSpan(coff, start - 1)
                    matches.insert(0, fms)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'adding start match: {start}, {coff}: ' +
                                     f'{gtext[fms[0]:fms[1]]}')
                # and any trailing content when match doesn't include last char
                if subdoc_len > end:
                    matches.append(LexicalSpan(end, subdoc_len))
                # treat matches as a LIFO stack
                while len(matches) > 0:
                    # pop the first match in the stack
                    span: LexicalSpan = matches.pop(0)
                    cont: TokenContainer = None
                    if logger.isEnabledFor(logging.DEBUG):
                        st: str = trunc(gtext[span[0]:span[1]])
                        logger.debug(
                            f'span begin: {span.begin}, start: {start}, ' +
                            f'match {span}: {st}')
                    if span.begin > start:
                        # when the match comes after the last ending marker,
                        # added this content to the last match entry
                        cont = self._create_container(
                            LexicalSpan(start, span.begin - 1))
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'create (trailing): {cont}')
                        # content exists if it's text we keep (ie non-space)
                        if cont is not None:
                            if len(conts) > 0:
                                # tack on to the last entry since it trailed
                                # (probably after a newline)
                                conts[-1] = self._merge_containers(
                                    conts[-1], cont)
                            else:
                                # add a new entry
                                conts.append(cont)
                            # indcate we already added the content so we don't
                            # double add it
                            cont = None
                        # we dealt with the last trailling content from the
                        # previous span, but we haven't taken care of this span
                        matches.insert(0, span)
                    else:
                        # create and add the content for the exact match (again,
                        # we skip empty space etc.)
                        cont = self._create_container(span)
                        if logger.isEnabledFor(logging.DEBUG):
                            st: str = trunc(gtext[span[0]:span[1]])
                            logger.debug(f'create (not empty) {st} -> {cont}')
                        if cont is not None:
                            conts.append(cont)
                    # walk past this span to detect unmatched content for the
                    # next iteration (if there is one)
                    start = span.end + 1
        # adhere to iterable contract for potentially more dynamic subclasses
        return iter(conts)

    def _merge_containers(self, a: TokenContainer, b: TokenContainer) -> \
            TokenContainer:
        """Merge two token containers into one, which is used for straggling
        content tacked to previous entries for text between matches.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging: {a}||{b}')
        return FeatureDocument((a, b)).to_sentence()

    @abstractmethod
    def _create_container(self, span: LexicalSpan) -> Optional[TokenContainer]:
        """Create content from :obj:`doc` and :obj:`sub_doc` as a subdocument
        for span ``span``.

        """
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
    For this reason, this class will probably be used as an iterable since
    clients will usually want just the separated paragraphs as documents

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
    def _merge_containers(self, a: TokenContainer, b: TokenContainer) -> \
            TokenContainer:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging: {a}||{b}')
        # return documents to keep as much of the sentence structure as possible
        return FeatureDocument.combine_documents((a, b))

    def _create_container(self, span: LexicalSpan) -> Optional[TokenContainer]:
        doc: FeatureDocument = self.doc.get_overlapping_document(span)
        slen: int = len(doc.sents)
        # remove double newline empty sentences, happens at beginning or ending
        sents: Tuple[FeatureSentence] = tuple(
            filter(lambda s: len(s) > 0, map(lambda x: x.strip(), doc)))
        if slen != len(sents):
            # when we find surrounding whitespace, create a (sentence) stripped
            # document
            doc = FeatureDocument(sents=tuple(sents), text=doc.text.strip())
        if len(doc.sents) > 0:
            # we still need to strip per sentence for whitespace added at the
            # sentence level
            return doc.strip()

    def to_document(self, conts: Iterable[TokenContainer]) -> FeatureDocument:
        """It usually makes sense to use instances of this class as an iterable
        rather than this (see class docs).

        """
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
        # skip empty sentences, usually (spaCy) sentence chunked from text with
        # two newlines in a row
        sent.strip()
        if sent.token_len > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'narrowed sent: <{sent.text}>')
            return sent

    def to_document(self, conts: Iterable[TokenContainer]) -> FeatureDocument:
        sents: Tuple[FeatureSentence] = tuple(conts)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('creating doc from:')
            for s in sents:
                logger.debug(f'  {s}')
        return FeatureDocument(
            sents=sents,
            text='\n'.join(map(lambda s: s.text, sents)))
