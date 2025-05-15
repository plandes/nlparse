"""A heuristic text indexing and search class.

"""
__author__ = 'Paul Landes'

from typing import Dict, List, Tuple, Iterable, Set, Type
from dataclasses import dataclass, field
import logging
import re
import textwrap as tw
from zensols.persist import persisted
from .container import (
    LexicalSpan, FeatureToken, TokenContainer, FeatureDocument, FeatureSentence
)

logger = logging.getLogger(__name__)

TokenOrth = Tuple[str, FeatureToken]


@dataclass
class FeatureDocumentIndexer(object):
    """A utility class that indexes and searches for text in potentially
    whitespace mangled documents.  It does this by trying more efficient means
    first, then resorts to methods that are more computationaly expensive.

    """
    doc: FeatureDocument = field()
    """The document to index."""

    @staticmethod
    def _get_norm(cont: TokenContainer, no_space: bool = False) -> str:
        """Create normalized text by removing continuous whitespace with a
        single space or the empty string.

        """
        repl: str = '' if no_space else ' '
        return re.sub(r'\s+', repl, cont.text).strip()

    @staticmethod
    def _get_tok_orths(cont: TokenContainer) -> Tuple[TokenOrth, ...]:
        """Return tuples of (<orthographic text>, <token>)."""
        return tuple(map(lambda t: (t.text.strip(), t),
                         filter(lambda t: not t.is_space, cont.token_iter())))

    @classmethod
    def _spans_equal(cls: Type, a: str, b: str) -> bool:
        """Return whether strings ``a`` and ``b`` are the same and log it."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'cmp <{cls._shorten(a)}> ?= <{cls._shorten(b)}>')
        return a == b

    @staticmethod
    def _get_pack2ix(text: str) -> Dict[int, int]:
        """Return a dictionary of character positions in ``text`` to respective
        positions in the same string without whitespace.

        """
        ixs: List[int] = []
        ws: Set[str] = set(' \r\n\t')
        text: str = text
        ix: int = 0
        c: str
        for c in text:
            ixs.append(ix)
            if c not in ws:
                ix += 1
        return dict(zip(ixs, range(len(ixs))))

    @property
    @persisted('_text2sent')
    def text2sent(self) -> Dict[str, FeatureSentence]:
        """Return a dictionary of sentence normalized text to respective
        sentence in :obj:`.doc`.

        """
        return {self._get_norm(s): s for s in self.doc}

    @property
    @persisted('_doc_tok_orths')
    def doc_tok_orths(self) -> Tuple[TokenOrth, ...]:
        """Reutrn tuples of (<orthographic text>, <token>)."""
        return self._get_tok_orths(self.doc)

    @property
    @persisted('_packed_doc_text')
    def packed_doc_text(self) -> str:
        """Return the document' (:obj:`doc`) no-space normalized text."""
        return self._get_norm(self.doc, True)

    @property
    @persisted('_pack2ix')
    def pack2ix(self) -> Dict[int, int]:

        """Return a dictionary of character positions in the document
        (:obj:`doc`) text to respective positions in the same string without
        whitespace.

        """
        return self._get_pack2ix(self.doc.text.rstrip())

    @staticmethod
    def _shorten(s: str) -> str:
        """Shorten text used for logging."""
        return tw.shorten(s, 80)

    def _find_start_offset(self, query: TokenContainer,
                           candidate: TokenContainer) -> TokenContainer:
        """Find the sub-sentence by exact matches on the indexed text."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'looking for candidate in <{query.text}>')
        cont: TokenContainer = None
        at_toks: Iterable[FeatureToken] = filter(
            lambda t: not t.is_space, query.token_iter())
        ca_toks: Iterable[FeatureToken] = filter(
            lambda t: not t.is_space, candidate.token_iter())
        at: FeatureToken
        ct: FeatureToken
        for i, (at, ct) in enumerate(zip(at_toks, ca_toks)):
            if at.text != ct.text:
                break
        if i > 0:
            lspan = LexicalSpan(candidate.lexspan[0], ct.lexspan[1])
            cont = self.doc.get_overlapping_span(lspan)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'found sentence start: <{cont.text}>')
        return cont

    def _find_by_sent_ix(self, sent_ix: int, query: TokenContainer,
                         targ_comp_text: str) -> TokenContainer:
        """Find the sentence by an index (when provided)."""
        sent: TokenContainer = None
        query_text: str = query.text
        if sent_ix < len(self.doc):
            candidate: TokenContainer = self.doc[sent_ix]
            cand_norm: str = self._get_norm(candidate)
            sent = self.text2sent.get(targ_comp_text)
            if sent is None and candidate is not None:
                if cand_norm.startswith(query_text) or \
                   query_text.startswith(cand_norm):
                    sent = self._find_start_offset(query, candidate)
        if sent is not None and \
           not self._spans_equal(targ_comp_text, self._get_norm(sent)):
            sent = None
        return sent

    def _find_doc_offset(self, query: TokenContainer,
                         targ_comp_text: str) -> TokenContainer:
        """Find the sub-sentence by finding subsequences of token text."""
        dorth: Tuple[TokenOrth, ...] = self.doc_tok_orths
        dtoks: Tuple[str, ...] = tuple(map(
            lambda t: t[0], self.doc_tok_orths))
        atoks: Tuple[str, ...] = tuple(map(
            lambda t: t[0], self._get_tok_orths(query)))
        alen: int = len(atoks)
        dpos: int = -1
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f'atoks: <{atoks}>')
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f'dtoks: <{dtoks}>')
        for i in range(len(dtoks)):
            if dtoks[i:i + alen] == atoks:
                dpos = i
                break
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'find offset: pos: {dpos}')
        if dpos > -1:
            lspan = LexicalSpan(dorth[dpos][1].lexspan[0],
                                dorth[dpos + alen - 1][1].lexspan[1])
            span: TokenContainer = self.doc.get_overlapping_span(lspan)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'found in doc: <{self._shorten(span.text)}>')
            if not self._spans_equal(targ_comp_text, self._get_norm(span)):
                span = None
            return span

    def _find_by_char(self, query: TokenContainer) -> TokenContainer:
        """Find the sub-span by removing all space, which is needed in cases
        where parsed tokens (such as entites and MIMIC redacted tokens) have
        space.

        """
        span: TokenContainer = None
        targ_comp: str = self._get_norm(query, True)
        doc_comp: str = self.packed_doc_text
        pack_ix: int = doc_comp.find(targ_comp)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'find by char: packed text index: {pack_ix}')
            logger.debug(f'comp span: <{self._shorten(query.text)}>')
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f'annotated: <{targ_comp}>')
            logger.trace(f'doc: <{doc_comp}>')
        if pack_ix > -1:
            pack2ix: Dict[int, int] = self.pack2ix
            start_ix: int = pack2ix.get(pack_ix)
            end_ix: int = pack2ix.get(pack_ix + len(targ_comp) - 1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'span: ({start_ix}, {end_ix})')
            if start_ix is not None and end_ix is not None:
                lspan = LexicalSpan(start_ix, end_ix)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'doc span: {lspan}')
                span = self.doc.get_overlapping_span(lspan)
                span_comp: str = self._get_norm(span, True)
                if not self._spans_equal(targ_comp, span_comp):
                    span = None
        return span

    def find(self, query: TokenContainer, sent_ix: int = None) -> \
            TokenContainer:
        """Find a sentence in document :obj:`doc`.  If a sentence index is
        given, it treats the query as a sentence to find in :obj:`doc`.

        :param query: the sentence to find in :obj:`doc`

        :param sent_ix: the sentence index hint if available

        :return: the matched text from :obj:`doc`

        """
        targ_comp_text: str = self._get_norm(query)
        span: TokenContainer = None
        if sent_ix is not None:
            span = self._find_by_sent_ix(sent_ix, query, targ_comp_text)
        if span is None:
            span = self._find_doc_offset(query, targ_comp_text)
        if span is None:
            span = self._find_by_char(query)
        return span
