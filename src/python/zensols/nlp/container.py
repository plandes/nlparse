from __future__ import annotations
"""Domain objects that define features associated with text.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Iterable, Dict, Type, Any
from dataclasses import dataclass, field
import dataclasses
from abc import ABCMeta, abstractmethod
import sys
import logging
from itertools import chain
import itertools as it
import copy
from io import TextIOBase, StringIO
from frozendict import frozendict
from interlap import InterLap
from spacy.tokens import Doc, Span, Token
from zensols.persist import PersistableContainer, persisted
from . import NLPError, TextContainer, FeatureToken, LexicalSpan

logger = logging.getLogger(__name__)


class TokenContainer(PersistableContainer, TextContainer, metaclass=ABCMeta):
    """Each instance has the following attributes:

    """
    _SPACE_SKIP = set("""`‘“[({<""")
    _CONTRACTIONS = set("'s n't 'll 'm 've 'd 're".split())
    _LONGEST_CONTRACTION = max(map(len, _CONTRACTIONS))

    @abstractmethod
    def token_iter(self, *args, **kwargs) -> Iterable[FeatureToken]:
        """Return an iterator over the token features.

        :param args: the arguments given to :meth:`itertools.islice`

        """
        pass

    def norm_token_iter(self, *args, **kwargs) -> Iterable[str]:
        """Return a list of normalized tokens.

        :param args: the arguments given to :meth:`itertools.islice`

        """
        return map(lambda t: t.norm, self.token_iter(*args, **kwargs))

    @property
    @persisted('_norm', transient=True)
    def norm(self) -> str:
        """The normalized version of the sentence."""
        return self._calc_norm()

    def _calc_norm(self) -> str:
        """Create a string that follows English spacing rules."""
        nsent: str
        toks = self.tokens
        tlen = len(toks)
        has_punc = tlen > 0 and hasattr(toks[0], 'is_punctuation')
        if has_punc:
            space_skip = self._SPACE_SKIP
            contracts = self._CONTRACTIONS
            ncontract = self._LONGEST_CONTRACTION
            sio = StringIO()
            last_avoid = False
            for tix, tok in enumerate(toks):
                norm = tok.norm
                if tix > 0 and tix < tlen:
                    do_space_skip = False
                    nlen = len(norm)
                    if nlen == 1:
                        do_space_skip = norm in space_skip
                    if (not tok.is_punctuation or do_space_skip) and \
                       not last_avoid and \
                       not (nlen <= ncontract and norm in contracts):
                        sio.write(' ')
                    last_avoid = do_space_skip or tok.norm == '--'
                sio.write(norm)
            nsent = sio.getvalue()
        else:
            nsent = ' '.join(self.norm_token_iter())
        return nsent

    @property
    @persisted('_tokens', transient=True)
    def tokens(self) -> Tuple[FeatureToken]:
        """Return the token features as a tuple.

        """
        return tuple(self.token_iter())

    @property
    @persisted('_token_len', transient=True)
    def token_len(self) -> int:
        """Return the number of tokens."""
        return sum(1 for i in self.token_iter())

    @property
    @persisted('_lexspan', transient=True)
    def lexspan(self) -> LexicalSpan:
        """The document indexed lexical span using :obj:`idx`.

        """
        toks: Tuple[FeatureToken] = self.tokens
        if len(toks) == 0:
            return LexicalSpan.EMPTY_SPAN
        else:
            return LexicalSpan(toks[0].lexspan.begin, toks[-1].lexspan.end)

    @persisted('_interlap', transient=True)
    def _get_interlap(self) -> InterLap:
        """Create an interlap with all tokens of the container added."""
        il = InterLap()
        # adding with tuple inline is ~3 times as fast than a list, and ~9 times
        # faster than an individual add in a for loop
        il.add(tuple(map(lambda t: (t.lexspan.begin, t.lexspan.end, t),
                         self.token_iter())))
        return il

    def map_overlapping_tokens(self, spans: Iterable[LexicalSpan]) -> \
            Iterable[Tuple[FeatureToken]]:
        """Return a tuple of tokens, each tuple in the range given by the
        respective span in ``spans``.

        :param spans: the document 0-index character based spans to compare with
                      :obj:`.FeatureToken.lexspan`

        :return: a tuple of matching tokens for the respective ``span`` query

        """
        il = self._get_interlap()
        return map(lambda s: tuple(map(lambda m: m[2], il.find(s.astuple))),
                   spans)

    def get_overlapping_tokens(self, span: LexicalSpan) -> \
            Iterable[FeatureToken]:
        """Get all tokens that overlap lexical span ``span``.

        :param span: indicates the portion of the document to retain

        :return: a token sequence containing the 0 index offset of ``span``

        """
        return next(iter(self.map_overlapping_tokens((span,))))

    @abstractmethod
    def to_sentence(self, limit: int = sys.maxsize) -> FeatureSentence:
        """Coerce this instance to a single sentence.

        :param limit: the limit in the number of chunks to return

        :return: an instance of ``FeatureSentence`` that represents this token
                 sequence

        """
        pass

    @abstractmethod
    def to_document(self, limit: int = sys.maxsize) -> FeatureDocument:
        """Coerce this instance in to a document.

        """
        pass

    def clone(self, cls: Type[TokenContainer] = None, **kwargs) -> \
            TokenContainer:
        """Clone an instance of this token container.

        :param cls: the type of the new instance

        :param kwargs: arguments to add to as attributes to the clone

        :return: the cloned instance of this instance

        """
        cls = self.__class__ if cls is None else cls
        return cls(**kwargs)

    @property
    @persisted('_entities', transient=True)
    def entities(self) -> Tuple[FeatureSpan]:
        """The named entities of the container with each multi-word entity as elements.

        """
        return self._get_entities()

    @abstractmethod
    def _get_entities(self) -> Tuple[FeatureSpan]:
        pass

    @property
    @persisted('_tokens_by_idx', transient=True)
    def tokens_by_idx(self) -> Dict[int, FeatureToken]:
        """A map of tokens with keys as their character offset and values as tokens.

        **Limitations**: Multi-word entities will have have a mapping only for
        the first word of that entity if tokens were split by spaces (for
        example with :class:`~zensols.nlp.SplitTokenMapper`).  However,
        :obj:`tokens_by_i` does not have this limitation.

        :see: obj:`tokens_by_i`

        :see: :obj:`zensols.nlp.FeatureToken.idx`

        """
        by_idx = {}
        cnt = 0
        tok: FeatureToken
        for tok in self.token_iter():
            by_idx[tok.idx] = tok
            cnt += 1
        assert cnt == self.token_len
        return frozendict(by_idx)

    @property
    @persisted('_tokens_by_i', transient=True)
    def tokens_by_i(self) -> Dict[int, FeatureToken]:
        """A map of tokens with keys as their position offset and values as tokens.
        The entries also include named entity tokens that are grouped as
        multi-word tokens.  This is helpful for multi-word entities that were
        split (for example with :class:`~zensols.nlp.SplitTokenMapper`), and
        thus, have many-to-one mapped indexes.

        :see: :obj:`zensols.nlp.FeatureToken.i`

        """
        return frozendict(self._get_tokens_by_i())

    @abstractmethod
    def _get_tokens_by_i(self) -> Dict[int, FeatureToken]:
        pass

    def update_indexes(self):
        """Update all :obj:`.FeatureToken.i` attributes to those provided by
        :obj:`tokens_by_i`.  This corrects the many-to-one token index mapping
        for split multi-word named entities.

        :see: :obj:`tokens_by_i`

        """
        i: int
        ft: FeatureToken
        for i, ft in self.tokens_by_i.items():
            ft.i = i

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_original: bool = True, include_normalized: bool = True,
              n_tokens: int = sys.maxsize):
        super().write(depth, writer,
                      include_original=include_original,
                      include_normalized=include_normalized)
        if n_tokens > 0:
            self._write_line('tokens:', depth + 1, writer)
            for t in it.islice(self.token_iter(), n_tokens):
                t.write(depth + 2, writer)

    def __str__(self):
        return TextContainer.__str__(self)

    def __repr__(self):
        return TextContainer.__repr__(self)


@dataclass(eq=True, repr=False)
class FeatureSpan(TokenContainer):
    """A span of tokens as a :class:`.TokenContainer`, much like
    :class:`spacy.tokens.Span`.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'spacy_span'}
    """Don't serialize the spacy document on persistance pickling."""

    tokens: Tuple[FeatureToken] = field()
    """The tokens that make up the span."""

    text: str = field(default=None)
    """The original raw text of the span."""

    spacy_span: Span = field(default=None, repr=False, compare=False)
    """The parsed spaCy span this feature set is based.

    :see: :meth:`.FeatureDocument.spacy_doc`

    """
    def __post_init__(self):
        super().__init__()
        if self.text is None:
            self.text = ' '.join(map(lambda t: t.text, self.tokens))
        # the _tokens setter is called to set the tokens before the the
        # spacy_span set; so call it again since now we have spacy_span set
        self._set_entity_spans()

    @property
    def _tokens(self) -> Tuple[FeatureToken]:
        return self._tokens_val

    @_tokens.setter
    def _tokens(self, tokens: Tuple[FeatureToken]):
        if not isinstance(tokens, tuple):
            raise NLPError(
                f'Expecting tuple of tokens, but got {type(tokens)}')
        self._tokens_val = tokens
        self._ents: List[Tuple[int, int]] = []
        self._set_entity_spans()

    def _set_entity_spans(self):
        if self.spacy_span is not None:
            for ents in self.spacy_span.ents:
                start, end = None, None
                ents = iter(ents)
                try:
                    start = end = next(ents)
                    while True:
                        end = next(ents)
                except StopIteration:
                    pass
                if start is not None:
                    self._ents.append((start.idx, end.idx))

    def to_sentence(self, limit: int = sys.maxsize) -> FeatureSentence:
        if limit == 0:
            return iter(())
        else:
            return self.clone(FeatureSentence)

    def to_document(self) -> FeatureDocument:
        return FeatureDocument(self.to_sentence(),)

    def clone(self, cls: Type = None, **kwargs) -> TokenContainer:
        params = dict(kwargs)
        if 'tokens' not in params:
            params['tokens'] = tuple(
                map(lambda t: t.clone(), self._tokens_val))
        if 'text' not in params:
            params['text'] = self.text
        clone = super().clone(cls, **params)
        clone._ents = list(self._ents)
        return clone

    def token_iter(self, *args, **kwargs) -> Iterable[FeatureToken]:
        if len(args) == 0:
            return iter(self._tokens_val)
        else:
            return it.islice(self._tokens_val, *args, **kwargs)

    @property
    def token_len(self) -> int:
        return len(self._tokens_val)

    @property
    @persisted('_tokens_by_i_sent', transient=True)
    def tokens_by_i_sent(self) -> Dict[int, FeatureToken]:
        """A map of tokens with keys as their sentanal position offset and values as
        tokens.

        :see: :obj:`zensols.nlp.FeatureToken.i`

        """
        by_i_sent = {}
        cnt = 0
        tok: FeatureToken
        for tok in self.token_iter():
            by_i_sent[tok.i_sent] = tok
            cnt += 1
        assert cnt == self.token_len
        # add indexes for multi-word entities that otherwise have mappings for
        # only the first word of the entity
        ent_span: Tuple[FeatureToken]
        for ent_span in self.entities:
            t: FeatureToken
            for six, t in enumerate(ent_span):
                by_i_sent[t.i_sent + six] = t
        return frozendict(by_i_sent)

    def _get_tokens_by_i(self) -> Dict[int, FeatureToken]:
        by_i = {}
        cnt = 0
        tok: FeatureToken
        for tok in self.token_iter():
            by_i[tok.i] = tok
            cnt += 1
        assert cnt == self.token_len
        # add indexes for multi-word entities that otherwise have mappings for
        # only the first word of the entity
        ent_span: Tuple[FeatureToken]
        for ent_span in self.entities:
            t: FeatureToken
            for six, t in enumerate(ent_span):
                by_i[t.i + six] = t
        return by_i

    def _branch(self, node: FeatureToken, toks: Tuple[FeatureToken],
                tid_to_idx: Dict[int, int]) -> \
            Dict[FeatureToken, List[FeatureToken]]:
        clds = {}
        for c in node.children:
            cix = tid_to_idx.get(c)
            if cix:
                child = toks[cix]
                clds[child] = self._branch(child, toks, tid_to_idx)
        return clds

    @property
    @persisted('_dependency_tree', transient=True)
    def dependency_tree(self) -> Dict[FeatureToken, List[Dict[FeatureToken]]]:
        tid_to_idx: Dict[int, int] = {}
        toks = self.tokens
        for i, tok in enumerate(toks):
            tid_to_idx[tok.i] = i
        root = tuple(
            filter(lambda t: t.dep_ == 'ROOT' and not t.is_punctuation, toks))
        if len(root) == 1:
            return {root[0]: self._branch(root[0], toks, tid_to_idx)}
        else:
            return {}

    def _get_entities(self) -> Tuple[FeatureSpan]:
        ents = []
        for start, end in self._ents:
            ent = []
            for tok in self.token_iter():
                if tok.idx >= start and tok.idx <= end:
                    ent.append(tok)
            if len(ent) > 0:
                ents.append(FeatureSpan(tuple(ent)))
        return tuple(ents)

    def _from_dictable(self, recurse: bool, readable: bool,
                       class_name_param: str = None) -> Dict[str, Any]:
        return {'text': self.text,
                'tokens': self._from_object(self.tokens, recurse, readable)}

    def __getitem__(self, key) -> FeatureToken:
        return self.tokens[key]

    def __len__(self) -> int:
        return self.token_len

    def __iter__(self):
        return self.token_iter()


# keep the dataclass semantics, but allow for a setter
FeatureSpan.tokens = FeatureSpan._tokens


@dataclass(eq=True, repr=False)
class FeatureSentence(FeatureSpan):
    """A container class of tokens that make a sentence.  Instances of this class
    iterate over :class:`.FeatureToken` instances, and can create documents
    with :meth:`to_document`.

    """
    def __post_init__(self):
        super().__post_init__()

    def to_sentence(self, limit: int = sys.maxsize) -> FeatureSentence:
        if limit == 0:
            return iter(())
        else:
            return self

    def to_document(self) -> FeatureDocument:
        return FeatureDocument((self,))


@dataclass(eq=True, repr=False)
class FeatureDocument(TokenContainer):
    """A container class of tokens that make a document.  This class contains a one
    to many of sentences.  However, it can be treated like any
    :class:`.TokenContainer` to fetch tokens.  Instances of this class iterate
    over :class:`.FeatureSentence` instances.

    :param sents: the sentences defined for this document

    .. document private functions
    .. automethod:: _combine_documents

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'spacy_doc'}
    """Don't serialize the spacy document on persistance pickling."""

    sents: Tuple[FeatureSentence] = field()
    """The sentences that make up the document."""

    text: str = field(default=None)
    """The original raw text of the sentence."""

    spacy_doc: Doc = field(default=None, repr=False, compare=False)
    """The parsed spaCy document this feature set is based.  As explained in
    :class:`~zensols.nlp.FeatureToken`, spaCy documents are heavy weight and
    problematic to pickle.  For this reason, this attribute is dropped when
    pickled, and only here for ad-hoc predictions.

    """
    def __post_init__(self):
        super().__init__()
        if self.text is None:
            self.text = ''.join(map(lambda s: s.text, self.sent_iter()))
        if not isinstance(self.sents, tuple):
            raise NLPError(
                f'Expecting tuple of sentences, but got {type(self.sents)}')

    def set_spacy_doc(self, doc: Doc):
        ft_to_i: Dict[int, FeatureToken] = self.tokens_by_i
        st_to_i: Dict[int, Token] = {st.i: st for st in doc}
        i: int
        ft: FeatureToken
        for i, ft in ft_to_i.items():
            st: Token = st_to_i.get(i)
            if st is not None:
                ft.spacy_token = st
        fs: FeatureSentence
        ss: Span
        for ft, ss in zip(self.sents, doc.sents):
            ft.spacy_span = ss
        self.spacy_doc = doc

    def clone(self, cls: Type = None, **kwargs) -> TokenContainer:
        """
        :param kwargs: if `copy_spacy` is ``True``, the spacy document is
                       copied to the clone in addition parameters passed to new
                       clone initializer
        """
        params = dict(kwargs)
        if 'sents' not in params:
            params['sents'] = tuple(map(lambda s: s.clone(), self.sents))
        if 'text' not in params:
            params['text'] = self.text
        if params.pop('copy_spacy', False):
            for ss, cs in zip(self.sents, params['sents']):
                cs.spacy_span = ss.spacy_span
            params['spacy_doc'] = self.spacy_doc
        return super().clone(cls, **params)

    def token_iter(self, *args, **kwargs) -> Iterable[FeatureToken]:
        sent_toks = chain.from_iterable(
            map(lambda s: s.token_iter(), self.sents))
        if len(args) == 0:
            return sent_toks
        else:
            return it.islice(sent_toks, *args, **kwargs)

    def sent_iter(self, *args, **kwargs) -> Iterable[FeatureSentence]:
        if len(args) == 0:
            return iter(self.sents)
        else:
            return it.islice(self.sents, *args, **kwargs)

    @property
    def max_sentence_len(self) -> int:
        """Return the length of tokens from the longest sentence in the document.

        """
        return max(map(len, self.sent_iter()))

    def _sent_class(self) -> Type[FeatureSentence]:
        if len(self.sents) > 0:
            cls = self.sents[0].__class__
        else:
            cls = FeatureSentence
        return cls

    def to_sentence(self, *args, **kwargs) -> FeatureSentence:
        sents: Tuple[FeatureSentence] = tuple(self.sent_iter(*args, **kwargs))
        toks: Iterable[FeatureToken] = chain.from_iterable(
            map(lambda s: s.tokens, sents))
        cls: Type = self._sent_class()
        sent: FeatureSentence = cls(tokens=tuple(toks), text=self.text)
        sent._ents = list(chain.from_iterable(map(lambda s: s._ents, sents)))
        return sent

    def _combine_update(self, other: FeatureDocument):
        """Update internal data structures from another combined document.  This
        includes merging entities.

        :see :class:`.CombinerFeatureDocumentParser`

        :see: :class:`.MappingCombinerFeatureDocumentParser`

        """
        ss: FeatureSentence
        ts: FeatureSentence
        for ss, ts in zip(other, self):
            ents = set(ss._ents) | set(ts._ents)
            ts._ents = sorted(ents, key=lambda x: x[0])

    def to_document(self) -> FeatureDocument:
        return self

    @persisted('_id_to_sent_pw', transient=True)
    def _id_to_sent(self) -> Dict[int, int]:
        id_to_sent = {}
        for six, sent in enumerate(self):
            for tok in sent:
                id_to_sent[tok.idx] = six
        return id_to_sent

    def _get_tokens_by_i(self) -> Dict[int, FeatureToken]:
        by_i = {}
        for sent in self.sents:
            by_i.update(sent.tokens_by_i)
        return by_i

    def sentence_index_for_token(self, token: FeatureToken) -> int:
        """Return index of the parent sentence having ``token``."""
        return self._id_to_sent()[token.idx]

    def sentence_for_token(self, token: FeatureToken) -> FeatureSentence:
        """Return the parent sentence that has ``token``."""
        six: int = self.sentence_index_for_token(token)
        return self.sents[six]

    def sentences_for_tokens(self, tokens: Tuple[FeatureToken]) -> \
            Tuple[FeatureSentence]:
        """Find sentences having a set of tokens.

        :param tokens: the query used to finding containing sentences

        :return: the document ordered tuple of sentences containing `tokens`

        """
        id_to_sent = self._id_to_sent()
        sent_ids = sorted(set(map(lambda t: id_to_sent[t.idx], tokens)))
        return tuple(map(lambda six: self[six], sent_ids))

    def _combine_documents(self, docs: Tuple[FeatureDocument],
                           cls: Type[FeatureDocument],
                           concat_tokens: bool,
                           **kwargs) -> FeatureDocument:
        """Override if there are any fields in your dataclass.  In most cases,
        the only time this is called is by an embedding vectorizer to batch
        muultiple sentences in to a single document, so the only feature that
        matter are the sentence level.

        :param docs: the documents to combine in to one

        :param cls: the class of the instance to create

        :param concat_tokens:
            if ``True`` each sentence of the returned document are the
            concatenated tokens of each respective document; otherwise simply
            concatenate sentences in to one document

        :param kwargs: additional keyword arguments to pass to the new feature
                       document's initializer

        """
        if concat_tokens:
            sents = tuple(chain.from_iterable(
                map(lambda d: d.combine_sentences(), docs)))
        else:
            sents = tuple(chain.from_iterable(docs))
        if 'text' not in kwargs:
            kwargs = dict(kwargs)
            kwargs['text'] = ' '.join(map(lambda d: d.text, docs))
        return cls(sents, **kwargs)

    @classmethod
    def combine_documents(cls, docs: Iterable[FeatureDocument],
                          concat_tokens: bool = True,
                          **kwargs) -> FeatureDocument:
        """Coerce a tuple of token containers (either documents or sentences) in to one
        synthesized document.

        :param docs: the documents to combine in to one

        :param cls: the class of the instance to create

        :param concat_tokens:
            if ``True`` each sentence of the returned document are the
            concatenated tokens of each respective document; otherwise simply
            concatenate sentences in to one document

        :param kwargs: additional keyword arguments to pass to the new feature
                       document's initializer

        """
        docs = tuple(docs)
        if len(docs) == 0:
            doc = cls([], **kwargs)
        else:
            fdoc = docs[0]
            doc = fdoc._combine_documents(
                docs, type(fdoc), concat_tokens, **kwargs)
        return doc

    @persisted('_combine_all_sentences_pw', transient=True)
    def _combine_all_sentences(self) -> FeatureDocument:
        if len(self.sents) == 1:
            return self
        else:
            sent_cls = self._sent_class()
            sent = sent_cls(self.tokens)
            doc = dataclasses.replace(self)
            doc.sents = [sent]
            doc._combined = True
            return doc

    def combine_sentences(self, sents: Iterable[FeatureSentence] = None) -> \
            FeatureDocument:
        """Combine the sentences in this document in to a new document with a
        single sentence.

        :param sents: the sentences to combine in the new document or all if
                      ``None``

        """
        if sents is None:
            return self._combine_all_sentences()
        else:
            return self.__class__(tuple(sents))

    def _reconstruct_sents_iter(self) -> Iterable[FeatureSentence]:
        for sent in self.sents:
            stoks = []
            ip_sent = -1
            for tok in sent:
                if tok.i_sent < ip_sent:
                    sent = FeatureSentence(tuple(stoks))
                    stoks = []
                    yield sent
                stoks.append(tok)
                ip_sent = tok.i_sent
        if len(stoks) > 0:
            yield FeatureSentence(tuple(stoks))

    def uncombine_sentences(self) -> FeatureDocument:
        """Reconstruct the sentence structure that we combined in
        :meth:`combine_sentences`.  If that has not been done in this instance,
        then return ``self``.

        """
        if hasattr(self, '_combined'):
            return FeatureDocument(tuple(self._reconstruct_sents_iter()))
        else:
            return self

    def _get_entities(self) -> Tuple[FeatureSpan]:
        return tuple(chain.from_iterable(
            map(lambda s: s.entities, self.sents)))

    def get_overlapping_sentences(self, span: LexicalSpan) -> \
            Iterable[FeatureSentence]:
        """Return sentences that overlaps with ``span`` from this document."""
        for sent in self.sents:
            if sent.lexspan.overlaps_with(span):
                yield sent

    def get_overlapping_document(self, span: LexicalSpan) -> FeatureDocument:
        """Get the portion of the document that overlaps ``span``.  For sentences that
        are completely enclosed in the span, the sentences are copied.
        Otherwise, new sentences are created from those tokens that overlap the
        span.

        :param span: indicates the portion of the document to retain

        :return: a new document that contains the 0 index offset of ``span``

        """
        doc = self.clone()
        if span != self.lexspan:
            doc_text: str = self.text
            sents: List[FeatureSentence] = []
            for sent in self.sent_iter():
                toks = list(sent.get_overlapping_tokens(span))
                if len(toks) == 0:
                    continue
                elif len(toks) == len(sent):
                    pass
                else:
                    text: str = doc_text[toks[0].idx:toks[-1].idx + 1]
                    hang = (span.end + 1) - toks[-1].lexspan.end
                    if hang < 0:
                        tok = toks[-1]
                        clone = copy.deepcopy(tok)
                        clone.norm = tok.norm[:hang]
                        clone.text = tok.text[:hang]
                        toks[-1] = clone
                    hang = toks[0].lexspan.begin - span.begin
                    if hang < 0:
                        hang *= -1
                        tok = toks[0]
                        clone = copy.deepcopy(tok)
                        clone.norm = tok.norm[hang:]
                        clone.text = tok.text[hang:]
                        toks[0] = clone
                    sent = sent.clone(tokens=tuple(toks), text=text)
                sents.append(sent)
            text: str = doc_text[span.begin:span.end + 1]
            doc.sents = tuple(sents)
            doc.text = text
            body_len = sum(1 for _ in doc.get_overlapping_tokens(span))
            assert body_len == doc.token_len
        return doc

    def from_sentences(self, sents: Iterable[FeatureSentence],
                       deep: bool = False) -> FeatureDocument:
        """Return a new cloned document using the given sentences.

        :param sents: the sentences to add to the new cloned document

        :param deep: whether or not to clone the sentences

        :see: :meth:`clone`

        """
        if deep:
            sents = tuple(map(lambda s: s.clone(), sents))
        clone = self.clone(sents=sents)
        clone.text = ' '.join(map(lambda s: s.text, sents))
        clone.spacy_doc = None
        return clone

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              n_sents: int = sys.maxsize, n_tokens: int = 0):
        """Write the document and optionally sentence features.

        :param n_sents: the number of sentences to write

        :param n_tokens: the number of tokens to print across all sentences

        """
        TextContainer.write(self, depth, writer)
        self._write_line('sentences:', depth + 1, writer)
        s: FeatureSentence
        for s in it.islice(self.sents, n_sents):
            s.write(depth + 2, writer, n_tokens=n_tokens)

    def _from_dictable(self, recurse: bool, readable: bool,
                       class_name_param: str = None) -> Dict[str, Any]:
        return {'text': self.text,
                'sentences': self._from_object(self.sents, recurse, readable)}

    def __getitem__(self, key):
        return self.sents[key]

    def __len__(self):
        return len(self.sents)

    def __iter__(self):
        return self.sent_iter()


@dataclass
class TokenAnnotatedFeatureSentence(FeatureSentence):
    """A feature sentence that contains token annotations.

    """
    annotations: Tuple[Any] = field(default=())
    """A token level annotation, which is one-to-one to tokens."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              n_tokens: int = 0):
        super().write(depth, writer, n_tokens=n_tokens)
        n_ann = len(self.annotations)
        self._write_line(f'annotations ({n_ann}): {self.annotations}',
                         depth, writer)


@dataclass
class TokenAnnotatedFeatureDocuemnt(FeatureDocument):
    """A feature sentence that contains token annotations.  Sentences can be
    modeled with :class:`.TokenAnnotatedFeatureSentence` or just
    :class:`.FeatureSentence` since this sets the `annotations` attribute when
    combining.

    """
    @persisted('_combine_sentences', transient=True)
    def combine_sentences(self) -> FeatureDocument:
        """Combine all the sentences in this document in to a new document with a
        single sentence.

        """
        if len(self.sents) == 1:
            return self
        else:
            sent_cls = self._sent_class()
            anns = chain.from_iterable(map(lambda s: s.annotations, self))
            sent = sent_cls(self.tokens)
            sent.annotations = tuple(anns)
            doc = dataclasses.replace(self)
            doc.sents = [sent]
            doc._combined = True
            return doc

    def _combine_documents(self, docs: Tuple[FeatureDocument],
                           cls: Type[FeatureDocument],
                           concat_tokens: bool) -> FeatureDocument:
        if concat_tokens:
            return super()._combine_documents(docs, cls, concat_tokens)
        else:
            sents = chain.from_iterable(docs)
            text = ' '.join(chain.from_iterable(map(lambda s: s.text, docs)))
            anns = chain.from_iterable(map(lambda s: s.annotations, self))
            doc = cls(tuple(sents), text)
            doc.sents[0].annotations = tuple(anns)
            return doc
