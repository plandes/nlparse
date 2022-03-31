from __future__ import annotations
"""Domain objects that define features associated with text.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Set, Iterable, Dict, Type, Any
from dataclasses import dataclass, field
import dataclasses
from abc import ABCMeta, abstractmethod
import sys
import logging
from itertools import chain
import itertools as it
from io import TextIOBase
from frozendict import frozendict
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from zensols.persist import PersistableContainer, persisted, PersistedWork
from . import TextContainer, FeatureToken

logger = logging.getLogger(__name__)


class TokenContainer(PersistableContainer, TextContainer, metaclass=ABCMeta):
    """Each instance has the following attributes:

    """
    @abstractmethod
    def token_iter(self, *args) -> Iterable[FeatureToken]:
        """Return an iterator over the token features.

        :param args: the arguments given to :meth:`itertools.islice`

        """
        pass

    def norm_token_iter(self, *args) -> Iterable[str]:
        """Return a list of normalized tokens.

        :param args: the arguments given to :meth:`itertools.islice`

        """
        return map(lambda t: t.norm, self.token_iter(*args))

    @property
    @persisted('_norm', transient=True)
    def norm(self) -> str:
        """The normalized version of the sentence."""
        return ' '.join(self.norm_token_iter())

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

    @property
    def norms(self) -> Set[str]:
        return set(map(lambda t: t.norm.lower(),
                       filter(lambda t: not t.is_punctuation and not t.is_stop,
                              self.tokens)))

    @property
    def lemmas(self) -> Set[str]:
        return set(map(lambda t: t.lemma.lower(),
                       filter(lambda t: not t.is_punctuation and not t.is_stop,
                              self.tokens)))

    @property
    @persisted('_entities', transient=True)
    def entities(self) -> Tuple[Tuple[FeatureToken]]:
        """The named entities of the container with each multi-word entity as elements.

        """
        return self._get_entities()

    @abstractmethod
    def _get_entities(self) -> Tuple[Tuple[FeatureToken]]:
        pass

    @property
    @persisted('_tokens_by_idx', transient=True)
    def tokens_by_idx(self) -> Dict[int, FeatureToken]:
        """A map of tokens with keys as their character offset and values as tokens.

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

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              n_tokens: int = sys.maxsize):
        super().write(depth, writer)
        if n_tokens > 0:
            self._write_line('tokens:', depth + 1, writer)
            for t in it.islice(self.token_iter(), n_tokens):
                t.write(depth + 2, writer)

    def __str__(self):
        return TextContainer.__str__(self)

    def __repr__(self):
        return TextContainer.__repr__(self)


@dataclass(eq=True)
class FeatureSentence(TokenContainer):
    """A container class of tokens that make a sentence.  Instances of this class
    iterate over :class:`.FeatureToken` instances, and can create documents
    with :meth:`to_document`.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'spacy_sent'}
    """Don't serialize the spacy document on persistance pickling."""

    sent_tokens: Tuple[FeatureToken] = field()
    """The tokens that make up the sentence."""

    text: str = field(default=None)
    """The original raw text of the sentence."""

    spacy_sent: Span = field(default=None, repr=False, compare=False)
    """The parsed spaCy sentence this feature set is based.

    :see: :meth:`.FeatureDocument.spacy_doc`

    """
    def __post_init__(self):
        super().__init__()
        if self.text is None:
            self.text = ' '.join(map(lambda t: t.text, self.sent_tokens))
        self._ents = []
        self._set_entity_spans()

    def _set_entity_spans(self):
        if self.spacy_sent is not None:
            for ents in self.spacy_sent.ents:
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

    def token_iter(self, *args) -> Iterable[FeatureToken]:
        if len(args) == 0:
            return iter(self.sent_tokens)
        else:
            return it.islice(self.sent_tokens, *args)

    @property
    def tokens(self) -> Tuple[FeatureToken]:
        return self.sent_tokens

    @property
    def token_len(self) -> int:
        return len(self.sent_tokens)

    def to_sentence(self, limit: int = sys.maxsize) -> FeatureSentence:
        return self

    def to_document(self) -> FeatureDocument:
        return FeatureDocument([self])

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

    def _get_entities(self) -> Tuple[Tuple[FeatureToken]]:
        ents = []
        for start, end in self._ents:
            ent = []
            for tok in self.token_iter():
                if tok.idx >= start and tok.idx <= end:
                    ent.append(tok)
            if len(ent) > 0:
                ents.append(tuple(ent))
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


@dataclass
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

    sents: List[FeatureSentence] = field()
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

    def token_iter(self, *args) -> Iterable[FeatureToken]:
        sent_toks = chain.from_iterable(map(lambda s: s.tokens, self.sents))
        if len(args) == 0:
            return sent_toks
        else:
            return it.islice(sent_toks, *args)

    def sent_iter(self, *args) -> Iterable[FeatureSentence]:
        if len(args) == 0:
            return iter(self.sents)
        else:
            return it.islice(self.sents, *args)

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

    def to_sentence(self, *args) -> FeatureSentence:
        sents = self.sent_iter(*args)
        toks = chain.from_iterable(map(lambda s: s.tokens, sents))
        cls = self._sent_class()
        return cls(tuple(toks), self.text)

    def to_document(self) -> FeatureDocument:
        return self

    @persisted('_id_to_sent_pw', transient=True)
    def _id_to_sent(self) -> Dict[int, int]:
        id_to_sent = {}
        for six, sent in enumerate(self):
            for tok in sent:
                id_to_sent[tok.idx] = six
        return id_to_sent

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
        """Override if there are any fields in your dataclass.  In most cases, the only
        time this is called is by an embedding vectorizer to batch muultiple
        sentences in to a single document, so the only feature that matter are
        the sentence level.

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
            sents = list(chain.from_iterable(
                map(lambda d: d.combine_sentences(), docs)))
        else:
            sents = list(chain.from_iterable(docs))
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

    @persisted('_combine_sentences', transient=True)
    def combine_sentences(self) -> FeatureDocument:
        """Combine all the sentences in this document in to a new document with a
        single sentence.

        """
        if len(self.sents) == 1:
            return self
        else:
            sent_cls = self._sent_class()
            sent = sent_cls(self.tokens)
            doc = dataclasses.replace(self)
            doc.sents = [sent]
            doc._combined = True
            return doc

    def _reconstruct_sents_iter(self) -> Iterable[FeatureSentence]:
        for sent in self.sents:
            stoks = []
            ip_sent = -1
            for tok in sent:
                if tok.i_sent < ip_sent:
                    sent = FeatureSentence(stoks)
                    stoks = []
                    yield sent
                stoks.append(tok)
                ip_sent = tok.i_sent
        if len(stoks) > 0:
            yield FeatureSentence(stoks)

    def uncombine_sentences(self) -> FeatureDocument:
        """Reconstruct the sentence structure that we combined in
        :meth:`combine_sentences`.  If that has not been done in this instance,
        then return ``self``.

        """
        if hasattr(self, '_combined'):
            return FeatureDocument(tuple(self._reconstruct_sents_iter()))
        else:
            return self

    def _get_entities(self) -> Tuple[Tuple[FeatureToken]]:
        return tuple(chain.from_iterable(
            map(lambda s: s.entities, self.sents)))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              n_sents: int = sys.maxsize, n_tokens: int = 0):
        """Write the document and optionally sentence features.

        :param n_sents the number of sentences to write

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
            doc = cls(list(sents), text)
            doc.sents[0].annotations = tuple(anns)
            return doc
