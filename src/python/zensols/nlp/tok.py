from __future__ import annotations
"""Feature token and related base classes

"""
__author__ = 'Paul Landes'

from typing import (
    Tuple, Union, Optional, Any, Set, Iterable, Dict, Sequence, ClassVar, Type
)
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
import sys
from io import TextIOBase
from frozendict import frozendict
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from zensols.persist import PersistableContainer
from . import NLPError, TextContainer, LexicalSpan


@dataclass
class FeatureToken(PersistableContainer, TextContainer):
    """A container class for features about a token.  Subclasses such as
    :class:`.SpacyFeatureToken` extracts only a subset of features from the
    heavy Spacy C data structures and is hard/expensive to pickle.

    **Feature note**: features ``i``, ``idx`` and ``i_sent`` are always added
    to features tokens to be able to reconstruct sentences (see
    :meth:`.FeatureDocument.uncombine_sentences`), and alwyas included.

    """
    _DICTABLE_WRITABLE_DESCENDANTS = True
    """Use write method."""

    REQUIRED_FEATURE_IDS: ClassVar[Set[str]] = frozenset(
        'i idx i_sent norm'.split())
    """Features retained regardless of configuration for basic functionality.

    """
    FEATURE_IDS_BY_TYPE: ClassVar[Dict[str, Set[str]]] = frozendict({
        'bool': frozenset(('is_space is_stop is_ent is_wh is_contraction ' +
                           'is_superlative is_pronoun').split()),
        'int': frozenset(('i idx i_sent sent_i is_punctuation tag ' +
                          'ent ent_iob dep shape norm_len').split()),
        'str': frozenset('norm lemma_ tag_ pos_ ent_ ent_iob_ dep_ shape_'.split()),
        'list': frozenset('children'.split()),
        'object': frozenset('lexspan'.split())})
    """Map of class type to set of feature IDs."""

    TYPES_BY_FEATURE_ID: ClassVar[Dict[str, str]] = frozendict(
        chain.from_iterable(
            map(lambda itm: map(lambda f: (f, itm[0]), itm[1]),
                FEATURE_IDS_BY_TYPE.items())))
    """A map of feature ID to string type.  This is used by
    :meth:`.FeatureToken.write_attributes` to dump the type features.

    """
    FEATURE_IDS: ClassVar[Set[str]] = frozenset(
        reduce(lambda res, x: res | x, FEATURE_IDS_BY_TYPE.values()))
    """All default available feature IDs."""

    WRITABLE_FEATURE_IDS: ClassVar[Tuple[str, ...]] = tuple(
        ('text norm idx sent_i i i_sent tag pos ' +
         'is_wh entity dep children').split())
    """Feature IDs that are dumped on :meth:`write` and :meth:`write_attributes`.

    """
    NONE: ClassVar[str] = '-<N>-'
    """Default string for *not a feature*, or missing features."""

    i: int = field()
    """The index of the token within the parent document."""

    idx: int = field()
    """The character offset of the token within the parent document."""

    i_sent: int = field()
    """The index of the token within the parent sentence.

    The index of the token in the respective sentence.  This is not to be
    confused with the index of the sentence to which the token belongs, which
    is :obj:`sent_i`.

    """
    norm: str = field()
    """Normalized text, which is the text/orth or the named entity if tagged as a
        named entity.

    """
    def __post_init__(self):
        super().__init__()
        self._detatched_feature_ids = None

    def detach(self, feature_ids: Set[str] = None,
               skip_missing: bool = False,
               cls: Type[FeatureToken] = None) -> FeatureToken:
        """Create a detected token (i.e. from spaCy artifacts).

        :param feature_ids: the features to write, which defaults to
                          :obj:`FEATURE_IDS`

        :param skip_missing: whether to only keep ``feature_ids``

        :param cls: the type of the new instance

        """
        cls = FeatureToken if cls is None else cls
        if feature_ids is None:
            feature_ids = set(self.FEATURE_IDS)
        else:
            feature_ids = set(feature_ids)
        feature_ids.update(self.REQUIRED_FEATURE_IDS)
        feats: Dict[str, Any] = self.get_features(feature_ids, skip_missing)
        clone = FeatureToken.__new__(cls)
        clone.__dict__.update(feats)
        if hasattr(self, '_text'):
            clone.text = self._text
        if feature_ids is not None:
            clone._detatched_feature_ids = feature_ids
        return clone

    @property
    def default_detached_feature_ids(self) -> Optional[Set[str]]:
        """The default set of feature IDs used when cloning or detaching
        with :meth:`clone` or :meth:`detatch`.

        """
        return self._detatched_feature_ids

    @default_detached_feature_ids.setter
    def default_detached_feature_ids(self, feature_ids: Set[str]):
        """The default set of feature IDs used when cloning or detaching
        with :meth:`clone` or :meth:`detatch`.

        """
        self._detatched_feature_ids = feature_ids

    def clone(self, cls: Type = None, **kwargs) -> FeatureToken:
        """Clone an instance of this token.

        :param cls: the type of the new instance

        :param kwargs: arguments to add to as attributes to the clone

        :return: the cloned instance of this instance

        """
        clone = self.detach(self._detatched_feature_ids, cls=cls)
        clone.__dict__.update(kwargs)
        return clone

    @property
    def text(self) -> str:
        """The initial text before normalized by any :class:`.TokenNormalizer`.

        """
        if hasattr(self, '_text'):
            return self._text
        else:
            return self.norm

    @text.setter
    def text(self, text: str):
        """The initial text before normalized by any :class:`.TokenNormalizer`.

        """
        self._text = text

    @property
    def is_none(self) -> bool:
        """Return whether or not this token is represented as none or empty."""
        return self._is_none(self.norm)

    @classmethod
    def _is_none(cls, targ: Any) -> bool:
        return targ is None or targ == cls.NONE or targ == 0

    def get_value(self, attr: str) -> Optional[Any]:
        """Get a value by attribute.

        :return: ``None`` when the value is not set

        """
        val = None
        if hasattr(self, attr):
            targ = getattr(self, attr)
            if not self._is_none(targ):
                val = targ
        return val

    def get_features(self, feature_ids: Iterable[str] = None,
                     skip_missing: bool = False) -> Dict[str, Any]:
        """Get features as a :class:`dict`.

        :param feature_ids: the features to write, which defaults to
                          :obj:`FEATURE_IDS`

        :param skip_missing: whether to only keep ``feature_ids``

        """
        feature_ids = self.FEATURE_IDS if feature_ids is None else feature_ids
        if skip_missing:
            feature_ids = filter(lambda fid: hasattr(self, fid), feature_ids)
        return {k: getattr(self, k) for k in feature_ids}

    def _from_dictable(self, recurse: bool, readable: bool,
                       class_name_param: str = None) -> Dict[str, Any]:
        dct = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                dct[k] = self._from_object(v, recurse, readable)
        return dct

    def to_vector(self, feature_ids: Sequence[str] = None) -> Iterable[str]:
        """Return an iterable of feature data.

        """
        if feature_ids is None:
            feature_ids = set(self.__dict__.keys()) - \
                {'_detatched_feature_ids'}
        return map(lambda a: getattr(self, a), sorted(feature_ids))

    def write_attributes(self, depth: int = 0, writer: TextIOBase = sys.stdout,
                         include_type: bool = True,
                         feature_ids: Iterable[str] = None,
                         inline: bool = False,
                         include_none: bool = True):
        """Write feature attributes.

        :param depth: the starting indentation depth

        :param writer: the writer to dump the content of this writable

        :param include_type: if ``True`` write the type of value (if available)

        :param feature_ids: the features to write, which defaults to
                          :obj:`WRITABLE_FEATURE_IDS`

        :param inline: whether to print attributes all on the same line

        """
        if feature_ids is None:
            feature_ids = self._detatched_feature_ids
        if feature_ids is None:
            feature_ids = self.WRITABLE_FEATURE_IDS
        dct = self.get_features(feature_ids, True)
        if 'text' in dct and dct['norm'] == dct['text']:
            del dct['text']
        for i, k in enumerate(sorted(dct.keys())):
            val: str = dct[k]
            ptype: str = None
            if not include_none and self._is_none(val):
                continue
            if include_type:
                ptype = self.TYPES_BY_FEATURE_ID.get(k)
                if ptype is not None:
                    ptype = f' ({ptype})'
            ptype = '' if ptype is None else ptype
            sout = f'{k}={val}{ptype}'
            if inline:
                if i == 0:
                    writer.write(self._sp(depth))
                else:
                    writer.write(', ')
                writer.write(sout)
            else:
                self._write_line(sout, depth, writer)
        if inline:
            self._write_empty(writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_type: bool = True,
              feature_ids: Iterable[str] = None):
        con = f'norm=<{self.norm}>'
        if self.text != self.norm:
            con += f' org=<{self.text}>'
        self._write_line(f'{self.__class__.__name__}: ' + con, depth, writer)
        self._write_line('attributes:', depth + 1, writer)
        self.write_attributes(depth + 2, writer, include_type, feature_ids)

    def __eq__(self, other: FeatureToken) -> bool:
        if id(self) == id(other):
            return True
        if self.i == other.i and self.idx == other.idx:
            a = dict(self.__dict__)
            b = dict(other.__dict__)
            del a['_detatched_feature_ids']
            del b['_detatched_feature_ids']
            return a == b
        return False

    def __lt__(self, other: FeatureToken) -> int:
        return self.idx < other.idx

    def __hash__(self) -> int:
        return (self.i + 1) * (self.idx + 1) * (self.i_sent + 1) * 13

    def __str__(self) -> str:
        return TextContainer.__str__(self)

    def __repr__(self) -> str:
        return self.__str__()

    # speed up none compares by using interned NONE
    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        if self.norm == self.NONE:
            del state['norm']
        return state

    # speed up none compares by using interned NONE
    def __setstate__(self, state: Dict[str, Any]):
        if 'norm' not in state:
            state['norm'] = self.NONE
        super().__setstate__(state)

    def long_repr(self) -> str:
        attrs = []
        for s in 'norm lemma_ tag_ ent_'.split():
            v = getattr(self, s) if hasattr(self, s) else None
            if v is not None:
                attrs.append(f'{s}: {v}')
        return ', '.join(attrs)


@dataclass(init=False)
class SpacyFeatureToken(FeatureToken):
    """Contains and provides the same features as a spaCy
    :class:`~spacy.tokens.Token`.

    """
    spacy_token: Union[Token, Span] = field(repr=False, compare=False)
    """The parsed spaCy token (or span if entity) this feature set is based.

    :see: :meth:`.FeatureDocument.spacy_doc`

    """
    def __init__(self, spacy_token: Union[Token, Span], norm: str):
        self.spacy_token = spacy_token
        self.is_ent: bool = not isinstance(self.spacy_token, Token)
        self._doc: Doc = self.spacy_token.doc
        i = self.token.i
        idx = self.token.idx
        i_sent = self.token.i - self.token.sent.start
        self._text = spacy_token.orth_
        super().__init__(i, idx, i_sent, norm)

    def __getstate__(self):
        raise NLPError('Not persistable')

    @property
    def token(self) -> Token:
        """Return the SpaCy token.

        """
        tok = self.spacy_token
        if isinstance(tok, Span):
            tok = self._doc[tok.start]
        return tok

    @property
    def is_wh(self) -> bool:
        """Return ``True`` if this is a WH word (i.e. what, where).

        """
        return self.token.tag_.startswith('W')

    @property
    def is_stop(self) -> bool:
        """Return ``True`` if this is a stop word.

        """
        return not self.is_ent and self.token.is_stop

    @property
    def is_punctuation(self) -> bool:
        """Return ``True`` if this is a punctuation (i.e. '?') token.

        """
        return self.token.is_punct

    @property
    def is_pronoun(self) -> bool:
        """Return ``True`` if this is a pronoun (i.e. 'he') token.

        """
        return False if self.is_ent else self.spacy_token.pos_ == 'PRON'

    @staticmethod
    def _is_apos(tok: Token) -> bool:
        """Return whether or not ``tok`` is an apostrophy (') symbol.

        :param tok: the token to copmare

        """
        return (tok.orth != tok.lemma_) and (tok.orth_.find('\'') >= 0)

    @property
    def lemma_(self) -> str:
        """Return the string lemma or text of the named entitiy if tagged as a named
        entity.

        """
        return self.spacy_token.orth_ if self.is_ent else self.spacy_token.lemma_

    @property
    def is_contraction(self) -> bool:
        """Return ``True`` if this token is a contradiction.

        """
        if self.is_ent:
            return False
        else:
            t = self.spacy_token
            if self._is_apos(t):
                return True
            else:
                doc = t.doc
                dl = len(doc)
                return ((t.i + 1) < dl) and self._is_apos(doc[t.i + 1])

    @property
    def ent(self) -> int:
        """Return the entity numeric value or 0 if this is not an entity.

        """
        return self.spacy_token.label if self.is_ent else 0

    @property
    def ent_(self) -> str:
        """Return the entity string label or ``None`` if this token has no entity.

        """
        return self.spacy_token.label_ if self.is_ent else self.NONE

    @property
    def ent_iob(self) -> int:
        """Return the entity IOB tag, which ``I`` for in, ```O`` for out, `B`` for
        begin.

        """
        return self.token.ent_iob if self.is_ent else 0

    @property
    def ent_iob_(self) -> str:
        """Return the entity IOB nominal index for :obj:``ent_iob``.

        """
        return self.token.ent_iob_ if self.is_ent else 'O'

    def conll_iob_(self) -> str:
        """Return the CoNLL formatted IOB tag, such as ``B-ORG`` for a beginning
        organization token.

        """
        if not self.is_ent:
            return 'O'
        return f'{self.self.token.ent_iob_}-{self.token.ent_type_}'

    @property
    def is_superlative(self) -> bool:
        """Return ``True`` if this token is the superlative.

        """
        return self.token.tag_ == 'JJS'

    @property
    def is_space(self) -> bool:
        """Return ``True`` if this token is white space only.

        """
        return self.token.is_space

    @property
    def sent_i(self) -> int:
        """The index of the sentence to which the token belongs.  This is not to be
        confused with the index of the token in the respective sentence, which
        is :obj:`.FeatureToken.i_sent`.

        This attribute does not exist in a spaCy token, and was named as such
        to follow the naming conventions of their API.

        """
        targ = self.i
        for six, sent in enumerate(self._doc.sents):
            for tok in sent:
                if tok.i == targ:
                    return six

    @property
    def lexspan(self) -> LexicalSpan:
        """The document indexed lexical span using :obj:`idx`.

        """
        return LexicalSpan.from_token(self.spacy_token)

    @property
    def tag(self) -> int:
        """Fine-grained part-of-speech text.

        """
        return self.token.tag

    @property
    def tag_(self) -> str:
        """Fine-grained part-of-speech text.

        """
        return self.token.tag_

    @property
    def pos(self) -> int:
        """The simple UPOS part-of-speech tag.

        """
        return self.token.pos

    @property
    def pos_(self) -> str:
        """The simple UPOS part-of-speech tag.

        """
        return self.token.pos_

    @property
    def shape(self) -> int:
        """Transform of the tokens’s string, to show orthographic features. For
        example, “Xxxx” or “d.

        """
        return self.token.shape

    @property
    def shape_(self) -> str:
        """Transform of the tokens’s string, to show orthographic features. For
        example, “Xxxx” or “d.

        """
        return self.token.shape_

    @property
    def children(self):
        """A sequence of the token’s immediate syntactic children.

        """
        return [c.i for c in self.token.children]

    @property
    def dep(self) -> int:
        """Syntactic dependency relation.

        """
        return self.token.dep

    @property
    def dep_(self) -> str:
        """Syntactic dependency relation string representation.

        """
        return self.token.dep_

    @property
    def norm_len(self) -> int:
        """The length of the norm in characters."""
        return len(self.norm)

    def __str__(self):
        if hasattr(self, 'spacy_token'):
            tokstr = self.spacy_token
        else:
            tokstr = self.norm
        return f'{tokstr} ({self.norm})'
