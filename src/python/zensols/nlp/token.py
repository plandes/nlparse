from __future__ import annotations
"""Feature token and related base classes

"""
__author__ = 'Paul Landes'

from typing import Union, Optional, Any, Set, Iterable, Dict, Sequence
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
import sys
from io import TextIOBase
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from zensols.persist import PersistableContainer
from . import NLPError, TextContainer, LexicalSpan


@dataclass
class FeatureToken(PersistableContainer, TextContainer):
    """A container class for features about a token.  This extracts only a subset
    of features from the heavy object :class:`.TokenFeatures`, which contains
    Spacy C data structures and is hard/expensive to pickle.

    **Feature note**: features ``i`` and ``i_sent`` are always added to
    features tokens to be able to reconstruct sentences (see
    :meth:`.FeatureDocument.uncombine_sentences`).

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'spacy_token'}
    """Don't serialize the spacy document on persistance pickling."""

    _DICTABLE_WRITABLE_DESCENDANTS = True
    """Use write method."""

    FIELD_IDS_BY_TYPE = {
        'bool': frozenset(('is_space is_stop is_ent is_wh is_contraction ' +
                           'is_superlative is_pronoun').split()),
        'int': frozenset(('i idx i_sent sent_i is_punctuation tag ' +
                          'ent dep shape').split()),
        'str': frozenset('norm lemma_ tag_ pos_ ent_ dep_ shape_'.split()),
        'list': frozenset('children'.split()),
        'object': frozenset('lexspan'.split())}
    """Map of class type to set of feature IDs."""

    TYPES_BY_FIELD_ID = dict(chain.from_iterable(
        map(lambda itm: map(lambda f: (f, itm[0]), itm[1]),
            FIELD_IDS_BY_TYPE.items())))
    """A map of feature ID to string type.  This is used by
    :meth:`.FeatureToken.write_attributes` to dump the type features.

    """
    FIELD_IDS = frozenset(
        reduce(lambda res, x: res | x, FIELD_IDS_BY_TYPE.values()))
    """All default available field IDs."""

    WRITABLE_FIELD_IDS = tuple(('text norm idx sent_i i i_sent tag pos ' +
                                'is_wh entity dep children').split())
    """Field IDs that are dumped on :meth:`write`."""

    NONE = '<none>'
    """Default string for *not a feature*, or missing features."""

    i: int = field()
    """The index of the token within the parent document."""

    idx: int = field()
    """The character offset of the token within the parent document."""

    i_sent: int = field()
    """The index of the within the parent sentence."""

    norm: str = field()
    """Normalized text, which is the text/orth or the named entity if tagged as a
        named entity.

    """
    def __post_init__(self):
        super().__init__()

    def detach(self, feature_ids: Set[str] = None) -> FeatureToken:
        """Create a detected token (i.e. from spacy artifacts).

        """
        return FeatureToken(**self.asdict())

    @property
    def text(self) -> str:
        """The initial text before normalized by any :class:`.TokenNormalizer`.

        """
        return self.norm

    def get_value(self, attr: str) -> Optional[Any]:
        """Get a value by attribute.

        :return: ``None`` when the value is not set

        """
        val = None
        if hasattr(self, attr):
            targ = getattr(self, attr)
            if targ is not None and targ != self.NONE and targ != 0:
                val = targ
        return val

    def get_features(self, field_ids: Iterable[str] = None,
                     skip_missing: bool = False) -> Dict[str, Any]:
        """Get both as a `:class:`dict`.

        :param field_ids: the fields to write, which defaults to
                          :obj:`FIELD_IDS`

        """
        field_ids = self.FIELD_IDS if field_ids is None else field_ids
        if skip_missing:
            field_ids = filter(lambda fid: hasattr(self, fid), field_ids)
        return {k: getattr(self, k) for k in field_ids}

    def to_vector(self, feature_ids: Sequence[str] = None) -> Iterable[str]:
        """Return an iterable of feature data.

        """
        if feature_ids is None:
            feature_ids = self.__dict__.keys()
        return map(lambda a: getattr(self, a), sorted(feature_ids))

    def _from_dictable(self, recurse: bool, readable: bool,
                       class_name_param: str = None) -> Dict[str, Any]:
        return self.get_features(skip_missing=True)

    def write_attributes(self, depth: int = 0, writer: TextIOBase = sys.stdout,
                         include_type: bool = True):
        """Write feature attributes.

        :param depth: the starting indentation depth

        :param writer: the writer to dump the content of this writable

        :param include_type: if ``True`` write the type of value (if available)

        """
        dct = self.asdict()
        for k in sorted(dct.keys()):
            val: str = dct[k]
            if include_type:
                ptype = self.TYPES_BY_FIELD_ID.get(k)
                ptype = '?' if ptype is None else ptype
                ptype = f' ({ptype})'
            else:
                ptype = ''
            self._write_line(f'{k}={val}{ptype}', depth, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        con = f'norm=<{self.norm}>'
        if self.text != self.norm:
            con += f' org=<{self.text}>'
        self._write_line(f'{self.__class__.__name__}: ' + con, depth, writer)
        self._write_line('attributes:', depth + 1, writer)
        self.write_attributes(depth + 2, writer)

    def __eq__(self, other: FeatureToken) -> bool:
        return self.i == other.i and self.idx == other.idx and \
            self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(self.i) * hash(self.i_sent)

    def __str__(self) -> str:
        return TextContainer.__str__(self)

    def __repr__(self) -> str:
        return self.__str__()

    def long_repr(self) -> str:
        attrs = []
        for s in 'norm lemma_ tag_ ent_'.split():
            v = getattr(self, s) if hasattr(self, s) else None
            if v is not None:
                attrs.append(f'{s}: {v}')
        return ', '.join(attrs)


@dataclass
class SpacyFeatureToken(FeatureToken):
    spacy_token: Union[Token, Span] = field(repr=False, compare=False)
    """The parsed spaCy token (or span if entity) this feature set is based.

    :see: :meth:`.FeatureDocument.spacy_doc`

    """
    def __post_init__(self):
        super().__post_init__()
        #self.norm = self.spacy_token.orth_
        self.is_ent: bool = not isinstance(self.spacy_token, Token)
        self._doc: Doc = self.spacy_token.doc

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
    def i(self) -> int:
        """The index of the token within the parent document.

        """
        return self.token.i

    @property
    def idx(self) -> int:
        """The character offset of the token within the parent document.

        """
        return self.token.idx

    @property
    def lexspan(self) -> LexicalSpan:
        """The document indexed lexical span using :obj:`idx`.

        """
        return LexicalSpan.from_token(self.spacy_token)

    @property
    def i_sent(self) -> int:
        """The index of the token in the respective sentence.  This is not to be
        confused with the index of the sentence to which the token belongs,
        which is :obj:`sent_i`.

        This attribute does not exist in a spaCy token, and was named as such
        to follow the naming conventions of their API.

        """
        return self.token.i - self.token.sent.start

    @property
    def sent_i(self) -> int:
        """The index of the sentence to which the token belongs.  This is not to be
        confused with the index of the token in the respective sentence, which
        is :obj:`i_sent`.

        This attribute does not exist in a spaCy token, and was named as such
        to follow the naming conventions of their API.

        """
        targ = self.i
        for six, sent in enumerate(self._doc.sents):
            for tok in sent:
                if tok.i == targ:
                    return six

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

    def __str__(self):
        if hasattr(self, 'spacy_token'):
            tokstr = self.spacy_token
        else:
            tokstr = self.norm
        return f'{tokstr} ({self.norm})'
