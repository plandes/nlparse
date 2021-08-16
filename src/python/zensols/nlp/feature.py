"""Domain container classes that can generate numeric and string features from
SpaCy artifacts.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any, Union, Iterable
import logging
import sys
from itertools import chain
from functools import reduce
from io import TextIOBase
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from zensols.config import Dictable

logger = logging.getLogger(__name__)


class TokenFeatures(Dictable):
    """A container class that create features from text.  The features are derived
    from parsed Spacy artifacts.

    The attributes of this class, such as ``norm`` and ``is_wh`` are referred
    to as *feature ids*.

    **Implementation note**: Many of the properties in this class are not
    efficient and many attributes (i.e. the spaCy
    :class:`~spacy.tokens.doc.Doc`) is heavy weight and not can not be pickled.
    For this reason, use a :class:`.FeatureDocumentParser` to create
    :class:`.FeatureToken` instances.

    """
    FIELD_IDS_BY_TYPE = {
        'bool': frozenset('is_space is_stop is_ent is_wh is_contraction is_superlative is_pronoun'.split()),
        'int': frozenset('i idx i_sent sent_i is_punctuation tag ent dep shape'.split()),
        'str': frozenset('norm lemma_ tag_ pos_ ent_ dep_ shape_'.split()),
        'list': frozenset('children'.split())}
    """Map of class type to set of feature IDs."""

    TYPES_BY_FIELD_ID = dict(chain.from_iterable(
        map(lambda itm: map(lambda f: (f, itm[0]), itm[1]),
            FIELD_IDS_BY_TYPE.items())))

    FIELD_IDS = frozenset(
        reduce(lambda res, x: res | x, FIELD_IDS_BY_TYPE.values()))

    WRITABLE_FIELD_IDS = tuple('text norm i i_sent tag pos is_wh entity dep children'.split())

    NONE = '<none>'
    """Default string for *not a feature*, or missing features."""

    def __init__(self, doc: Doc, tok_or_ent: Union[Token, Span], norm: str):
        """Initialize a features instance.

        :param doc: the spacy document

        :tok_or_ent: either a token or entity parsed from ``doc``

        :norm: the normalized text of the token (i.e. lemmatized version)

        """
        super().__init__()
        self.doc = doc
        self.tok_or_ent = tok_or_ent
        self.is_ent = not isinstance(tok_or_ent, Token)
        self._norm = norm

    # properties
    @property
    def text(self) -> str:
        """Return the unmodified parsed tokenized text.

        """
        return self.tok_or_ent.text

    @property
    def norm(self) -> str:
        """Return the normalized text, which is the text/orth or the named entity if
        tagged as a named entity.

        """
        return self._norm or self.NONE

    @property
    def token(self) -> Token:
        """Return the SpaCy token.

        """
        tok = self.tok_or_ent
        if self.is_ent:
            tok = self.doc[tok.start]
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
        return False if self.is_ent else self.tok_or_ent.pos_ == 'PRON'

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
        return self.tok_or_ent.orth_ if self.is_ent else self.tok_or_ent.lemma_

    @property
    def is_contraction(self) -> bool:
        """Return ``True`` if this token is a contradiction.

        """
        if self.is_ent:
            return False
        else:
            t = self.tok_or_ent
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
        return self.tok_or_ent.label if self.is_ent else 0

    @property
    def ent_(self) -> str:
        """Return the entity string label or ``None`` if this token has no entity.

        """
        return self.tok_or_ent.label_ if self.is_ent else self.NONE

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
        for six, sent in enumerate(self.doc.sents):
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

    # method
    def asdict(self, recurse: bool = True, readable: bool = True,
               class_name_param: str = None) -> Dict[str, Any]:
        """Return the token attributes as a dictionary representation.

        :see: :meth:`get_features`

        """
        return self.get_features(skip_missing=True)

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

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              field_ids: Iterable[str] = None):
        """Write the features in a human readable format.

        :param depth: the starting indentation depth

        :param writer: the writer to dump the content of this writable

        :param field_ids: the fields to write, which defaults to
                          :obj:`WRITABLE_FIELD_IDS`

        :see: :meth:`get_writable_features`

        """
        feats = self.get_features(self.WRITABLE_FIELD_IDS, True)
        for k in sorted(feats.keys()):
            self._write_line(f'{k}: {feats[k]}', depth, writer)

    def __str__(self):
        if hasattr(self, 'tok_or_ent'):
            tokstr = self.tok_or_ent
        else:
            tokstr = self.norm
        return '{} ({})'.format(tokstr, self.norm)

    def __repr__(self):
        return self.__str__()
