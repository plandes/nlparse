"""Generate features from text.

"""
__author__ = 'Paul Landes'

from typing import Dict
import logging
import sys
import inspect
import re
from spacy.tokens.token import Token

logger = logging.getLogger(__name__)


class TokenAttributes(object):
    """Contains token properties and a few utility methods.

    """
    FIELDS = 'text norm i tag is_wh entity dep children'.split()

    def to_dict(self):
        """Return the token attributes as a dictionary representation.

        """
        return self.__dict__

    @property
    def string_features(self) -> Dict[str, str]:
        """The features of the token as strings (both keys and values).

        """
        params = {'text': self.text,
                  'norm': self.norm,
                  'lemma': self.lemma,
                  'is_wh': self.is_wh,
                  'is_stop': self.is_stop,
                  'is_space': self.is_space,
                  'is_punctuation': self.is_punctuation,
                  'is_contraction': self.is_contraction,
                  'i': self.i,
                  'index': self.idx,
                  'tag': self.tag_,
                  'entity': self.ent_,
                  'is_entity': self.is_ent,
                  'shape': self.shape_,
                  'children': len(self.children),
                  'superlative': self.is_superlative,
                  'dep': self.dep_}
        return params

    @property
    def features(self) -> Dict[str, str]:
        """Return the features as numeric values.

        """
        if not hasattr(self, '_feats'):
            self._feats = {'tag': self.tag,
                           'is_wh': self.is_wh,
                           'is_stop': self.is_stop,
                           'is_pronoun': self.is_pronoun,
                           'index': self.idx,
                           'i': self.i,
                           'is_space': self.is_space,
                           'is_punctuation': self.is_punctuation,
                           'is_contraction': self.is_contraction,
                           'entity': self.ent,
                           'is_entity': self.is_ent,
                           'shape': self.shape,
                           'is_superlative': self.is_superlative,
                           'children': len(self.children),
                           'dep': self.dep}
        return self._feats

    def write(self, writer=sys.stdout, level=0):
        """Write the features in a human readable format.

        :param writer: where to output, defaults to standard out
        :param level: the indentation level

        """
        params = self.string_features
        sp = ' ' * level * 2
        fmt = '\n'.join(map(lambda x: '{}{}: {{{}}}'.format(sp, x, x),
                            self.FIELDS))
        fmt += '\n'
        writer.write(fmt.format(**params))
        writer.write('    num: {}\n'.format(self.features))

    def __str__(self):
        if hasattr(self, 'tok_or_ent'):
            tokstr = self.tok_or_ent
        else:
            tokstr = self.norm
        return '{} ({})'.format(tokstr, self.norm)

    def __repr__(self):
        return self.__str__()


class TokenFeatures(TokenAttributes):
    """Convenience class to create features from text.  The features are derived
    from parsed Spacy artifacts.

    """
    NONE = '<none>'
    PROP_REGEX = re.compile(r'^[a-z][a-z_-]*')

    def __init__(self, doc, tok_or_ent, norm):
        """Initialize a features instance.

        :param doc: the spacy document
        :tok_or_ent: either a token or entity parsed from ``doc``
        :norm: the normalized text of the token (i.e. lemmatized version)

        """
        self.doc = doc
        self.tok_or_ent = tok_or_ent
        self.is_ent = not isinstance(tok_or_ent, Token)
        self._norm = norm

    def detach(self, keeps=None) -> TokenAttributes:
        """Return a new instance of the object detached from SpaCy C data structures.
        This is useful for pickling of the object.

        :param keeps: the names of attributes to populate in the returned
                      instance; defaults to all

        """
        attrs = {}
        skips = set('doc token tok_or_ent features string_features'.split())
        for p, v in inspect.getmembers(self, lambda x: not callable(x)):
            if self.PROP_REGEX.match(p) and \
               p not in skips and \
               (keeps is None or p in keeps):
                attrs[p] = v
        ta = TokenAttributes()
        ta.__dict__.update(attrs)
        return ta

    def to_dict(self):
        return self.detach().to_dict()

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
        "Return the SpaCy token."
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
        return self.tok_or_ent.lemma_ == '-PRON-'

    @staticmethod
    def _is_apos(t) -> bool:
        """Return whether or not ``t`` is an apostrophy (') symbol.

        :param t: the string to compare

        """
        return (t.orth != t.lemma) and (t.orth_.find('\'') >= 0)

    @property
    def lemma(self) -> str:
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
