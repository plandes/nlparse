"""Generate features from text.

"""
__author__ = 'Paul Landes'

import logging
import sys
from spacy.tokens.token import Token

logger = logging.getLogger(__name__)


class TokenFeatures(object):
    """Convenience class to create features from text.  The features are derived
    from parsed Spacy artifacts.

    """
    FIELDS = 'text norm i tag is_wh entity dep children'.split()
    NONE = '<none>'

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

    @property
    def norm(self):
        return self._norm or self.NONE

    @property
    def token(self):
        tok = self.tok_or_ent
        if self.is_ent:
            tok = self.doc[tok.start]
        return tok

    @property
    def is_wh(self):
        return self.token.tag_.startswith('W')

    @property
    def is_stop(self):
        return not self.is_ent and self.token.is_stop

    @property
    def is_punctuation(self):
        return self.token.is_punct

    @property
    def is_pronoun(self):
        return self.tok_or_ent.lemma_ == '-PRON-'

    @staticmethod
    def _is_apos(t):
        return (t.orth != t.lemma) and (t.orth_.find('\'') >= 0)

    @property
    def lemma(self):
        return self.tok_or_ent.orth_ if self.is_ent else self.tok_or_ent.lemma_

    @property
    def is_contraction(self):
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
    def ent_(self):
        return self.tok_or_ent.label_ if self.is_ent else self.NONE

    @property
    def ent(self):
        return self.tok_or_ent.label if self.is_ent else 0

    @property
    def is_superlative(self):
        return self.token.tag_ == 'JJS'

    @property
    def string_features(self):
        params = {'text': self.tok_or_ent.text,
                  'norm': self.norm,
                  'lemma': self.lemma,
                  'is_wh': self.is_wh,
                  'is_stop': self.is_stop,
                  'is_punctuation': self.is_punctuation,
                  'is_contraction': self.is_contraction,
                  'i': self.token.i,
                  'tag': self.token.tag_,
                  'entity': self.ent_,
                  'is_entity': self.is_ent,
                  'shape': self.token.shape_,
                  'children': tuple(map(lambda x: x.orth_, self.token.children)),
                  'superlative': self.is_superlative,
                  'dep': self.token.dep_}
        return params

    def write(self, writer=sys.stdout, level=0):
        params = self.string_features
        sp = ' ' * level * 2
        fmt = '\n'.join(map(lambda x: '{}{}: {{{}}}'.format(sp, x, x),
                            self.FIELDS))
        fmt += '\n'
        writer.write(fmt.format(**params))
        writer.write('    num: {}\n'.format(self.features))

    @property
    def features(self):
        if not hasattr(self, '_feats'):
            self._feats = {'tag': self.token.tag,
                           'is_wh': self.is_wh,
                           'is_stop': self.is_stop,
                           'is_pronoun': self.is_pronoun,
                           'i': self.token.i,
                           'is_punctuation': self.is_punctuation,
                           'is_contraction': self.is_contraction,
                           'entity': self.ent,
                           'is_entity': self.is_ent,
                           'shape': self.token.shape,
                           'is_superlative': self.is_superlative,
                           'children': len(tuple(self.token.children)),
                           'dep': self.token.dep}
        return self._feats

    def __str__(self):
        return '{} ({})'.format(self.tok_or_ent, self.norm)

    def __repr__(self):
        return self.__str__()
