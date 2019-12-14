"""Feature (ID) normalization.

"""
__author__ = 'Paul Landes'

import sys
import itertools as it
from abc import ABC
from spacy.vocab import Vocab


class SpacyFeatureNormalizer(ABC):
    """This normalizes feature IDs of parsed token features in to a number between
    [0, 1].  This is useful for normalized feature vectors as input to neural
    networks.  Input to this would be strings like ``token.ent_`` found on a
    ``TokenAttributes`` instance.

    The class is also designed to create features using indexes, so there are
    methods to resolve to a unique ID from an identifier.

    Instances of this class behave like a ``dict``.

    All symbols are taken from spacy.glossary.

    :see: spacy.glossary
    :see: feature.TokenAttributes

    """
    def __init__(self, vocab: Vocab = None):
        """Initialize.

        :param vocab: the vocabulary used for ``from_spacy`` to compute the
                      normalized feature from the spacy ID (i.e. token.ent_,
                      token.tag_ etc.)

        """
        self.vocab = vocab
        self.as_list = tuple(self.SYMBOLS.split())
        syms = dict(zip(self.as_list, it.count()))
        self.symbol_to_id = syms
        q = (len(syms) - 1)
        self.symbol_to_norm = {k: syms[k] / q for k in syms}

    def get(self, symbol: str, default: float = None) -> float:
        """Return a normalized feature float if ``symbol`` is found.

        :return: a normalized value between [0 - 1] or ``None`` if the symbol
                 isn't found

        """
        return self.symbol_to_norm.get(symbol, default)

    def __getitem__(self, symbol: str) -> float:
        "See ``get``."
        v = self.get(symbol, None)
        if v is None:
            raise KeyError(f'no such symbol: {symbol}')
        return v

    def __len__(self) -> int:
        return len(self.as_list)

    def id_from_spacy_symbol(self, id: int, default: int = -1) -> str:
        """Return the Spacy text symbol for it's ID (token.ent -> token.ent_).

        """
        strs = self.vocab.strings
        if id in strs:
            return strs[id]
        else:
            return default

    def from_spacy(self, id: int, default: float = None) -> float:
        """Return a normalized feature from a Spacy ID.

        """
        symbol = self.id_from_spacy_symbol(id)
        return self.get(symbol, default)

    def id_from_spacy(self, id: int, default: int = -1) -> int:
        """Return the ID of this normalizer for the Spacy ID or -1 if not found.

        """
        symbol = self.id_from_spacy_symbol(id)
        return self.symbol_to_id.get(symbol, default)

    def pprint(self, writer=sys.stdout):
        """Pretty print a human readable representation of this feature normalizer.

        """
        syms = self.symbol_to_id
        writer.write(f'{self.NAME}:\n')
        for k in sorted(syms.keys()):
            writer.write(f'  {k} => {syms[k]} ({self[k]})\n')

    @property
    def name(self) -> str:
        """A human readable name.

        """
        return self.NAME

    @property
    def attribute(self) -> str:
        """The attribute name found in an instance of ``TokenAttributes``.

        """
        return self.ATTRIBUTE

    def __str__(self):
        return f'{self.NAME} ({self.ATTRIBUTE})'

    def __repr__(self):
        return f'{self.__class__}: {self.__str__()}'


class NamedEntityRecognitionFeatureNormalizer(SpacyFeatureNormalizer):
    NAME = 'named entity recognition'
    LANG = 'en'
    ATTRIBUTE = 'ent'
    SYMBOLS = """PERSON NORP FACILITY FAC ORG GPE LOC PRODUCT EVENT WORK_OF_ART LAW LANGUAGE
    DATE TIME PERCENT MONEY QUANTITY ORDINAL CARDINAL PER MISC"""


class DependencyFeatureNormalizer(SpacyFeatureNormalizer):
    NAME = 'dependency'
    LANG = 'en'
    ATTRIBUTE = 'dep'
    SYMBOLS = """acl acomp advcl advmod agent amod appos attr aux auxpass case cc ccomp clf
complm compound conj cop csubj csubjpass dative dep det discourse dislocated
dobj expl fixed flat goeswith hmod hyph infmod intj iobj list mark meta neg
nmod nn npadvmod nsubj nsubjpass nounmod npmod num number nummod oprd obj obl
orphan parataxis partmod pcomp pobj poss possessive preconj prep prt punct
quantmod rcmod relcl reparandum root vocative xcomp ROOT"""


class PartOfSpeechFeatureNormalizer(SpacyFeatureNormalizer):
    NAME = 'part of speech'
    LANG = 'en'
    ATTRIBUTE = 'tag'
    SYMBOLS = """ADJ ADP ADV AUX CONJ CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM
VERB X EOL SPACE . , -LRB- -RRB- `` " ' $ # AFX CC CD DT EX FW HYPH IN JJ JJR
JJS LS MD NIL NN NNP NNPS NNS PDT POS PRP PRP$ RB RBR RBS RP TO UH VB VBD VBG
VBN VBP VBZ WDT WP WP$ WRB SP ADD NFP GW XX BES HVS NP PP VP ADVP ADJP SBAR PRT
PNP"""
