"""Parse documents and generate features in an organized taxonomy.

"""
__author__ = 'Paul Landes'

from typing import List, Set, Type, Iterable, Tuple
from dataclasses import dataclass, field
import logging
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from . import (
    ParseError, LanguageResource, TokenFeatures,
    FeatureToken, FeatureSentence, FeatureDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureDocumentParser(object):
    """This class parses text in to instances of :class:`.FeatureDocument`
    instances.

    """
    TOKEN_FEATURE_IDS = FeatureToken.TOKEN_FEATURE_IDS

    langres: LanguageResource = field()
    """The language resource used to parse documents and create token attributes.

    """

    token_feature_ids: Set[str] = field(
        default_factory=lambda: FeatureDocumentParser.TOKEN_FEATURE_IDS)
    """The features to keep from spaCy tokens."""

    additional_token_feature_ids: Set[str] = field(default_factory=set)
    """More feature IDs in addition to :obj:`token_feature_ids`."""

    doc_class: Type[FeatureDocument] = field(default=FeatureDocument)
    """The type of document instances to create."""

    sent_class: Type[FeatureSentence] = field(default=FeatureSentence)
    """The type of sentence instances to create."""

    token_class: Type[FeatureToken] = field(default=FeatureToken)
    """The type of document instances to create."""

    remove_empty_sentences: bool = field(default=False)
    """If ``True``, remove sentences that only have space tokens."""

    def __post_init__(self):
        self.token_feature_ids = \
            self.token_feature_ids | self.additional_token_feature_ids

    def _create_token(self, feature: TokenFeatures) -> FeatureToken:
        return self.token_class(feature, self.token_feature_ids)

    def _create_sent(self, spacy_sent: Span, stoks: Iterable[TokenFeatures],
                     text: str) -> FeatureSentence:
        sent = tuple(map(self._create_token, stoks))
        sent = self.sent_class(sent, text)
        return sent

    def _from_string(self, text: str) -> Tuple[Doc, List[FeatureSentence]]:
        """Parse a document from a string.

        """
        lr: LanguageResource = self.langres
        doc: Doc = lr.parse(text)
        toks: Tuple[TokenFeatures] = tuple(lr.features(doc))
        ntoks = len(toks)
        tix = 0
        sents = []
        sent: Span
        for sent in doc.sents:
            if self.remove_empty_sentences and \
               (all(map(lambda t: t.is_space, sent)) or len(sent) == 0):
                continue
            e = sent[-1].i
            stoks = []
            while tix < ntoks:
                tok = toks[tix]
                if tok.i <= e:
                    stoks.append(tok)
                else:
                    break
                tix += 1
            sents.append(self._create_sent(sent, stoks, sent.text))
        return doc, sents

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        """Parse text or a text as a list of sentences.

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        """
        if not isinstance(text, str):
            raise ParseError(f'Expecting string text but got: {text}')
        spacy_doc, sents = self._from_string(text)
        try:
            return self.doc_class(sents, spacy_doc, *args, **kwargs)
        except Exception as e:
            raise ParseError(
                f'Could not parse <{text}> for {self.doc_class}') from e
