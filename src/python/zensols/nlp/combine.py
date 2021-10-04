"""A class that combines features.

"""
__author__ = 'Paul Landes'

from typing import Set, List
from dataclasses import dataclass, field
import logging
from . import (
    ParseError, TokenFeatures,
    FeatureDocumentParser, FeatureDocument, FeatureSentence, FeatureToken,
)

logger = logging.getLogger(__name__)


@dataclass
class CombinerFeatureDocumentParser(FeatureDocumentParser):
    """A class that combines features from two :class:`.FeatureDocumentParser`
    instances.  Features parsed using each :obj:`replica_parser` are optionally
    copied or overwritten on a token by token basis in the feature document
    parsed by this instance.

    """
    replica_parsers: List[FeatureDocumentParser] = field(default=None)
    """The language resource used to parse documents and create token attributes.

    """
    validate_features: Set[str] = field(default_factory=set)
    """A set of features to compare across all tokens when copying.  If any of the
    given features don't match, an mismatch token error is raised.

    """

    yield_features: List[str] = field(default_factory=list)
    """A list of features to be copied (in order) if the primary token is not set.

    """

    overwrite_features: List[str] = field(default_factory=list)
    """A list of features to be copied/overwritten in order given in the list.

    """

    def _get_tok_value(self, token: FeatureToken, attr: str):
        val = None
        if hasattr(token, attr):
            targ = getattr(token, attr)
            if targ is not None and targ != TokenFeatures.NONE and targ != 0:
                val = targ
        return val

    def _merge_tokens(self, primary_tok: FeatureToken,
                      replica_tok: FeatureToken):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging tokens: {replica_tok} -> {primary_tok}')
        for f in self.validate_features:
            prim = getattr(primary_tok, f)
            rep = getattr(replica_tok, f)
            if prim != rep:
                raise ParseError(
                    f'Mismatch tokens: {primary_tok.text}({f}={prim}) ' +
                    f'!= {replica_tok.text}({f}={rep})')
        for f in self.yield_features:
            targ = self._get_tok_value(primary_tok, f)
            if targ is None:
                src = self._get_tok_value(replica_tok, f)
                if src is not None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'{src} -> {primary_tok.text}.{f}')
                    setattr(primary_tok, f, src)
        for f in self.overwrite_features:
            src = self._get_tok_value(replica_tok, f)
            if src is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{src} -> {primary_tok.text}.{f}')
                setattr(primary_tok, f, src)

    def _merge_sentence(self, primary_sent: FeatureSentence,
                        replica_sent: FeatureSentence):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging sentences: {replica_sent.tokens} ' +
                         f'-> {primary_sent.tokens}')
        for primary_tok, replica_tok in zip(primary_sent, replica_sent):
            self._merge_tokens(primary_tok, replica_tok)

    def _merge_doc(self, primary_doc: FeatureDocument, replica_doc):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging docs: {replica_doc} -> {primary_doc}')
        for primary_sent, replica_sent in zip(primary_doc, replica_doc):
            self._merge_sentence(primary_sent, replica_sent)

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        primary_doc = super().parse(text, *args, **kwargs)
        if self.replica_parsers is None or len(self.replica_parsers) == 0:
            logger.warning(f'No replica parsers set on {self}, ' +
                           'which disables feature combining')
        else:
            for replica_parser in self.replica_parsers:
                replica_doc = replica_parser.parse(text, *args, **kwargs)
                self._merge_doc(primary_doc, replica_doc)
        return primary_doc
