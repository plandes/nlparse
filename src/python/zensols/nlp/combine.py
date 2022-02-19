"""A class that combines features.

"""
__author__ = 'Paul Landes'

from typing import Set, List, Dict, Tuple
from dataclasses import dataclass, field
import logging
from spacy.tokens.token import Token
from . import (
    ParseError, TokenContainer, FeatureDocumentParser,
    SpacyFeatureDocumentParser,
    FeatureDocument, FeatureSentence, FeatureToken,
)

logger = logging.getLogger(__name__)


@dataclass
class CombinerFeatureDocumentParser(SpacyFeatureDocumentParser):
    """A class that combines features from two :class:`.FeatureDocumentParser`
    instances.  Features parsed using each :obj:`replica_parser` are optionally
    copied or overwritten on a token by token basis in the feature document
    parsed by this instance.

    The primary tokens are sometimes added to or clobbered from the replica,
    but not the other way around.

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
    def _validate_features(self, primary_tok: FeatureToken,
                           replica_tok: FeatureToken,
                           context_container: TokenContainer):
        for f in self.validate_features:
            prim = getattr(primary_tok, f)
            rep = getattr(replica_tok, f)
            if prim != rep:
                raise ParseError(
                    f'Mismatch tokens: {primary_tok.text}({f}={prim}) ' +
                    f'!= {replica_tok.text}({f}={rep}) ' +
                    f'in container: {context_container}')

    def _merge_tokens(self, primary_tok: FeatureToken,
                      replica_tok: FeatureToken,
                      context_container: TokenContainer):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging tokens: {replica_tok} -> {primary_tok}')
        self._validate_features(primary_tok, replica_tok, context_container)
        for f in self.yield_features:
            targ = primary_tok.get_value(f)
            if targ is None:
                src = replica_tok.get_value(f)
                if src is not None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'{src} -> {primary_tok.text}.{f}')
                    setattr(primary_tok, f, src)
        for f in self.overwrite_features:
            src = replica_tok.get_value(f)
            if src is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{src} -> {primary_tok.text}.{f}')
                setattr(primary_tok, f, src)

    def _debug_sentence(self, sent: FeatureSentence, name: str):
        logger.debug(f'{name}:')
        for i, tok in enumerate(sent.tokens):
            logger.debug(f'  {i}: i={tok.i}, pos={tok.pos_}, ' +
                         f'ent={tok.ent_}: {tok}')

    def _merge_sentence(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging sentences: {self._replica_sent.tokens} ' +
                         f'-> {self._primary_sent.tokens}')
        for primary_tok, replica_tok in zip(
                self._primary_sent, self._replica_sent):
            self._merge_tokens(primary_tok, replica_tok, self._primary_sent)

    def _prepare_merge_doc(self):
        pass

    def _complete_merge_doc(self):
        pass

    def _merge_doc(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging docs: {self._replica_doc} -> ' +
                         f'{self._primary_doc}')
        for primary_sent, replica_sent in zip(
                self._primary_doc, self._replica_doc):
            self._primary_sent = primary_sent
            self._replica_sent = replica_sent
            self._prepare_merge_doc()
            try:
                self._merge_sentence()
            finally:
                del self._primary_sent
                del self._replica_sent
                self._complete_merge_doc()

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        primary_doc = super().parse(text, *args, **kwargs)
        if self.replica_parsers is None or len(self.replica_parsers) == 0:
            logger.warning(f'No replica parsers set on {self}, ' +
                           'which disables feature combining')
        else:
            for replica_parser in self.replica_parsers:
                replica_doc = replica_parser.parse(text, *args, **kwargs)
                self._primary_doc = primary_doc
                self._replica_doc = replica_doc
                try:
                    self._merge_doc()
                finally:
                    del self._primary_doc
                    del self._replica_doc
        return primary_doc


@dataclass
class MappingCombinerFeatureDocumentParser(CombinerFeatureDocumentParser):
    """Maps the replica to respective tokens in the primary document using spaCy
    artifacts.

    """
    validate_features: Set[str] = field(default=frozenset({'norm'}))
    """A set of features to compare across all tokens when copying.  If any of the
    given features don't match, an mismatch token error is raised.

    """

    clone_and_norm_replica_token: bool = field(default=True)
    """If ``True``, clone the replica token and clobber the ``norm`` field with the
    text of the spaCy token.

    """
    merge_sentences: bool = field(default=True)
    """If ``False`` ignore sentences and map everything at the token level.
    Otherwise, it use the same hierarchy mapping as the super class.  This is
    useful when sentence demarcations are not aligned across replica document
    parsers and this parser.

    """
    def _merge_token_containers(self, primary_container: TokenContainer,
                                rmap: Dict[int, Tuple[FeatureToken, Token]]):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merge: {primary_container}, mapping: {rmap}')
        for primary_tok in primary_container.token_iter():
            replica_tok = rmap.get(primary_tok.idx)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'entry: {primary_tok.idx}/{primary_tok} ' +
                             f'-> {replica_tok}')
            if replica_tok is not None:
                self._merge_tokens(primary_tok, replica_tok, primary_container)

    def _merge_sentence(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging sentences: {self._replica_sent.tokens} ' +
                         f'-> {self._primary_sent.tokens}')
        rmap: Dict[int, Tuple[FeatureToken, Token]] = self._replica_token_mapping
        self._merge_token_containers(self._primary_sent, rmap)

    def _get_token_mapping(self, doc: FeatureDocument) -> \
            Dict[int, Tuple[FeatureToken, Token]]:
        mapping = {}
        tok: FeatureToken
        for tok in doc.token_iter():
            prev = mapping.get(tok.idx)
            if prev is not None:
                raise ParseError(
                    f'Refusing to clobber previous mapping {tok} -> {prev}')
            mapping[tok.idx] = tok
        return mapping

    def _prepare_merge_doc(self):
        if self.merge_sentences:
            self._replica_token_mapping = self._get_token_mapping(
                self._replica_sent)

    def _complete_merge_doc(self):
        if self.merge_sentences:
            del self._replica_token_mapping

    def _merge_doc(self):
        if self.merge_sentences:
            super()._merge_doc()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'merging docs: {self._replica_doc} -> ' +
                             f'{self._primary_doc}')
            replica_token_mapping = self._get_token_mapping(self._replica_doc)
            self._merge_token_containers(self._primary_doc, replica_token_mapping)
