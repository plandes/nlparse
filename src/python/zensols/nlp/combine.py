"""A class that combines features.

"""
__author__ = 'Paul Landes'

from typing import Set, List, Union
from dataclasses import dataclass, field
import logging
from spacy.tokens.span import Span
from spacy.tokens.token import Token
from . import (
    ParseError, FeatureDocumentParser,
    FeatureDocument, FeatureSentence, FeatureToken,
)

logger = logging.getLogger(__name__)


@dataclass
class CombinerFeatureDocumentParser(FeatureDocumentParser):
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
                           context_sent: FeatureSentence):
        for f in self.validate_features:
            prim = getattr(primary_tok, f)
            rep = getattr(replica_tok, f)
            if prim != rep:
                raise ParseError(
                    f'Mismatch tokens: {primary_tok.text}({f}={prim}) ' +
                    f'!= {replica_tok.text}({f}={rep}) ' +
                    f'in sentence: {context_sent}')

    def _merge_tokens(self, primary_tok: FeatureToken,
                      replica_tok: FeatureToken,
                      context_sent: FeatureSentence):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging tokens: {replica_tok} -> {primary_tok}')
        self._validate_features(primary_tok, replica_tok, context_sent)
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
    def _merge_sentence(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging sentences: {self._replica_sent.tokens} ' +
                         f'-> {self._primary_sent.tokens}')
        rmap = self._replica_token_mapping
        for primary_tok in self._primary_sent:
            entry = rmap.get(primary_tok.idx)
            logger.debug(f'entry: {primary_tok.idx}/{primary_tok} -> {entry}')
            if entry is not None:
                replica_tok, spacy_tok = entry
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'rep: {replica_tok}, spacy: {spacy_tok}')
                if self.clone_and_norm_replica_token and \
                   replica_tok.norm != spacy_tok.orth_:
                    replica_tok = replica_tok.clone()
                    replica_tok.norm = spacy_tok.orth_
                self._merge_tokens(
                    primary_tok, replica_tok, self._primary_sent)

    def _get_token_mapping(self, doc: FeatureDocument):
        mapping = {}
        for tok in doc.token_iter():
            tok_or_ent: Union[Span, Token] = tok.spacy_token
            if isinstance(tok_or_ent, Span):
                stoks = tuple(tok_or_ent)
                if len(stoks) > 1:
                    raise ParseError(
                        'Only support of singleton token spans is ' +
                        f'supportered, got: {stoks} in {doc}')
                stok = stoks[0]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'match span: {stok.idx} -> {tok}/{stok}')
                prev = mapping.get(stok.idx)
                if prev is not None:
                    raise ParseError(
                        f'Refusing to clobber previous mapping {tok} -> {prev}')
                mapping[stok.idx] = (tok, stok)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'match tok: {tok_or_ent.idx} -> ' +
                                 f'{tok}/{tok_or_ent}')
                prev = mapping.get(tok_or_ent.idx)
                if prev is not None:
                    raise ParseError(
                        f'Refusing to clobber previous mapping {tok} -> {prev}')
                mapping[tok_or_ent.idx] = (tok, tok_or_ent)
        return mapping

    def _prepare_merge_doc(self):
        self._replica_token_mapping = self._get_token_mapping(
            self._replica_sent)

    def _complete_merge_doc(self):
        del self._replica_token_mapping
