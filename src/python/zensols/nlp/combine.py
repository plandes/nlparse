"""A class that combines features.

"""
__author__ = 'Paul Landes'

from typing import Set, List, Dict, Any
from dataclasses import dataclass, field
import logging
from . import (
    ParseError, TokenContainer, FeatureDocumentParser,
    FeatureDocument, FeatureSentence, FeatureToken,
)
from .parser import DecoratedFeatureDocumentParser

logger = logging.getLogger(__name__)


@dataclass
class CombinerFeatureDocumentParser(DecoratedFeatureDocumentParser):
    """A class that combines features from two :class:`.FeatureDocumentParser`
    instances.  Features parsed using each :obj:`source_parser` are optionally
    copied or overwritten on a token by token basis in the feature document
    parsed by this instance.

    The target tokens are sometimes added to or clobbered from the source,
    but not the other way around.

    """
    source_parsers: List[FeatureDocumentParser] = field(default=None)
    """The language resource used to parse documents and create token
    attributes.

    """
    validate_features: Set[str] = field(default_factory=set)
    """A set of features to compare across all tokens when copying.  If any of
    the given features don't match, an mismatch token error is raised.

    """
    yield_features: List[str] = field(default_factory=list)
    """A list of features to be copied (in order) if the target token is not
    set.

    """
    yield_feature_defaults: Any = field(default=None)
    """A default value to use when no yielded value is found.  If ``None``, do
    not add the feature if missing.

    """
    overwrite_features: List[str] = field(default_factory=list)
    """A list of features to be copied/overwritten in order given in the list.

    """
    overwrite_nones: bool = field(default=False)
    """Whether to write ``None`` for missing :obj:`overwrite_features`.  This
    always write the *target* feature; if you only to write when the *source* is
    not set or missing, then use :obj:`yield_features`.

    """
    include_detached_features: bool = field(default=True)
    """Whether to include copied (yielded or overwritten) features as listed
    detected features.  This controls what is compared, cloned and for printed
    in :meth:`~zensols.config.writable.Writable.write`.

    :see: :obj:`.FeatureToken.default_detached_feature_ids`

    """
    def _validate_features(self, target_tok: FeatureToken,
                           source_tok: FeatureToken,
                           context_container: TokenContainer):
        for f in self.validate_features:
            prim = getattr(target_tok, f)
            rep = getattr(source_tok, f)
            if prim != rep:
                raise ParseError(
                    f'Mismatch tokens: {target_tok.text}({f}={prim}) ' +
                    f'!= {source_tok.text}({f}={rep}) ' +
                    f'in container: {context_container}')

    def _merge_tokens(self, target_tok: FeatureToken,
                      source_tok: FeatureToken,
                      context_container: TokenContainer):
        overwrite_nones: bool = self.overwrite_nones
        include_detached: bool = self.include_detached_features
        yield_default: Any = self.yield_feature_defaults
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f'merging tokens: {source_tok} ({type(source_tok)}) '
                         f'-> {target_tok} ({type(target_tok)})')
        self._validate_features(target_tok, source_tok, context_container)
        f: str
        for f in self.yield_features:
            targ = getattr(target_tok, f) if hasattr(target_tok, f) else None
            if logger.isEnabledFor(logging.TRACE):
                logger.trace(f'yield feature: {f}, target={targ}')
            if targ is None or targ == FeatureToken.NONE:
                src = getattr(source_tok, f) if hasattr(source_tok, f) else None
                if logger.isEnabledFor(logging.TRACE):
                    logger.trace(f'yield feature: {f}, src={src}')
                if (src is None or src == FeatureToken.NONE) and \
                   yield_default is not None:
                    src = yield_default
                if src is not None:
                    if logger.isEnabledFor(logging.TRACE):
                        logger.trace(f'{src} -> {target_tok.text}.{f}')
                    setattr(target_tok, f, src)
                    if include_detached and \
                       target_tok._detatched_feature_ids is not None:
                        target_tok._detatched_feature_ids.add(f)
        for f in self.overwrite_features:
            if overwrite_nones:
                src = getattr(source_tok, f)
            else:
                src = source_tok.get_value(f)
            src = FeatureToken.NONE if src is None else src
            if logger.isEnabledFor(logging.TRACE):
                prev = target_tok.get_value(f)
                logger.trace(
                    f'overwrite: {src} -> {prev} ({target_tok.text}.{f})')
            setattr(target_tok, f, src)
            if include_detached and \
               target_tok._detatched_feature_ids is not None:
                target_tok._detatched_feature_ids.add(f)

    def _debug_sentence(self, sent: FeatureSentence, name: str):
        if logging.isEnabledFor(logging.DEBUG):
            logger.debug(f'{name}:')
            for i, tok in enumerate(sent.tokens):
                logger.debug(f'  {i}: i={tok.i}, pos={tok.pos_}, ' +
                             f'ent={tok.ent_}: {tok}')

    def _merge_sentence(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging sentences: {self._source_sent.tokens} ' +
                         f'-> {self._target_sent.tokens}')
        n_toks: int = 0
        assert len(self._target_sent) == len(self._source_sent)
        for target_tok, source_tok in zip(
                self._target_sent, self._source_sent):
            assert target_tok.idx == source_tok.idx
            self._merge_tokens(target_tok, source_tok, self._target_sent)
        assert n_toks == len(self._target_sent)

    def _prepare_merge_doc(self):
        pass

    def _complete_merge_doc(self):
        pass

    def _merge_doc(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging docs: {self._source_doc} -> ' +
                         f'{self._target_doc}')
        for target_sent, source_sent in zip(
                self._target_doc, self._source_doc):
            self._target_sent = target_sent
            self._source_sent = source_sent
            self._prepare_merge_doc()
            try:
                self._merge_sentence()
            finally:
                del self._target_sent
                del self._source_sent
                self._complete_merge_doc()
        self._target_doc._combine_update(self._source_doc)

    def _merge_docs(self, target_doc: FeatureDocument,
                    source_doc: FeatureDocument):
        self._target_doc = target_doc
        self._source_doc = source_doc
        try:
            self._merge_doc()
        finally:
            del self._target_doc
            del self._source_doc

    def _parse(self, parsed: Dict[int, FeatureDocument], text: str,
               *args, **kwargs) -> FeatureDocument:
        self._log_parse(text, logger)
        key: int = id(self.delegate)
        target_doc: FeatureDocument = parsed.get(key)
        if target_doc is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'parsing with {self.delegate}')
            target_doc = self.delegate.parse(text, *args, **kwargs)
            parsed[key] = target_doc
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'resuing parsed doc from {self.delegate}')
        if self.source_parsers is None or len(self.source_parsers) == 0:
            logger.warning(f'No source parsers set on {self}, ' +
                           'which disables feature combining')
        else:
            for source_parser in self.source_parsers:
                source_doc: FeatureDocument
                if isinstance(source_parser, CombinerFeatureDocumentParser):
                    source_doc = source_parser._parse(
                        parsed, text, *args, **kwargs)
                else:
                    source_doc = source_parser.parse(text, *args, **kwargs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'merging {source_parser} -> {self.delegate}')
                self._merge_docs(target_doc, source_doc)
            parsed[id(source_parser)] = source_doc
        self.decorate(target_doc)
        return target_doc

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        return self._parse({}, text, *args, **kwargs)


@dataclass
class MappingCombinerFeatureDocumentParser(CombinerFeatureDocumentParser):
    """Maps the source to respective tokens in the target document using spaCy
    artifacts.

    """
    validate_features: Set[str] = field(default=frozenset({'idx'}))
    """A set of features to compare across all tokens when copying.  If any of
    the given features don't match, an mismatch token error is raised.  The
    default is the token's index in the document, which should not change in
    most cases.

    """
    merge_sentences: bool = field(default=True)
    """If ``False`` ignore sentences and map everything at the token level.
    Otherwise, the same hierarchy mapping as the super class is used.  This is
    useful when sentence demarcations are not aligned across source document
    parsers and this parser.

    """
    def _merge_entities_by_token(self, target_tok, source_tok):
        """Add the source sentence entity spans to the target sentence.  This is
        important so the original spaCy predicted entities are merged from the
        source to the target.

        """
        tdoc, sdoc = self._target_doc, self._source_doc
        tsent = None
        try:
            tsent = sdoc.get_overlapping_sentences(source_tok.lexspan)
        except StopIteration:
            pass
        if tsent is not None:
            tsent = next(tdoc.get_overlapping_sentences(target_tok.lexspan))
            skips = set(tsent._ents)
            for ent in tsent._ents:
                begin, end = ent
                if begin == source_tok.idx and ent not in skips:
                    tsent._ents.append(ent)

    def _merge_token_containers(self, target_container: TokenContainer,
                                rmap: Dict[int, FeatureToken]):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merge: {target_container}, mapping: {rmap}')
        include_detached: bool = self.include_detached_features
        visited: Set[FeatureToken] = set()
        targ_toks: Set[FeatureToken] = set(target_container.token_iter())
        for target_tok in target_container.token_iter():
            source_tok: FeatureToken = rmap.get(target_tok.idx)
            if logger.isEnabledFor(logging.TRACE):
                logger.trace(f'entry: {target_tok.idx}/{target_tok} ' +
                             f'-> {source_tok}')
            if source_tok is not None:
                visited.add(source_tok)
                self._merge_tokens(target_tok, source_tok, target_container)
                if source_tok.ent_ != FeatureToken.NONE:
                    self._merge_entities_by_token(target_tok, source_tok)
        for target_tok in (targ_toks - visited):
            fname: str
            for fname in self.overwrite_features:
                if not hasattr(target_tok, fname):
                    setattr(target_tok, fname, FeatureToken.NONE)
                    if include_detached and \
                       target_tok._detatched_feature_ids is not None:
                        target_tok._detatched_feature_ids.add(fname)

    def _merge_sentence(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'merging sentences: {self._source_sent.tokens} ' +
                         f'-> {self._target_sent.tokens}')
        rmp: Dict[int, FeatureToken] = self._source_token_mapping
        self._merge_token_containers(self._target_sent, rmp)
        self._target_doc._combine_update(self._source_doc)

    def _prepare_merge_doc(self):
        if self.merge_sentences:
            self._source_token_mapping = self._source_doc.tokens_by_idx

    def _complete_merge_doc(self):
        if self.merge_sentences:
            del self._source_token_mapping

    def _merge_doc(self):
        if self.merge_sentences:
            super()._merge_doc()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'merging docs: {self._source_doc} -> ' +
                             f'{self._target_doc}')
            source_token_mapping = self._source_doc.tokens_by_idx
            self._merge_token_containers(self._target_doc, source_token_mapping)
            self._target_doc._combine_update(self._source_doc)
