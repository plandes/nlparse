"""A class that combines features.

"""
__author__ = 'Paul Landes'

from typing import Set, List, Tuple, Dict, Any
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
    yield_features: Set[str] = field(default_factory=list)
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
    map_features: List[Tuple[str, str, Any]] = field(default_factory=list)
    """Like :obj:`yield_features` but the feature ID can be different from the
    source to the target.  Each tuple has the form:

        ``(<source feature ID>, <target feature ID>, <default for missing>)``

    """
    def _validate_features(self, target_tok: FeatureToken,
                           source_tok: FeatureToken,
                           context_container: TokenContainer):
        for f in self.validate_features:
            prim = target_tok.get_feature(f, False)
            rep = source_tok.get_feature(f, False)
            if prim != rep:
                raise ParseError(
                    f'Mismatch tokens: {target_tok.text}({f}={prim}) ' +
                    f'!= {source_tok.text}({f}={rep}) ' +
                    f'in container: {context_container}')

    def _merge_tokens(self, target_tok: FeatureToken,
                      source_tok: FeatureToken,
                      context_container: TokenContainer):
        overwrite_nones: bool = self.overwrite_nones
        yield_default: Any = self.yield_feature_defaults
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f'merging tokens: {source_tok} ({type(source_tok)}) '
                         f'-> {target_tok} ({type(target_tok)})')
        self._validate_features(target_tok, source_tok, context_container)
        f: str
        dstf: str
        for f, dstf, _ in self.map_features:
            targ = target_tok.get_feature(dstf, False)
            src = source_tok.get_feature(f, False)
            if logger.isEnabledFor(logging.TRACE):
                logger.trace(f'map feature: {f} ({src}) -> {dstf} ({targ})')
            if (src is None or src == FeatureToken.NONE) and \
               yield_default is not None:
                src = yield_default
            if src is not None:
                if logger.isEnabledFor(logging.TRACE):
                    logger.trace(f'{src} -> {target_tok.text}.{dstf}')
                target_tok.set_feature(dstf, src)
        for f in self.yield_features:
            targ = target_tok.get_feature(f, False)
            if logger.isEnabledFor(logging.TRACE):
                logger.trace(f'yield feature: {f}, target={targ}')
            if targ is None or targ == FeatureToken.NONE:
                src = source_tok.get_feature(f, False)
                if logger.isEnabledFor(logging.TRACE):
                    logger.trace(f'yield feature: {f}, src={src}')
                if (src is None or src == FeatureToken.NONE) and \
                   yield_default is not None:
                    src = yield_default
                if src is not None:
                    if logger.isEnabledFor(logging.TRACE):
                        logger.trace(f'{src} -> {target_tok.text}.{f}')
                    target_tok.set_feature(f, src)
        for f in self.overwrite_features:
            src = source_tok.get_feature(f, False, not overwrite_nones)
            src = FeatureToken.NONE if src is None else src
            if logger.isEnabledFor(logging.TRACE):
                prev = target_tok.get_feature(f, False, True)
                logger.trace(
                    f'overwrite: {src} -> {prev} ({target_tok.text}.{f})')
            target_tok.set_feature(f, src)

    def _debug_sentence(self, sent: FeatureSentence, name: str):
        if logging.isEnabledFor(logging.DEBUG):
            logger.debug(f'{name}:')
            for i, tok in enumerate(sent.tokens):
                logger.debug(f'  {i}: i={tok.i}, {tok}')

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
        targ2idx: Dict[int, FeatureToken] = target_container.tokens_by_idx
        visited: Set[int] = set()
        for target_tok in target_container.token_iter():
            source_tok: FeatureToken = rmap.get(target_tok.idx)
            if logger.isEnabledFor(logging.TRACE):
                logger.trace(f'entry: {target_tok.idx}/{target_tok} ' +
                             f'-> {source_tok}')
            if source_tok is not None:
                visited.add(source_tok.idx)
                self._merge_tokens(target_tok, source_tok, target_container)
                self._merge_entities_by_token(target_tok, source_tok)
        targ_toks: Set[int] = set(map(
            lambda t: t.idx, target_container.token_iter()))
        not_visited: Tuple[FeatureToken, ] = tuple(map(
            lambda idx: targ2idx[idx], (targ_toks - visited)))
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f'not visited: {not_visited}')
        if len(self.map_features) > 0 or len(self.yield_features) > 0 or \
           len(self.overwrite_features) > 0:
            for target_tok in not_visited:
                targ_fid: str
                for _, targ_fid, default in self.map_features:
                    if not hasattr(target_tok, targ_fid):
                        if default is None:
                            default = FeatureToken.NONE
                        target_tok.set_feature(targ_fid, default)
                for targ_fid in self.yield_features:
                    if not hasattr(target_tok, targ_fid):
                        target_tok.set_feature(targ_fid, FeatureToken.NONE)
                for targ_fid in self.overwrite_features:
                    if not hasattr(target_tok, targ_fid):
                        target_tok.set_feature(targ_fid, FeatureToken.NONE)

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
