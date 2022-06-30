"""Components useful for reuse.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Any, Union, Sequence, Optional
from dataclasses import dataclass, field
import logging
import re
from itertools import chain
import json
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.matcher import Matcher
from spacy.tokens import Span

logger = logging.getLogger(__name__)


@Language.component('remove_sent_boundaries')
def create_remove_sent_boundaries_component(doc: Doc):
    """Remove sentence boundaries from tokens.

    :param doc: the spaCy document to remove sentence boundaries

    """
    for token in doc:
        # this will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


@dataclass
class EntityRecognizer(object):
    """Base class regular expression and spaCy match patterns named entity
    recognizer.  Both subclasses allow for an optional label for each
    respective pattern or regular expression.  If the label is provided, then
    the match is made a named entity with a label.  In any case, a span is
    created on the token, and in some cases, retokenized.

    """
    nlp: Language = field()
    """The NLP model."""

    name: str = field()
    """The component name."""

    import_file: Optional[str] = field()
    """An optional JSON file used to append the pattern configuration."""

    patterns: List = field()
    """A list of the regular expressions to find."""

    def __post_init__(self):
        if self.import_file is not None:
            self._append_config(self.patterns)

    def _append_config(self, patterns: List):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating regex component for: {self.name}')
        if self.import_file is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'reading file config file: {self.import_file}')
            with open(self.import_file) as f:
                add_pats = json.load(f)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'adding to patterns: {add_pats}')
            patterns.extend(add_pats)

    def _make_span(self, doc: Doc, start: int, end: int, label: str,
                   is_char: bool, retok: bool):
        span: Span
        if is_char:
            if label is None:
                span = doc.char_span(start, end)
            else:
                span = doc.char_span(start, end, label=label)
        else:
            if label is None:
                span = Span(doc, start, end)
            else:
                span = Span(doc, start, end, label=label)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'span ({start}, {end}) for {label}: {span}')
        if span is not None:
            # this is a span object or none if match doesn't map to valid token
            # sequence
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'match: {span.text}')
            if label is not None:
                doc.ents += (span,)
            if retok:
                # https://github.com/explosion/spaCy/discussions/4806
                with doc.retokenize() as retokenizer:
                    # Iterate over all spans and merge them into one
                    # token. This is done after setting the entities –
                    # otherwise, it would cause mismatched indices!
                    retokenizer.merge(span)


@dataclass
class RegexEntityRecognizer(EntityRecognizer):
    """Merges regular expression matches as a :class:`~spacy.tokens.Span`.  After
    matches are found, re-tokenization merges them in to one token per match.

    """
    patterns: List[Tuple[str, Tuple[re.Pattern]]] = field()
    """A list of the regular expressions to find."""

    def __call__(self, doc: Doc) -> Doc:
        for label, regex_list in self.patterns:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'label: {label}, regex: {regex_list}')
            matches = map(lambda r: re.finditer(r, doc.text), regex_list)
            for match in chain.from_iterable(matches):
                start, end = match.span()
                self._make_span(doc, start, end, label, True, True)
        return doc


@Language.factory(
    'regexner', default_config={'patterns': [], 'path': None})
def create_regexner_component(
        nlp: Language, name: str,
        patterns: Sequence[Tuple[Optional[str],
                                 Sequence[Union[re.Pattern, str]]]],
        path: str = None):
    def map_rlist(rlist):
        rl = map(lambda x: x if isinstance(x, re.Pattern) else re.compile(x),
                 rlist)
        return tuple(rl)

    regexes = map(lambda x: (x[0], map_rlist(x[1])), patterns)
    return RegexEntityRecognizer(nlp, name, path, list(regexes))


@dataclass
class PatternEntityRecognizer(EntityRecognizer):
    """Adds entities based on regular epxressions.

    :see: `Rule matching <https://spacy.io/usage/rule-based-matching>`_

    """
    _NULL_LABEL = '<_>'

    patterns: List[Tuple[str, List[List[Dict[str, Any]]]]] = field()
    """The patterns given to the :class:`~spacy.matcher.Matcher`."""

    def __post_init__(self):
        super().__post_init__()
        self._matchers = []
        self._labels = {}
        for label, patterns in self.patterns:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'label: {label}')
                logger.debug(f'pattern: {patterns}')
            matcher = Matcher(self.nlp.vocab)
            label = self._NULL_LABEL if label is None else label                
            matcher.add(label, patterns, on_match=self._add_event_ent)
            self._labels[id(matcher)] = label
            self._matchers.append(matcher)

    def _add_event_ent(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        label = self._labels[id(matcher)]
        label = None if label == self._NULL_LABEL else label
        self._make_span(doc, start, end, label, False, False)

    def __call__(self, doc: Doc) -> Doc:
        for matcher in self._matchers:
            match: List[Tuple[int, int, int]] = matcher(doc)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'matched: {match}')
                logger.debug(f'doc ents: {doc.ents}')
        return doc


@Language.factory(
    'patner', default_config={'patterns': [], 'path': None})
def create_patner_component(
        nlp: Language, name: str,
        patterns: List[Tuple[Optional[str], List[List[Dict[str, Any]]]]],
        path: str = None):
    return PatternEntityRecognizer(nlp, name, path, list(patterns))
