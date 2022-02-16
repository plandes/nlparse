"""Components useful for reuse.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Any, Union
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
    """Remove sentence boundaries since the corpus already delimits the sentences
    by newlines.  Otherwise, spaCy will delimit incorrectly as it gets confused
    with the capitalization in the clickbate "headlines".

    This configuration is used in the ``default.conf`` file's
    ``remove_sent_boundaries_component`` section.

    :param doc: the spaCy document to remove sentence boundaries

    """
    for token in doc:
        # this will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


@dataclass
class RegularExpressionMerger(object):
    """Merges regular expression matches as a :class:`~spacy.tokens.Span`.  After
    matches are found, re-tokenization merges them in to one token per match.

    """
    nlp: Language = field()
    """The NLP model."""

    regexs: Tuple[re.Pattern] = field()
    """A list of the regular expressions to find."""

    def __call__(self, doc: Doc) -> Doc:
        regexes = map(lambda r: re.finditer(r, doc.text), self.regexs)
        for match in chain.from_iterable(regexes):
            start, end = match.span()
            span = doc.char_span(start, end)
            # This is a Span object or None if match doesn't map to valid token
            # sequence
            if span is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'match: {span.text}')
                # https://github.com/explosion/spaCy/discussions/4806
                with doc.retokenize() as retokenizer:
                    # Iterate over all spans and merge them into one
                    # token. This is done after setting the entities –
                    # otherwise, it would cause mismatched indices!
                    retokenizer.merge(span)
        return doc


@Language.factory('regexmerge', default_config={'patterns': []})
def create_regex_merge_component(
        nlp: Language, name: str, patterns: List[Union[re.Pattern, str]]):
    regex = map(lambda x: x if isinstance(x, re.Pattern) else re.compile(x),
                patterns)
    return RegularExpressionMerger(nlp, tuple(regex))


@dataclass
class RegularExpressionEntityMatcher(object):
    """Adds entities based on regular epxressions.

    :see: `Rule matching <https://spacy.io/usage/rule-based-matching>`_
    """
    nlp: Language = field()
    """The NLP model."""

    patterns: List[Tuple[str, List[List[Dict[str, Any]]]]] = field()
    """The patterns given to the :class:`~spacy.matcher.Matcher`."""

    def __post_init__(self):
        self._matchers = []
        self._labels = {}
        for label, patterns in self.patterns:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'label: {label}')
                logger.debug(f'pattern: {patterns}')
            matcher = Matcher(self.nlp.vocab)
            matcher.add(label, patterns, on_match=self._add_event_ent)
            self._labels[id(matcher)] = label
            self._matchers.append(matcher)

    def _add_event_ent(self, matcher, doc, i, matches):
        # Get the current match and create tuple of entity label, start and
        # end.  Append entity to the doc's entity. (Don't overwrite doc.ents!)
        label = self._labels[id(matcher)]
        match_id, start, end = matches[i]
        entity = Span(doc, start, end, label=label)
        doc.ents += (entity,)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'matched <{entity}>: {label}')

        # https://github.com/explosion/spaCy/discussions/4806
        with doc.retokenize() as retokenizer:
            # Iterate over all spans and merge them into one token. This is
            # done after setting the entities – otherwise, it would cause
            # mismatched indices!
            retokenizer.merge(entity)

    def __call__(self, doc: Doc) -> Doc:
        for matcher in self._matchers:
            match: List[Tuple[int, int, int]] = matcher(doc)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'matched: {match}')
                logger.debug(f'doc ents: {doc.ents}')
        return doc


@Language.factory('regexents',
                  default_config={'patterns': [], 'import_file': None})
def create_regex_entities_component(
        nlp: Language, name: str,
        patterns: List[Tuple[str, List[List[Dict[str, Any]]]]],
        import_file: str = None):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'creating regex component for: {name} ({nlp})')
    if import_file is not None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reading file JSON config file: {import_file}')
        with open(import_file) as f:
            add_pats = json.load(f)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'adding to patterns: {add_pats}')
        patterns.extend(add_pats)
    matcher = RegularExpressionEntityMatcher(nlp, patterns)
    return matcher
