from unittest import TestCase
from typing import Tuple
import logging
import re
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import (
    FeatureToken, FeatureSentence, FeatureDocument, FeatureDocumentParser
)
from util import TestBase

logger = logging.getLogger(__name__)


class TestSentenceDecorate(TestBase):
    CONFIG = 'test-resources/decorate.conf'

    def test_decorate_sentences(self):
        if 0:
            print()
            print(self.sent_text)
            print('01234567890123456789012345678901234567890123456789')
        dp: FeatureDocumentParser = self.fac.instance('dec_sent_doc_parser')
        doc: FeatureDocument = dp.parse(self.sent_text)
        should = ('I', 'am', 'a', 'citizen', 'of',
                  'the', 'United', 'States', 'of', 'America', '.')
        self.assertEqual(should, tuple(doc.norm_token_iter()))
        for sti, t in enumerate(doc.tokens):
            st = doc.spacy_doc[sti]
            s = t.lexspan
            self.assertEqual(t.norm, self.sent_text[s[0]:s[1]])
            self.assertEqual(t.norm, st.orth_)
            self.assertEqual(t.idx, st.idx)


class TestDocumentDecorate(TestBase):
    CONFIG = 'test-resources/decorate.conf'

    def setUp(self):
        super().setUp()
        self.two_nl: str = 'Dan throws the ball.  \n\n  He throws it quite often.'

    def _sent_white_space_count(self, s: FeatureSentence) -> int:
        text: bool = re.match(r'^\s+', s.text) is not None or \
            re.match(r'\s+$', s.text) is not None
        tok: bool = False
        if len(s) > 0:
            s[0].is_space or s[-1].is_space
        return 1 if text or tok else 0

    def _sent_empty_count(self, doc) -> Tuple[int, int]:
        ws_toks: int = 0
        empties: int = 0
        for s in doc:
            if len(s) == 0:
                empties += 1
            ws_toks += sum(1 for _ in map(lambda t: 1 if t.is_space else 0, s))
        return ws_toks, empties

    def _assert_whitespace(self, sent: str, doc_parser, should_ws: int):
        doc: FeatureDocument = doc_parser(sent)
        ws: int = sum(map(self._sent_white_space_count, doc.sents))
        self.assertEqual(should_ws, ws)
        return doc

    def test_spacy_version_instra_sentence_left_space(self):
        doc_parser: FeatureDocumentParser = self.fac.instance(
            'default_doc_parser')

        # confirm intra-sentence whitespace behavior, which differs between
        # versions of spacy
        doc = self._assert_whitespace(
            'Dan throws the ball. He throws it quite often.', doc_parser, 0)
        self.assertEqual(2, len(doc.sents))

        doc = self._assert_whitespace(
            'Dan throws the ball.  He throws it quite often.', doc_parser, 1)
        self.assertEqual(2, len(doc.sents))
        self.assertEqual(' He throws it quite often.', doc[1].text)

        doc = self._assert_whitespace(
            ' Dan throws the ball.  He throws it quite often.', doc_parser, 2)
        self.assertEqual(2, len(doc.sents))

        doc = self._assert_whitespace(
            ' Dan throws the ball.  He throws it quite often. ', doc_parser, 2)
        self.assertEqual(' Dan throws the ball.', doc[0].text)
        self.assertEqual(' He throws it quite often.', doc[1].text)

        # test add empty sentences and space
        doc_parser: FeatureDocumentParser = self.fac.instance(
            'default_doc_parser')

        doc = self._assert_whitespace(self.two_nl, doc_parser, 1)
        self.assertEqual('Dan throws the ball.', doc[0].text)
        self.assertEqual(' \n\n  ', doc[1].text)
        self.assertEqual('He throws it quite often.', doc[2].text)
        self.assertEqual((12, 0), self._sent_empty_count(doc))

    def test_empty_sentence_whitespace_stripping(self):
        doc_parser: FeatureDocumentParser = self.fac.instance('strip_sent_doc_parser')
        self._assert_whitespace(
            'Dan throws the ball. He throws it quite often.', doc_parser, 0)
        doc = self._assert_whitespace(
            'Dan throws the ball.  He throws it quite often.', doc_parser, 0)
        self.assertEqual('Dan throws the ball.', doc[0].text)
        self.assertEqual('He throws it quite often.', doc[1].text)

        doc = self._assert_whitespace(
            ' Dan throws the ball.  He throws it quite often. ', doc_parser, 0)
        self.assertEqual('Dan throws the ball.', doc[0].text)
        self.assertEqual('He throws it quite often.', doc[1].text)

    def test_empty_sentence_filtering(self):
        doc_parser: FeatureDocumentParser = self.fac.instance('filter_sent_doc_parser')
        doc = self._assert_whitespace(
            ' Dan throws the ball.  He throws it quite often. ', doc_parser, 2)

        doc = self._assert_whitespace(self.two_nl, doc_parser, 0)
        self.assertEqual(2, len(doc.sents))
        self.assertEqual('Dan throws the ball.', doc[0].text)
        self.assertEqual('He throws it quite often.', doc[1].text)

    def test_both_strip_empty_sentence_filtering(self):
        # most common use case
        doc_parser: FeatureDocumentParser = self.fac.instance('strip_and_filter_sent_doc_parser')
        doc = self._assert_whitespace(
            ' Dan throws the ball.  He throws it quite often. ', doc_parser, 0)
        doc = self._assert_whitespace(self.two_nl, doc_parser, 0)

        self.assertEqual(2, len(doc.sents))
        self.assertEqual('Dan throws the ball.', doc[0].text)
        self.assertEqual('He throws it quite often.', doc[1].text)


class TestDecorateMatchOriginal(TestCase):
    CONFIG = 'test-resources/decorate-original.conf'

    def test_org_spans(self):
        conf = ImportIniConfig('test-resources/decorate-original.conf')
        fac = ImportConfigFactory(conf)
        doc_parser: FeatureDocumentParser = fac.instance('post_doc_parser')
        with open('test-resources/whitespace.txt') as f:
            content = f.read()
        doc: FeatureDocument = doc_parser(content)
        tok: FeatureToken
        for tok in doc.token_iter():
            ts = tok.lexspan
            org = content[ts.begin:ts.end]
            self.assertEqual(org, tok.norm)
