from unittest import TestCase
from zensols.util import APIError
from zensols.config import ImportConfigFactory
from zensols.nlp import LexicalSpan, FeatureDocumentParser, FeatureDocument
from config import AppConfig


class TestOverlap(TestCase):
    def setUp(self):
        fac = ImportConfigFactory(AppConfig(), shared=False)
        doc_parser: FeatureDocumentParser = fac('default_doc_parser')
        self.sent = '\
Dan throws the ball. He throws it quite often.'
#123456789012345678901234567890
        self.doc: FeatureDocument = doc_parser(self.sent)

    def test_complete(self):
        span = LexicalSpan(0, len(self.sent) - 1)
        self.assertTrue(isinstance(self.doc, FeatureDocument))
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual(self.doc.token_len, doc2.token_len)
        self.assertEqual(self.doc.text, doc2.text)
        self.assertEqual(len(self.doc), len(doc2))
        self.assertEqual(self.doc, doc2)

    def test_first(self):
        span = LexicalSpan(0, 19)
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual('Dan throws the ball.', doc2.text)
        self.assertEqual(1, len(doc2))
        self.assertEqual(5, doc2.token_len)
        self.assertEqual('Dan throws the ball .'.split(),
                         list(doc2.norm_token_iter()))

    def test_last(self):
        span = LexicalSpan(21, len(self.sent) - 1)
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual('He throws it quite often.', doc2.text)
        self.assertEqual(1, len(doc2))
        self.assertEqual(6, doc2.token_len)
        self.assertEqual('He throws it quite often .'.split(),
                         list(doc2.norm_token_iter()))

    def test_middle(self):
        span = LexicalSpan(11, 29)
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual('the ball. He throws', doc2.text)
        self.assertEqual(2, len(doc2))
        self.assertEqual(3, len(doc2[0]))
        self.assertEqual(2, len(doc2[1]))
        self.assertEqual('the ball . He throws'.split(),
                         list(doc2.norm_token_iter()))

    def test_split_token_sent_begin(self):
        span = LexicalSpan(12, len(self.sent) - 1)
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual('he ball. He throws it quite often.', doc2.text)
        self.assertEqual(2, len(doc2))
        self.assertEqual(3, len(doc2[0]))
        self.assertEqual(6, len(doc2[1]))
        self.assertEqual('he ball . He throws it quite often .'.split(),
                         list(doc2.norm_token_iter()))

    def test_split_token_sent_end(self):
        span = LexicalSpan(0, 27)
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual('Dan throws the ball. He thro', doc2.text)
        self.assertEqual(2, len(doc2))
        self.assertEqual(5, len(doc2[0]))
        self.assertEqual(2, len(doc2[1]))
        self.assertEqual('Dan throws the ball . He thro'.split(),
                         list(doc2.norm_token_iter()))

    def test_split_token_begin(self):
        span = LexicalSpan(12, 29)
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual('he ball. He throws', doc2.text)
        self.assertEqual(2, len(doc2))
        self.assertEqual(3, len(doc2[0]))
        self.assertEqual(2, len(doc2[1]))
        self.assertEqual('he ball . He throws'.split(),
                         list(doc2.norm_token_iter()))

    def test_split_token_end(self):
        span = LexicalSpan(11, 27)
        doc2 = self.doc.get_overlapping_document(span)
        self.assertEqual('the ball. He thro', doc2.text)
        self.assertEqual(2, len(doc2))
        self.assertEqual(3, len(doc2[0]))
        self.assertEqual(2, len(doc2[1]))
        self.assertEqual('the ball . He thro'.split(),
                         list(doc2.norm_token_iter()))

    def test_narrow(self):
        s1 = LexicalSpan(11, 27)
        s2 = LexicalSpan(5, 30)
        self.assertEqual((11, 27), s1.narrow(s2).astuple)
        self.assertEqual((11, 27), s2.narrow(s1).astuple)

        s1 = LexicalSpan(11, 27)
        s2 = LexicalSpan(13, 30)
        self.assertEqual((13, 27), s1.narrow(s2).astuple)
        self.assertEqual((13, 27), s2.narrow(s1).astuple)

        s1 = LexicalSpan(11, 27)
        s2 = LexicalSpan(13, 20)
        self.assertEqual((13, 20), s1.narrow(s2).astuple)
        self.assertEqual((13, 20), s2.narrow(s1).astuple)

        s1 = LexicalSpan(11, 27)
        s2 = LexicalSpan(13, 31)
        self.assertEqual((13, 27), s1.narrow(s2).astuple)

        s1 = LexicalSpan(2, 5)
        s2 = LexicalSpan(13, 31)
        with self.assertRaisesRegex(APIError, r"^Spans must overlap to be narrowed"):
            self.assertEqual((13, 27), s1.narrow(s2).astuple)
