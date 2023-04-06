from unittest import TestCase
from zensols.config import ImportConfigFactory
from zensols.nlp import LexicalSpan, FeatureDocumentParser, FeatureDocument
import random
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
        self.assertTrue(s1.narrow(s2) is None)
        self.assertTrue(s2.narrow(s1) is None)

    @staticmethod
    def _map_spans(doc, tups):
        spans = map(lambda x: LexicalSpan(*x), tups)
        return tuple(map(lambda r: '|'.join(map(lambda t: t.text, r)),
                         doc.map_overlapping_tokens(spans)))

    def test_map(self):
        doc = self.doc

#Dan throws the ball. He throws it quite often.
#0123456789012345678901234567890

        self.assertEqual(('Dan',),
                         self._map_spans(doc, ((0, 1),)))
        self.assertEqual(('Dan',),
                         self._map_spans(doc, ((0, 2),)))
        self.assertEqual(('Dan',),
                         self._map_spans(doc, ((0, 3),)))
        self.assertEqual(('Dan|throws',),
                         self._map_spans(doc, ((0, 9),)))
        self.assertEqual(('throws',),
                         self._map_spans(doc, ((3, 9),)))

        self.assertEqual(doc.tokens[0].lexspan, LexicalSpan(0, 3))
        self.assertEqual('Dan', self.sent[0:3])
        self.assertEqual(' throws', self.sent[3:10])
        self.assertEqual(('throws',),
                         self._map_spans(doc, ((3, 10),)))

        self.assertEqual('throws', self.sent[4:10])
        self.assertEqual(doc.tokens[1].lexspan, LexicalSpan(4, 10))
        self.assertEqual(('throws',),
                         self._map_spans(doc, ((4, 10),)))
        self.assertEqual(('throws|the',),
                         self._map_spans(doc, ((4, 11),)))

        self.assertEqual(('Dan', 'the|ball|.|He|throws'),
                         self._map_spans(doc, ((0, 1), (11, 29))))
        self.assertEqual(('Dan', 'the|ball|.|He|throws'),
                         self._map_spans(doc, ((0, 2), (11, 29))))
        self.assertEqual(('Dan', 'the|ball|.|He|throws'),
                         self._map_spans(doc, ((1, 2), (11, 29))))
        self.assertEqual(('Dan', 'quite|often|.'),
                         self._map_spans(doc, ((1, 2), ((34, 100)))))
        self.assertEqual(('Dan', 'He|throws', 'quite|often|.'),
                         self._map_spans(doc, ((1, 2), (21, 29), ((34, 100)))))
        self.assertEqual(('Dan', 'He|throws', 'quite|often|.'),
                         self._map_spans(doc, ((1, 2), (21, 30), ((34, 100)))))
        self.assertEqual(('Dan', 'He|throws|it', 'quite|often|.'),
                         self._map_spans(doc, ((1, 2), (21, 31), ((34, 100)))))

    def test_set(self):
        doc = self.doc
        self.assertEqual(doc.token_len, len(set(doc.tokens)))

    def test_sort(self):
        doc = self.doc
        ordered = list(doc.tokens)
        for _ in range(5):
            toks = list(doc.tokens)
            random.shuffle(toks)
            self.assertEqual(ordered, sorted(toks))
