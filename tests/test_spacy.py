from typing import Callable
from util import TestBase
from zensols.nlp import (
    FeatureToken, FeatureSentence, FeatureDocument, FeatureDocumentParser
)


class TestSpacyArtifact(TestBase):
    def setUp(self):
        super().setUp()
        self.doc_parser: FeatureDocumentParser = self.fac.instance('doc_parser')

    def _test_token_clone(self, tok: FeatureToken, clone_fn: Callable):
        self.assertEqual(FeatureToken, type(tok))

        tok2: FeatureToken = clone_fn(tok)
        self.assertEqual(FeatureToken, type(tok2))
        self.assertEqual(tok, tok2)
        self.assertFalse(tok is tok2)
        self.assertEqual(hash(tok), hash(tok2))

        tok2.i += 1
        self.assertNotEqual(tok, tok2)
        self.assertNotEqual(hash(tok), hash(tok2))

        tok2 = clone_fn(tok)
        self.assertEqual(tok, tok2)
        tok2.idx += 1
        self.assertNotEqual(tok, tok2)
        # only i, idx, i_sent are hashed
        self.assertNotEqual(hash(tok), hash(tok2))

        tok2 = clone_fn(tok)
        self.assertEqual(tok, tok2)
        tok2.norm = 'x'
        self.assertNotEqual(tok, tok2)

        if 0:
            print()
            for attr in 'i idx i_sent'.split():
                print(f'{attr}: {getattr(tok, attr)} == {getattr(tok2, attr)}')

        # only i, idx, i_sent are hashed
        self.assertEqual(hash(tok), hash(tok2))

    def _test_sent_clone(self, sent: FeatureSentence, clone_fn: Callable):
        self.assertEqual(FeatureSentence, type(sent))

        sent2: FeatureSentence = clone_fn(sent)
        self.assertEqual(FeatureSentence, type(sent2))
        self.assertEqual(sent, sent2)
        self.assertFalse(sent is sent2)
        self.assertFalse(sent[0] is sent2[0])
        self.assertEqual(hash(sent), hash(sent2))

        sent2[0].i += 1
        self.assertNotEqual(sent, sent2)
        self.assertNotEqual(hash(sent), hash(sent2))

        sent2 = clone_fn(sent)
        self.assertEqual(sent, sent2)
        sent2[0].idx += 1
        self.assertNotEqual(sent, sent2)
        # only i, idx, i_sent are hashed
        self.assertNotEqual(hash(sent), hash(sent2))

        sent2 = clone_fn(sent)
        self.assertEqual(sent, sent2)
        sent2[0].norm = 'x'
        self.assertNotEqual(sent, sent2)

        sent2 = clone_fn(sent)
        self.assertEqual(sent, sent2)

        sent2.tokens = tuple(sent.tokens[:-1])
        self.assertNotEqual(sent, sent2)

        sent2 = clone_fn(sent)
        self.assertEqual(sent, sent2)
        sent2.text = sent.text + 'X'
        self.assertNotEqual(sent, sent2)

    def _test_doc_clone(self, doc: FeatureDocument, clone_fn: Callable):
        self.assertEqual(FeatureDocument, type(doc))

        doc2: FeatureDocument = clone_fn(doc)
        self.assertEqual(FeatureDocument, type(doc2))
        self.assertEqual(doc, doc2)
        self.assertFalse(doc is doc2)
        self.assertFalse(doc[0] is doc2[0])
        self.assertFalse(doc[0][0] is doc2[0][0])
        self.assertEqual(hash(doc), hash(doc2))

        doc2[0][0].i += 1
        self.assertNotEqual(doc, doc2)
        self.assertNotEqual(hash(doc), hash(doc2))

        doc2 = clone_fn(doc)
        self.assertEqual(doc, doc2)
        doc2[0][0].idx += 1
        self.assertNotEqual(doc, doc2)
        # only i, idx, i_doc are hashed
        self.assertNotEqual(hash(doc), hash(doc2))

        doc2 = clone_fn(doc)
        self.assertEqual(doc, doc2)
        doc2[0][0].norm = 'x'
        self.assertNotEqual(doc, doc2)

        doc2 = clone_fn(doc)
        self.assertEqual(doc, doc2)

        doc2[0].tokens = tuple(doc[0].tokens[:-1])
        self.assertNotEqual(doc, doc2)

        doc2 = clone_fn(doc)
        self.assertEqual(doc, doc2)
        self.assertEqual(doc.sents, doc2.sents)
        doc2[0].text = doc.text + 'X'
        self.assertNotEqual(doc.sents, doc2.sents)

        doc2 = clone_fn(doc)
        self.assertEqual(doc, doc2)
        doc2.text = doc.text + 'X'
        self.assertNotEqual(doc, doc2)

        doc2 = clone_fn(doc)
        self.assertEqual(doc, doc2)
        doc2[0].text = doc[0].text + 'X'
        self.assertNotEqual(doc, doc2)

    def test_token_clone(self):
        doc: FeatureDocument = self.doc_parser(self.sent_text2)
        self._test_token_clone(doc.tokens[0], lambda t: t.clone())
        self._test_token_clone(
            doc.tokens[0], lambda t: self.doc_parser(self.sent_text2)[0][0])

    def test_sent_clone(self):
        doc: FeatureDocument = self.doc_parser(self.sent_text2)
        self._test_sent_clone(doc[0], lambda s: s.clone())
        self._test_sent_clone(
            doc[0], lambda _: self.doc_parser(self.sent_text2)[0])
        with self.assertRaises(AssertionError):
            self._test_sent_clone(
                doc[0], lambda s: self.doc_parser(self.sent_text2)[1])

    def test_doc_clone(self):
        doc: FeatureDocument = self.doc_parser(self.sent_text2)
        self._test_doc_clone(doc, lambda d: d.clone())
        self._test_doc_clone(doc, lambda _: self.doc_parser(self.sent_text2))
        with self.assertRaises(AssertionError):
            self._test_doc_clone(doc, lambda _: self.doc_parser(self.sent_text))
