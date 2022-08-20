import unittest
from zensols.config import ImportConfigFactory
from zensols.persist import Stash
from zensols.nlp import FeatureDocument
from config import AppConfig


class TestDocStash(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.fac = ImportConfigFactory(self.config)
        self.doc_stash = self.fac('default_doc_stash')
        self.sent = 'Dan throws the ball.'

    def test_parse(self):
        stash = self.doc_stash
        self.assertTrue(isinstance(stash, Stash))
        self.assertEqual(0, len(stash))
        self.assertEqual(0, len(stash.keys()))

        doc: FeatureDocument = stash.load(self.sent)
        self.assertEqual(FeatureDocument, type(doc))
        self.assertEqual(1, len(doc.sents))
        self.assertEqual(5, doc.token_len)

        self.assertEqual(1, len(stash.keys()))
        self.assertEqual(1, len(stash))
        self.assertEqual(64, len(next(iter(stash.keys()))))
        self.assertTrue(stash.exists(self.sent))
        stash.doc_parser = None
        doc2: FeatureDocument = stash.load(self.sent)
        self.assertEqual(id(doc), id(doc2))
