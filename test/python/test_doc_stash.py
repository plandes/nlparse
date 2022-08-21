import unittest
from zensols.config import ImportConfigFactory
from zensols.persist import Stash
from zensols.nlp import FeatureDocument, CachingFeatureDocumentParser
from config import AppConfig


class TestDocStash(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.fac = ImportConfigFactory(self.config)
        self.parser = self.fac('cache_doc_stash')
        self.sent = 'Dan throws the ball.'

    def test_parse(self):
        parser = self.parser
        self.assertEqual(CachingFeatureDocumentParser, type(parser))
        stash = parser.stash
        self.assertTrue(isinstance(stash, Stash))

        self.assertEqual(0, len(stash))
        self.assertEqual(0, len(stash.keys()))

        doc: FeatureDocument = parser(self.sent)
        self.assertEqual(FeatureDocument, type(doc))
        self.assertEqual(1, len(doc.sents))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'),
                         tuple(doc.norm_token_iter()))

        self.assertEqual(1, len(stash.keys()))
        self.assertEqual(1, len(stash))
        self.assertEqual(64, len(next(iter(stash.keys()))))

        doc2: FeatureDocument = parser(self.sent)
        self.assertEqual(id(doc), id(doc2))
