import logging
import unittest
from zensols.config import ImportConfigFactory
from zensols.nlp import FeatureDocument
from config import AppConfig

logger = logging.getLogger(__name__)


class TestSlice(unittest.TestCase):
    def setUp(self):
        self.maxDiff = 999999
        self.config = AppConfig()
        self.fac = ImportConfigFactory(self.config, shared=False)
        self.doc_parser = self.fac('default_doc_parser')
        self.sent = """Dan throws the ball. He threw it fast. Someone got hurt. They didn't duck."""

    def test_parse(self):
        doc_parser = self.doc_parser
        doc: FeatureDocument = doc_parser(self.sent)
        self.assertEqual(FeatureDocument, type(doc))
        self.assertEqual(4, len(doc))

        s1: FeatureDocument = doc.slice(1, 3)
        self.assertEqual(FeatureDocument, type(s1))
        self.assertEqual(2, len(s1))

        self.assertEqual(doc[1], s1[0])
        self.assertEqual(doc[2], s1[1])

        s2: FeatureDocument = doc.slice(0, 4)
        self.assertEqual(FeatureDocument, type(s2))
        self.assertEqual(4, len(s2))
        self.assertNotEqual(id(doc), id(s2))
        self.assertEqual(doc, s2)
