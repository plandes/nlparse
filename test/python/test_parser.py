from unittest import TestCase
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocumentParser


class TestOverlap(TestCase):
    def test_parser(self):
        config = ImportIniConfig('test-resources/parser.conf')
        fac = ImportConfigFactory(config)
        doc_parser: FeatureDocumentParser = fac('doc_parser')
        doc = doc_parser('My name is Joseph.')
        for t in doc.token_iter():
            print(t, type(t))
