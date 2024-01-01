import logging
import unittest
from spacy.tokens.doc import Doc
from zensols.config import ImportConfigFactory, ImportIniConfig
from zensols.nlp import FeatureDocument, FeatureDocumentParser

logger = logging.getLogger(__name__)


class TestParse(unittest.TestCase):
    def _parser(self, conf, s):
        config = ImportIniConfig(f'test-resources/{conf}-comp.conf')
        fac = ImportConfigFactory(config)
        doc_parser: FeatureDocumentParser = fac('doc_parser')
        return doc_parser(s)

    def test_regex(self):
        text: str = 'A [**masked**] test with an <angle tok> and {curly tok}.'
        fd: FeatureDocument = self._parser('regex', text)
        doc: Doc = fd.spacy_doc
        ents = doc.ents
        self.assertEqual(3, len(ents))
        self.assertEqual('[**masked**]', ents[0].orth_)
        self.assertEqual('MASK', ents[0].label_)

    def test_pattern(self):
        text: str = 'John <registered> her on [**01-01-2020**] becuase <they> were <in> Chicago.'
        fd: FeatureDocument = self._parser('pat', text)
        doc: Doc = fd.spacy_doc
        self.assertEqual(5, len(doc.ents))
        self.assertEqual('John <registered> [**01-01-2020**] <in> Chicago'.split(),
                         list(map(lambda e: e.orth_, doc.ents)))
        self.assertEqual('PERSON MASK_VERB MASK_DATE MASK_VERB GPE'.split(),
                         list(map(lambda e: e.label_, doc.ents)))

    def test_split(self):
        text: str = 'A strange ID number is 123abc.'
        self.assertEqual(
            ('A', 'strange', 'ID', 'number', 'is', '123abc', '.'),
            tuple(self._parser('regex', text).norm_token_iter()))
        self.assertEqual(
            ('A', 'strange', 'ID', 'number', 'is', '123', 'abc', '.'),
            tuple(self._parser('regex-split', text).norm_token_iter()))
