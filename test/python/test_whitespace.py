import unittest
from zensols.nlp import (
    FeatureDocument, WhiteSpaceTokenizerFeatureDocumentParser
)


class TestBase(unittest.TestCase):
    def test_parser(self):
        s = """\
Obama was an American politician who served as the \
44th president of the United States from 2009 to 2017."""
        parser = WhiteSpaceTokenizerFeatureDocumentParser()
        doc: FeatureDocument = parser.parse(s)
        self.assertTrue(isinstance(doc, FeatureDocument))
        self.assertEqual(1, len(doc))
        self.assertEqual(19, len(doc[0]))
        should = ('Obama', 'was', 'an', 'American', 'politician', 'who',
                  'served', 'as', 'the', '44th', 'president', 'of', 'the',
                  'United', 'States', 'from', '2009', 'to', '2017.')
        self.assertEqual(should, tuple(doc.norm_token_iter()))
