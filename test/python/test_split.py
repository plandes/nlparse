from typing import List
import unittest
from zensols.config import ImportConfigFactory
from zensols.nlp import (
    LexicalSpan, FeatureDocumentParser, FeatureDocument, FeatureToken
)
from config import AppConfig


class TestTokenMassage(unittest.TestCase):
    def setUp(self):
        self.maxDiff = 999999
        self.config = AppConfig()
        self.fac = ImportConfigFactory(self.config, shared=False)
        self.doc_parser: FeatureDocumentParser = self.fac('default_doc_parser')

    def test_token_split(self):
        sent = '''\
Obama was the 44th president.'''
#  34567890123456789012345678
#         1         2
        doc: FeatureDocument = self.doc_parser(sent)
        tok: FeatureToken = doc.tokens[-2]
        self.assertEqual(tok.norm, 'president')
        stoks: List[FeatureToken] = tok.split([2, 4])

        self.assertEqual(('pr', 'es', 'ident'),
                         tuple(map(lambda t: t.norm, stoks)))

        self.assertEqual(19, tok.idx)
        self.assertEqual(19, stoks[0].idx)
        self.assertEqual(21, stoks[1].idx)
        self.assertEqual(23, stoks[2].idx)

        self.assertEqual(tok.lexspan, LexicalSpan(19, 28))
        self.assertEqual(stoks[0].lexspan, LexicalSpan(19, 21))
        self.assertEqual(stoks[1].lexspan, LexicalSpan(21, 23))
        self.assertEqual(stoks[2].lexspan, LexicalSpan(23, 28))

        attr: str
        for attr in 'text dep ent ent_ i i_sent'.split():
            stok: FeatureToken
            for stok in stoks:
                self.assertEqual(getattr(tok, attr), getattr(stok, attr))

        for attr in 'idx lexspan'.split():
            stok: FeatureToken
            for stok in stoks[1:]:
                self.assertNotEqual(getattr(tok, attr), getattr(stok, attr))
