from typing import Tuple, List
from pathlib import Path
import json
import unittest
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocumentParser, FeatureDocument, FeatureToken


class TestWhitespaceTokenizer(unittest.TestCase):
    RESET_GOLD = False

    def setUp(self):
        config = ImportIniConfig('test-resources/whitespace-tokenizer.conf')
        fac = ImportConfigFactory(config)
        self.doc_parser: FeatureDocumentParser = fac('doc_parser')

    def test_tokenizer(self):
        def feat(attr: str) -> Tuple[str]:
            return list(map(lambda t: str(getattr(t, attr)),
                            doc.token_iter()))

        sent = """`` Target was denied a restraining order against Canvass for a Cause , citing there was no evidence volunteers were threatening or harassing ! This is a win , '' the San Diego group said on its website ."""
        feat_path = Path('test-resources/whitespace-tokenizer.json')
        avoids: List[str] = set('tag shape pos ent'.split())
        attrs: List[str] = FeatureToken.FEATURE_IDS - avoids
        parser = self.doc_parser
        doc: FeatureDocument = parser(sent)
        if self.RESET_GOLD:
            gold = {}
            for attr in attrs:
                gold[attr] = feat(attr)
            with open(feat_path, 'w') as f:
                json.dump(gold, f, indent=4)
        else:
            with open(feat_path) as f:
                gold = json.load(f)
            for attr in attrs:
                self.assertEqual(gold[attr], feat(attr), f'feature: {attr} not equal')
