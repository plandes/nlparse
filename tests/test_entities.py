from zensols.nlp import FeatureDocument, FeatureDocumentParser
from util import TestBase


class TestEntity(TestBase):
    CONFIG = 'test-resources/entity.conf'

    def test_entity_update_spans(self):
        doc_parser: FeatureDocumentParser = self.fac.instance('doc_parser')
        s = 'Obama was the 44th president of the United States for 8 years.'
        doc: FeatureDocument = doc_parser(s)
        self.assertEqual(1, len(doc.sents))
        self.assertEqual(tuple(s[:-1].split() + ['.']),
                         tuple(doc.norm_token_iter()))
        should = ('PERSON', '-<N>-', '-<N>-', 'ORDINAL', '-<N>-', '-<N>-',
                  'GPE', 'GPE', 'GPE', '-<N>-', 'DATE', 'DATE', '-<N>-')
        self.assertEqual(should, tuple(map(lambda t: t.ent_, doc.token_iter())))
        should = ('Obama', '44th', 'the United States', '8 years')
        self.assertEqual(should, tuple(map(lambda e: e.text, doc.entities)))

        should = ('Obama', 'was', 'the', '44th', 'president', 'of',
                  'the United States', 'the United States', 'the United States',
                  'for', '8 years', '8 years', '.')
        self.assertEqual(should, tuple(
            map(lambda t: s[t.lexspan[0]:t.lexspan[1]], doc.token_iter())))

        self.assertEqual((0, 6, 10, 14, 19, 29, 32, 32, 32, 50, 54, 54, 61),
                         tuple(map(lambda t: t.idx, doc.token_iter())))

        doc.update_entity_spans()

        self.assertEqual(tuple(s[:-1].split() + ['.']), tuple(
            map(lambda t: s[t.lexspan[0]:t.lexspan[1]], doc.token_iter())))

        self.assertEqual(tuple(s[:-1].split() + ['.']), tuple(
            map(lambda t: s[t.lexspan[0]:t.lexspan[1]], doc.token_iter())))

        self.assertEqual((0, 6, 10, 14, 19, 29, 32, 36, 43, 50, 54, 56, 61),
                         tuple(map(lambda t: t.idx, doc.token_iter())))
