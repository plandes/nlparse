import logging
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from util import TestBase

logger = logging.getLogger(__name__)


class TestDecorate(TestBase):
    CONFIG = 'test-resources/decorate.conf'

    def test_decorate(self):
        if 0:
            print()
            print(self.sent_text)
            print('01234567890123456789012345678901234567890123456789')
        dp: FeatureDocumentParser = self.fac.instance('doc_parser')
        doc: FeatureDocument = dp.parse(self.sent_text)
        should = ('I', 'am', 'a', 'citizen', 'of',
                  'the', 'United', 'States', 'of', 'America', '.')
        self.assertEqual(should, tuple(doc.norm_token_iter()))
        for sti, t in enumerate(doc.tokens):
            st = doc.spacy_doc[sti]
            s = t.lexspan
            self.assertEqual(t.norm, self.sent_text[s[0]:s[1]])
            self.assertEqual(t.norm, st.orth_)
            self.assertEqual(t.idx, st.idx)
