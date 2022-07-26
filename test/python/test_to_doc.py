from unittest import TestCase
import sys
from spacy.tokens import Doc
from zensols.config import ImportConfigFactory
from zensols.nlp import FeatureDocument
from config import AppConfig


class TestToDoc(TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.fac = ImportConfigFactory(self.config, shared=False)
        self.doc_parser = self.fac('default_doc_parser')
        self.maxDiff = sys.maxsize

    def test_to_doc(self):
        s = """Apple is rumored to bring back the ‘Plus’ moniker after five years, as the last time the company used this branding was back in 2017 when it officially unveiled the iPhone 8 Plus, alongside the regular iPhone 8 and the iPhone X."""
        fdoc: FeatureDocument = self.doc_parser(s)
        sdoc: Doc = self.doc_parser.to_spacy_doc(fdoc)
        self.assertTrue(isinstance(fdoc, FeatureDocument))
        self.assertTrue(isinstance(sdoc, Doc))
        self.assertEqual(list(fdoc.norm_token_iter()), [t.orth_ for t in sdoc])
        self.assertEqual(' '.join(fdoc.norm_token_iter()), sdoc.text.strip())
        self.assertEqual([t.tag_ for t in fdoc.token_iter()],
                         [t.tag_ for t in sdoc])
        self.assertEqual([t.lemma_ for t in fdoc.token_iter()],
                         [t.lemma_ for t in sdoc])
        self.assertEqual([t.dep_ for t in fdoc.token_iter()],
                         [t.dep_ for t in sdoc])

    def test_norm(self):
        ss = ("""Apple will bring back the ‘Plus’ moniker, which is a top seller.""",
              """Apple's phones [have] (the largest) market and doesn't share it.""",
              """I'll see the ives in the morning. I wouldn't be remiss.""",
              """He called: look below! Then a twenty-one headed--dog.""")
        for s in ss:
            self.assertEqual(s, self.doc_parser(s).norm)
