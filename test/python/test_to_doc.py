from spacy.tokens import Doc
from zensols.nlp import FeatureDocument
from util import TestBase


class TestToDoc(TestBase):
    def setUp(self):
        super().setUp()
        self.parsers = 'doc_parser_default doc_parser_split_ents doc_parser_no_embed_ents doc_parser_split_space'.split()

    def _test_to_doc(self, doc_parser):
        s = """Apple is rumored to bring back the ‘Plus’ moniker after five years, as the last time the company used this branding was back in 2017 when it officially unveiled the iPhone 8 Plus, alongside the regular iPhone 8 and the iPhone X."""
        fdoc: FeatureDocument = doc_parser(s)
        sdoc: Doc = doc_parser.to_spacy_doc(fdoc)
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

    def test_to_doc(self):
        for name in self.parsers:
            self._test_to_doc(self.fac(name))

    def _test_norm(self, doc_parser):
        ss = ("""Apple will bring back the ‘Plus’ moniker, which is a top seller.""",
              """Apple's phones [have] (the largest) market and doesn't share it.""",
              """I'll see the ives in the morning. I wouldn't be remiss.""",
              """He called: look below! Then a twenty-one headed--dog.""",
              ("""Barack Hussein Obama II is an American politician who served as """ +
               """the 44th president of the United States from 2009 to 2017. """ +
               """A member of the Democratic Party, he was the first """ +
               """African-American president of the United States."""))
        for s in ss:
            self.assertEqual(s, doc_parser(s).norm)

    def test_norm(self):
        for name in self.parsers:
            self._test_norm(self.fac(name))
