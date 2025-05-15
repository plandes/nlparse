from typing import List
from util import TestBase
from zensols.nlp import FeatureDocument, FeatureDocumentParser


class TestIndex(TestBase):
    def setUp(self):
        super().setUp()
        self.sent = 'Chapter 1. Once when I was six years old I saw a magnificent picture in a book.'

    def _test_sent_index(self, doc_parser , should_toks: List[str] = None):
        if should_toks is None:
            should_toks = 'Once when I was six years old I saw a'.split()
        doc = doc_parser.parse(self.sent)
        sent = doc[1]
        self.assertEqual(1, len(sent.entities))
        self.assertEqual('six years old', sent.entities[0].text)
        tbs = sent.tokens_by_i_sent
        if 0:
            for k, v in sorted(tbs.items(), key=lambda x: x[0]):
                print(k, v)
        self.assertEqual(should_toks, list(
            map(lambda i: tbs[i].norm, range(len(should_toks)))))

    def test_sent_index(self):
        doc_parser: FeatureDocumentParser
        doc_parser = self.fac.instance('doc_parser_split_ents')
        self._test_sent_index(doc_parser)
        doc_parser = self.fac.instance('doc_parser_no_embed_ents')
        self._test_sent_index(doc_parser)
        doc_parser = self.fac.instance('doc_parser_split_space')
        self._test_sent_index(doc_parser)

    def test_embed(self):
        doc_parser = self.fac.instance('doc_parser_default')
        doc: FeatureDocument = doc_parser.parse(self.sent)
        sent = doc[1]
        tbs = sent.tokens_by_i_sent
        if 0:
            for k, v in sorted(tbs.items(), key=lambda x: x[0]):
                print(k, v)
        self.assertEqual(('Once', 'when', 'I', 'was', 'six years old'),
                         tuple(map(lambda i: tbs[i].norm, range(5))))
        self.assertEqual(('I', 'saw'),
                         tuple(map(lambda i: tbs[i].norm, range(7, 9))))

    def _test_char_index(self, doc_parser):
        doc: FeatureDocument = doc_parser.parse(self.sent)
        sent = doc[1]
        self.assertEqual(1, len(sent.entities))
        self.assertEqual('six years old', sent.entities[0].text)
        tbi = sent.tokens_by_idx
        if 0:
            for k, v in sorted(tbi.items(), key=lambda x: x[0]):
                print(k, v)
        self.assertEqual('Once when I was six years old I saw a'.split(),
                         list(map(lambda i: tbi[i].text,
                                  (11, 16, 21, 23, 27, 31, 37, 41, 43, 47))))

    def test_char_index(self):
        doc_parser: FeatureDocumentParser
        doc_parser = self.fac.instance('doc_parser_split_ents')
        self._test_char_index(doc_parser)
        doc_parser = self.fac.instance('doc_parser_no_embed_ents')
        self._test_char_index(doc_parser)

    def _test_doc_index(self, doc_parser):
        doc: FeatureDocument = doc_parser.parse(self.sent)
        sent = doc[1]
        self.assertEqual(1, len(sent.entities))
        self.assertEqual('six years old', sent.entities[0].text)
        tbi = sent.tokens_by_i
        if 0:
            for k, v in sorted(tbi.items(), key=lambda x: x[0]):
                print(k, v)
        self.assertEqual('Once when I was six years old I saw a'.split(),
                         list(map(lambda i: tbi[i].text, range(3, 13))))

    def test_doc_index(self):
        doc_parser: FeatureDocumentParser
        doc_parser = self.fac.instance('doc_parser_split_ents')
        self._test_doc_index(doc_parser)
        doc_parser = self.fac.instance('doc_parser_no_embed_ents')
        self._test_doc_index(doc_parser)
