from typing import Tuple
from zensols.nlp import (
    LexicalSpan, TokenContainer, FeatureToken,
    FeatureDocumentParser, FeatureDocument, FeatureSpan
)
from util import TestParagraphBase


class TestReIndex(TestParagraphBase):
    def setUp(self):
        super().setUp()
        self.doc_parser: FeatureDocumentParser = self.fac.instance('doc_parser')

    def _create_docs(self) -> Tuple[FeatureDocument, FeatureDocument]:
        doc: FeatureDocument = self.doc_parser(self._make_paras(3))
        subdoc = doc.get_overlapping_document(LexicalSpan(62, 121))
        return doc, subdoc

    def _indexes(self, cont: TokenContainer) -> \
            Tuple[Tuple[int], Tuple[int], Tuple[LexicalSpan]]:
        iz: Tuple[int] = tuple(map(lambda t: t.i, cont.token_iter()))
        idxs: Tuple[int] = tuple(map(lambda t: t.idx, cont.token_iter()))
        sent_is = tuple(map(lambda t: t.sent_i, cont.tokens))
        spans: Tuple[LexicalSpan] = tuple(
            map(lambda t: t.lexspan.astuple, cont.token_iter()))
        return iz, idxs, sent_is, spans

    def _write(self, text, iz, idxs, sent_is, spans):
        print()
        print('text', text)
        print('i', iz)
        print('idx', idxs)
        print('sent_i', sent_is)
        print('spans', spans)

    def test_index(self):
        doc, subdoc = self._create_docs()
        iz, idxs, sent_is, spans = self._indexes(subdoc)

        if 0:
            self._write(doc.text, iz, idxs, sent_is, spans)

        should = (13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24)
        self.assertEqual(should, iz)

        should = (62, 69, 78, 80, 86, 94, 96, 103, 108, 112, 121)
        self.assertEqual(should, idxs)

        should = (3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5)
        self.assertEqual(should, sent_is)

        should = ((62, 68), (69, 78), (78, 79), (80, 85), (86, 94), (94, 95),
                  (96, 102), (103, 107), (108, 111), (112, 121), (121, 122))
        self.assertEqual(should, spans)

    def test_reindex_sent(self):
        doc = self._create_docs()[1]
        sent = doc[2]
        iz, idxs, sent_is, spans = self._indexes(sent)

        #self._write(sent.text, iz, idxs, sent_is, spans)
        self.assertEqual((20, 21, 22, 23, 24), iz)
        self.assertEqual((96, 103, 108, 112, 121), idxs)
        self.assertEqual((5, 5, 5, 5, 5), sent_is)
        self.assertEqual(((96, 102), (103, 107), (108, 111), (112, 121), (121, 122)), spans)

        doc.reindex()
        iz, idxs, sent_is, spans = self._indexes(sent)
        #self._write(sent.text, iz, idxs, sent_is, spans)
        self.assertEqual((7, 8, 9, 10, 11), iz)
        self.assertEqual((34, 41, 46, 50, 59), idxs)
        self.assertEqual((2, 2, 2, 2, 2), sent_is)
        self.assertEqual(((34, 40), (41, 45), (46, 49), (50, 59), (59, 60)), spans)

        should = '(<Second>, <2st>)'
        self.assertEqual(should, str(sent.entities))

    def test_reindex_sent_reftok(self):
        doc = self._create_docs()[1]
        sent = doc[2]
        iz, idxs, sent_is, spans = self._indexes(sent)

        #self._write(sent.text, iz, idxs, sent_is, spans)
        self.assertEqual((20, 21, 22, 23, 24), iz)
        self.assertEqual((96, 103, 108, 112, 121), idxs)
        self.assertEqual((5, 5, 5, 5, 5), sent_is)
        self.assertEqual(((96, 102), (103, 107), (108, 111), (112, 121), (121, 122)), spans)

        doc.reindex(sent[0])
        iz, idxs, sent_is, spans = self._indexes(sent)
        #self._write(sent.text, iz, idxs, sent_is, spans)
        self.assertEqual((0, 1, 2, 3, 4), iz)
        self.assertEqual((0, 7, 12, 16, 25), idxs)
        self.assertEqual((0, 0, 0, 0, 0), sent_is)
        self.assertEqual(((0, 6), (7, 11), (12, 15), (16, 25), (25, 26)), spans)

        should = '(<Second>, <2st>)'
        self.assertEqual(should, str(sent.entities))

    def test_reindex_doc(self):
        doc = self._create_docs()[1]
        doc.reindex()
        iz, idxs, sent_is, spans = self._indexes(doc)
        ents: Tuple[FeatureSpan, ...] = doc.entities

        if 0:
            self._write(doc.text, iz, idxs, sent_is, spans)

        should = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
        self.assertEqual(should, iz)

        should = (0, 7, 16, 18, 24, 32, 34, 41, 46, 50, 59)
        self.assertEqual(should, idxs)

        should = (0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2)
        self.assertEqual(should, sent_is)

        should = ((0, 6), (7, 16), (16, 17), (18, 23), (24, 32), (32, 33),
                  (34, 40), (41, 45), (46, 49), (50, 59), (59, 60))
        self.assertEqual(should, spans)

        should = '(<Second>, <Fifth>, <Second>, <2st>)'
        self.assertEqual(should, str(ents))
