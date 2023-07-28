import logging
from zensols.nlp import LexicalSpan, FeatureDocument, FeatureDocumentParser
from zensols.nlp.chunker import ParagraphChunker, ListChunker
from util import TestBase, TestParagraphBase

logger = logging.getLogger(__name__)


class TestParagraphChunker(TestParagraphBase):
    def _test_multiple(self, plen: int, n_newlines: int = 2,
                       span: LexicalSpan = None, span_should: str = None,
                       span_should_idx: int = None):
        parser: FeatureDocumentParser = self.fac.instance(
            'doc_parser_split_ents_keep_space')
        text = self._make_paras(plen, n_newlines)
        if 0:
            print()
            print(text)
        doc: FeatureDocument = parser(text)
        if span is not None:
            sub_doc = doc.get_overlapping_document(span)
            if span_should is not None:
                self.assertEqual(span_should, sub_doc.text)
            pc = ParagraphChunker(
                doc=doc,
                sub_doc=sub_doc,
                char_offset=-1)
        else:
            pc = ParagraphChunker(doc)
        docs = tuple(pc)
        if span is None:
            self.assertEqual(plen, len(docs))
        else:
            self.assertEqual(1, len(docs))
        for doc in docs:
            self.assertEqual(3, len(doc))
        shoulds = tuple(self._make_shoulds(plen))
        doc_strs = self._make_doc_strs(docs)
        if 0:
            for ds in doc_strs:
                print(ds)
            for s in shoulds:
                print(s)
        if span_should_idx is None:
            self.assertEqual(len(shoulds), len(docs))
            for should, doc_str in zip(shoulds, doc_strs):
                self.assertEqual(should, doc_str)
        else:
            should = self._make_should(*self.para_forms[span_should_idx])
            self.assertEqual(should, doc_strs[0])

    def test_singleton(self):
        self._test_multiple(1)

    def test_multi(self):
        self._test_multiple(2)

    def test_massive(self):
        self._test_multiple(3)
        self._test_multiple(4)

    def test_multi_extra_space(self):
        self._test_multiple(2, 3)
        self._test_multiple(2, 4)

    def test_massive_extra_space(self):
        self._test_multiple(3, 3)
        self._test_multiple(3, 4)
        self._test_multiple(4, 3)
        self._test_multiple(4, 4)
        self._test_multiple(4, 5)

    def test_offset(self):
        span = LexicalSpan(62, 121)
        span_should = 'Second paragraph. Fifth sentence.\nSecond line 2st paragraph.'
        self._test_multiple(
            plen=3,
            span=span,
            span_should=span_should,
            span_should_idx=1)


class TestItemChunker(TestBase):
    def test_item_chunking(self):
        text = """\
My list of things to do. These are ordered by necessity:
- finish writing proposal
- write more code
- complete the results section in my large language model paper"""
        doc_parser: FeatureDocumentParser = self.fac.instance(
            'doc_parser_split_ents_keep_space')
        doc: FeatureDocument = doc_parser(text)
        chunker = ListChunker(doc=doc)
        items = tuple(map(lambda s: s.text, chunker()))
        if 0:
            print()
            print('\n'.join(items))
        should = (
            'My list of things to do.These are ordered by necessity:',
            '- finish writing proposal',
            '- write more code',
            '- complete the results section in my large language model paper'
        )
        self.assertEqual(should, items)
