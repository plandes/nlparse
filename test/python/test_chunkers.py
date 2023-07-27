import logging
from io import StringIO
from zensols.nlp import LexicalSpan, FeatureDocument, FeatureDocumentParser
from zensols.nlp.chunker import ParagraphChunker
from util import TestBase

logger = logging.getLogger(__name__)


class TestFeatureDocParse(TestBase):
    def setUp(self):
        super().setUp()
        self.para_forms = (
            ('first', 'second', '1st'),
            ('second', 'fifth', '2st'),
            ('third', 'eight', '3rd'),
            ('fourth', '11th', '4th'))

    def _make_para(self, pname, sname, cname) -> str:
        return f"""\
{pname.capitalize()} paragraph. {sname.capitalize()} sentence.
Second line {cname} paragraph.\
"""

    def _make_paras(self, n: int, n_newlines: int) -> str:
        sio = StringIO()
        for i, args in enumerate(self.para_forms[:n]):
            if i > 0:
                sio.write('\n' * n_newlines)
            sio.write(self._make_para(*args))
        return sio.getvalue()

    def _make_should(self, pname, sname, cname) -> str:
        return f"""\
<{pname.capitalize()}|paragraph|.>#<{sname.capitalize()}|sentence|.>#\
<Second|line|{cname}|paragraph|.>\
"""

    def _make_shoulds(self, n):
        for args in self.para_forms[:n]:
            yield self._make_should(*args)

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
        doc_strs = tuple(
            map(lambda d: '#'.join(map(lambda s: f'<{s.canonical}>', d)), docs))
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
