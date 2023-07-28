from typing import Tuple
import unittest
from io import StringIO
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.nlp import FeatureDocument


class TestBase(unittest.TestCase):
    CONFIG = 'test-resources/features.conf'

    def setUp(self):
        path = self.CONFIG
        config = AppConfig(path)
        self.fac = ImportConfigFactory(config, shared=True)
        self.sent_text = 'I am a citizen of the United States of America.'
        self.def_parse = ('I', 'am', 'a', 'citizen', 'of',
                          'the United States of America', '.')
        self.sent_text2 = self.sent_text + " My name is Paul Landes."


class TestParagraphBase(TestBase):
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

    def _make_paras(self, n: int, n_newlines: int = 2) -> str:
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

    def _make_doc_strs(self, docs: Tuple[FeatureDocument]) -> Tuple[str]:
        return tuple(map(lambda d: '#'.join(
            map(lambda s: f'<{s.canonical}>', d)), docs))
