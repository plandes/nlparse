import unittest
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory


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
