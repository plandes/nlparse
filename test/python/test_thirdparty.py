import logging
import unittest
from config import AppConfig
from zensols.config import ImportConfigFactory

logger = logging.getLogger(__name__)


class TestThirdParty(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig('stemmer')
        self.fac = ImportConfigFactory(self.config, shared=False)

    def test_stemmer(self):
        tnfac = ImportConfigFactory(self.config)
        sent = 'Bobby is fast and runs with dogs, armies, and sheep from the police.'
        doc_parser = self.fac.instance('doc_parser', token_normalizer=tnfac.instance('nonorm_token_normalizer'))
        doc = doc_parser.parse(sent)
        feats = tuple(doc.norm_token_iter())
        self.assertEqual(('Bobby', 'is', 'fast', 'and', 'runs', 'with',
                          'dogs', ',', 'armies', ',', 'and', 'sheep', 'from',
                          'the', 'police', '.'), feats)
        self.assertEqual(('Bobby', 'be', 'fast', 'and', 'run', 'with', 'dog',
                          ',', 'army', ',', 'and', 'sheep', 'from', 'the',
                          'police', '.'),
                         tuple(map(lambda f: f.lemma_, doc.token_iter())))
        doc_parser = self.fac.instance('doc_parser', token_normalizer=tnfac.instance('stemmer_token_normalizer'))
        doc = doc_parser.parse(sent)
        feats = tuple(doc.norm_token_iter())
        self.assertEqual(('bobbi', 'is', 'fast', 'and', 'run', 'with', 'dog',
                          ',', 'armi', ',', 'and', 'sheep', 'from', 'the',
                          'polic', '.'), feats)
