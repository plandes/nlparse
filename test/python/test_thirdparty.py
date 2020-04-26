import logging
import unittest
from config import AppConfig
from zensols.config import ImportConfigFactory
from zensols.nlp import LanguageResourceFactory

logger = logging.getLogger(__name__)


class TestThirdParty(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig('stemmer')
        self.fac = LanguageResourceFactory(self.config)

    def test_stemmer(self):
        tnfac = ImportConfigFactory(self.config)
        sent = 'Bobby is fast and runs with dogs, armies, and sheep from the police.'
        lr = self.fac.instance(token_normalizer=tnfac.instance('nonorm_token_normalizer'))
        doc = lr.parse(sent)
        feats = tuple(lr.features(doc))
        self.assertEqual(('Bobby', 'is', 'fast', 'and', 'runs', 'with',
                          'dogs', ',', 'armies', ',', 'and', 'sheep', 'from',
                          'the', 'police', '.'),
                         tuple(map(lambda f: f.norm, feats)))
        self.assertEqual(('Bobby', 'be', 'fast', 'and', 'run', 'with', 'dog',
                          ',', 'army', ',', 'and', 'sheep', 'from', 'the',
                          'police', '.'),
                         tuple(map(lambda f: f.lemma, feats)))
        lr = self.fac.instance(token_normalizer=tnfac.instance('stemmer_token_normalizer'))
        feats = tuple(lr.features(doc))
        self.assertEqual(('bobbi', 'is', 'fast', 'and', 'run', 'with', 'dog',
                          ',', 'armi', ',', 'and', 'sheep', 'from', 'the',
                          'polic', '.'),
                         tuple(map(lambda f: f.norm, feats)))
