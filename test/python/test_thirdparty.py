import logging
import unittest
from config import AppConfig
from zensols.nlp import (
    LanguageResourceFactory,
    TokenNormalizerFactory,
    PorterStemmer
)

logger = logging.getLogger(__name__)


class TestThirdParty(unittest.TestCase):
    def setUp(self):
        self.maxDiff = 999999
        self.config = AppConfig('stemmer')
        self.fac = LanguageResourceFactory(self.config)
        self.lr = self.fac.instance()

    def test_stemmer(self):
        tnfac = TokenNormalizerFactory(self.config)
        sent = 'Bobby is fast and runs with dogs, armies, and sheep from the police.'
        lr = self.fac.instance(token_normalizer=tnfac.instance('nonorm'))
        doc = self.lr.parse(sent)
        feats = tuple(lr.features(doc))
        print(tuple(map(lambda f: f.norm, feats)))
        print(tuple(map(lambda f: f.lemma, feats)))
        # self.assertEqual(('I', 'am', 'a', 'citizen', 'of', 'the United States of America', '.'),
        #                  tuple(map(lambda f: f.norm, feats)))
        lr = self.fac.instance(token_normalizer=tnfac.instance('stemmer'))
        doc = self.lr.parse(sent)
        feats = tuple(lr.features(doc))
        print(tuple(map(lambda f: f.norm, feats)))
