import logging
import unittest
from zensols.actioncli import ExtendedInterpolationConfig
from zensols.nlp import (
    LanguageResourceFactory,
    LanguageResource,
    DocUtil,
    TokenNormalizerFactory,
)

logger = logging.getLogger(__name__)


def rec_sort(x):
    if isinstance(x, list) or isinstance(x, tuple):
        x = sorted(list(map(rec_sort, x)))
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = rec_sort(v)
        x = sorted(x.items())
    return x


class AppConfig(ExtendedInterpolationConfig):
    SECTION = 'test'

    def __init__(self):
        super(AppConfig, self).__init__(
            config_file='test-resources/nlparse.conf', default_expect=True)

    @property
    def parse_path(self):
        return self.get_option_path('parse_test', self.SECTION)

    @property
    def feature_path(self):
        return self.get_option_path('feature_test', self.SECTION)


class TestParse(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.fac = LanguageResourceFactory(self.config)
        self.lr = self.fac.instance()

    def test_parse(self):
        lr = self.lr
        self.assertTrue(isinstance(lr, LanguageResource))
        doc = lr.parse('Dan throws the ball.')
        res = DocUtil.to_json(doc)
        with open(self.config.parse_path) as f:
            c = eval(f.read())
        self.assertEqual(rec_sort(c), rec_sort(res))

    def test_feature(self):
        lr = self.lr
        doc = lr.parse('Dan throws the ball.')
        tnfac = TokenNormalizerFactory(self.config)
        tn = tnfac.instance()
        self.assertEqual('TokenNormalizer: embed=True, normalize: True remove first stop: False', str(tn))
        res = tuple(map(lambda x: x.string_features, lr.features(doc, tn)))
        with open(self.config.feature_path) as f:
            c = eval(f.read())
        self.assertEqual(rec_sort(c), rec_sort(res))
        tn = tnfac.instance('nonorm')
        res = tuple(map(lambda x: x.norm, lr.features(doc, tn)))
        self.assertEqual(('Dan', 'throws', 'the', 'ball', '.'), res)

    def test_map(self):
        tnfac = TokenNormalizerFactory(self.config)
        doc = self.lr.parse('I am a citizen of the United States of America.')
        self.assertEqual(('citizen', 'the United States of America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter'))))
        self.assertEqual(('am', 'citizen', 'of', 'the United States of America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter_pron'))))
        self.assertEqual(('i', 'am', 'a', 'citizen', 'of', 'the united states of america', '.'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_lower'))))
        self.assertEqual(('citizen', 'United', 'States', 'America'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_embed'))))
        self.assertEqual(('citizen', 'the_united_states_of_america'),
                         tuple(self.lr.normalized_tokens(
                             doc, tnfac.instance('map_filter_subs'))))
